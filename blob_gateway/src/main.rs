use std::{
    io,
    net::SocketAddr,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Request, State},
    http::{
        HeaderMap, Response, StatusCode,
        header::{
            ACCEPT_RANGES, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, ETAG, RANGE,
        },
    },
    response::IntoResponse,
    routing::{get, put},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::Parser;
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter};
use tokio_util::io::ReaderStream;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Parser)]
#[command(
    name = "KrakenBlobGateway",
    about = "Kraken high-throughput immutable blob data plane"
)]
struct Args {
    #[arg(
        long,
        env = "KRAKEN_BLOB_GATEWAY_BIND",
        default_value = "127.0.0.1:8081"
    )]
    bind: SocketAddr,

    #[arg(long, env = "KRAKEN_BLOB_ROOT")]
    blob_root: PathBuf,

    #[arg(long, env = "KRAKEN_BLOB_GATEWAY_SECRET")]
    secret: String,

    #[arg(long, env = "KRAKEN_BLOB_GATEWAY_TLS_CERT")]
    tls_cert: Option<PathBuf>,

    #[arg(long, env = "KRAKEN_BLOB_GATEWAY_TLS_KEY")]
    tls_key: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    object_root: PathBuf,
    staging_root: PathBuf,
    secret: Arc<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
struct TicketClaims {
    v: u8,
    op: String,
    digest: String,
    size: u64,
    exp: u64,
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

#[derive(Debug, Serialize)]
struct PutResult {
    sha256: String,
    size_bytes: u64,
    already_existed: bool,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: &'static str,
}

impl ApiError {
    const fn new(status: StatusCode, message: &'static str) -> Self {
        Self { status, message }
    }

    fn io(error: io::Error) -> Self {
        eprintln!("blob I/O error: {error}");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "blob storage operation failed",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.secret.len() < 32 {
        eprintln!("KRAKEN_BLOB_GATEWAY_SECRET must contain at least 32 bytes");
        std::process::exit(2);
    }

    let object_root = args.blob_root.clone();
    let staging_root = args.blob_root.join(".gateway-staging");
    if let Err(error) = tokio::fs::create_dir_all(&object_root).await {
        eprintln!("cannot create blob root {}: {error}", object_root.display());
        std::process::exit(2);
    }
    if let Err(error) = tokio::fs::create_dir_all(&staging_root).await {
        eprintln!(
            "cannot create staging root {}: {error}",
            staging_root.display()
        );
        std::process::exit(2);
    }
    if let Err(error) = cleanup_staging(&staging_root).await {
        eprintln!("cannot clean stale Blob Gateway uploads: {error}");
    }

    let state = AppState {
        object_root,
        staging_root,
        secret: Arc::new(args.secret.into_bytes()),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route(
            "/v1/blobs/{digest}",
            put(put_blob).get(get_blob).head(head_blob),
        )
        .with_state(state);

    println!("Kraken Blob Gateway listening on {}", args.bind);
    let result = match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => serve_tls(args.bind, app, cert, key).await,
        (None, None) => serve_plain(args.bind, app).await,
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "both TLS certificate and key are required",
        )
        .into()),
    };
    if let Err(error) = result {
        eprintln!("Kraken Blob Gateway failed: {error}");
        std::process::exit(1);
    }
}

async fn serve_plain(bind: SocketAddr, app: Router) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn serve_tls(
    bind: SocketAddr,
    app: Router,
    certificate: PathBuf,
    key: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(certificate, key).await?;
    let handle = axum_server::Handle::new();
    let shutdown = handle.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown.graceful_shutdown(Some(std::time::Duration::from_secs(5)));
    });
    axum_server::bind_rustls(bind, config)
        .handle(handle)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

async fn cleanup_staging(staging_root: &FsPath) -> io::Result<()> {
    let mut entries = tokio::fs::read_dir(staging_root).await?;
    while let Some(entry) = entries.next_entry().await? {
        let metadata = tokio::fs::symlink_metadata(entry.path()).await?;
        if !metadata.file_type().is_file() {
            continue;
        }
        let stale = metadata
            .modified()
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= std::time::Duration::from_secs(24 * 60 * 60));
        if stale {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "service": "kraken-blob-gateway", "version": 1}))
}

async fn put_blob(
    State(state): State<AppState>,
    Path(digest): Path<String>,
    headers: HeaderMap,
    request: Request,
) -> Result<impl IntoResponse, ApiError> {
    validate_digest(&digest)?;
    let claims = authorize(&state, &headers, "upload", &digest)?;
    let declared = content_length(&headers)?;
    if declared != claims.size {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Content-Length does not match upload ticket",
        ));
    }

    let temporary = state
        .staging_root
        .join(format!("{}.upload", Uuid::new_v4()));
    let target = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .await
        .map_err(ApiError::io)?;
    let mut target = BufWriter::with_capacity(4 * 1024 * 1024, target);
    let mut received = 0_u64;
    let mut hasher = Sha256::new();
    let mut stream = request.into_body().into_data_stream();
    let result = async {
        while let Some(next) = stream.next().await {
            let bytes =
                next.map_err(|_| ApiError::new(StatusCode::BAD_REQUEST, "request body failed"))?;
            received = received.checked_add(bytes.len() as u64).ok_or_else(|| {
                ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "upload is too large")
            })?;
            if received > claims.size {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "upload exceeds declared size",
                ));
            }
            target.write_all(&bytes).await.map_err(ApiError::io)?;
            hasher.update(&bytes);
        }
        target.flush().await.map_err(ApiError::io)?;
        target.get_ref().sync_all().await.map_err(ApiError::io)?;
        drop(target);

        if received != claims.size {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "uploaded size does not match ticket",
            ));
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != digest {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "uploaded SHA-256 does not match ticket",
            ));
        }
        let already_existed = commit_blob(&state, &temporary, &digest, received).await?;
        Ok(PutResult {
            sha256: digest,
            size_bytes: received,
            already_existed,
        })
    }
    .await;
    let _ = tokio::fs::remove_file(&temporary).await;
    result.map(|value| (StatusCode::OK, Json(value)))
}

async fn get_blob(
    State(state): State<AppState>,
    Path(digest): Path<String>,
    headers: HeaderMap,
) -> Result<Response<Body>, ApiError> {
    serve_blob(&state, &headers, &digest, false).await
}

async fn head_blob(
    State(state): State<AppState>,
    Path(digest): Path<String>,
    headers: HeaderMap,
) -> Result<Response<Body>, ApiError> {
    serve_blob(&state, &headers, &digest, true).await
}

async fn serve_blob(
    state: &AppState,
    headers: &HeaderMap,
    digest: &str,
    head_only: bool,
) -> Result<Response<Body>, ApiError> {
    validate_digest(digest)?;
    let claims = authorize(state, headers, "download", digest)?;
    let path = blob_path(state, digest);
    let mut file = tokio::fs::File::open(&path).await.map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            ApiError::new(StatusCode::NOT_FOUND, "blob was not found")
        } else {
            ApiError::io(error)
        }
    })?;
    let metadata = file.metadata().await.map_err(ApiError::io)?;
    if !metadata.is_file() || metadata.len() != claims.size {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "blob size does not match download ticket",
        ));
    }

    let (start, end, status) = match headers.get(RANGE) {
        None => (0, claims.size.saturating_sub(1), StatusCode::OK),
        Some(value) => {
            let text = value.to_str().map_err(|_| {
                ApiError::new(StatusCode::RANGE_NOT_SATISFIABLE, "invalid byte range")
            })?;
            let (start, end) = parse_range(text, claims.size)?;
            (start, end, StatusCode::PARTIAL_CONTENT)
        }
    };
    let response_length = if claims.size == 0 { 0 } else { end - start + 1 };
    if start > 0 {
        file.seek(std::io::SeekFrom::Start(start))
            .await
            .map_err(ApiError::io)?;
    }
    let body = if head_only {
        Body::empty()
    } else {
        Body::from_stream(ReaderStream::with_capacity(
            file.take(response_length),
            1024 * 1024,
        ))
    };
    let mut response = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/octet-stream")
        .header(CONTENT_LENGTH, response_length)
        .header(ACCEPT_RANGES, "bytes")
        .header(ETAG, format!("\"sha256:{digest}\""));
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(
            CONTENT_RANGE,
            format!("bytes {start}-{end}/{}", claims.size),
        );
    }
    response
        .body(body)
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "cannot build response"))
}

fn authorize(
    state: &AppState,
    headers: &HeaderMap,
    operation: &str,
    digest: &str,
) -> Result<TicketClaims, ApiError> {
    let token = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "blob ticket is required"))?;
    let (payload_text, signature_text) = token
        .split_once('.')
        .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "invalid blob ticket"))?;
    let signature = URL_SAFE_NO_PAD
        .decode(signature_text)
        .map_err(|_| ApiError::new(StatusCode::UNAUTHORIZED, "invalid blob ticket"))?;
    let mut mac = HmacSha256::new_from_slice(&state.secret)
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "invalid gateway secret"))?;
    mac.update(payload_text.as_bytes());
    mac.verify_slice(&signature)
        .map_err(|_| ApiError::new(StatusCode::UNAUTHORIZED, "invalid blob ticket"))?;
    let payload = URL_SAFE_NO_PAD
        .decode(payload_text)
        .map_err(|_| ApiError::new(StatusCode::UNAUTHORIZED, "invalid blob ticket"))?;
    let claims: TicketClaims = serde_json::from_slice(&payload)
        .map_err(|_| ApiError::new(StatusCode::UNAUTHORIZED, "invalid blob ticket"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "system clock is invalid"))?
        .as_secs();
    if claims.v != 1 || claims.op != operation || claims.digest != digest || claims.exp < now {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "blob ticket is expired or does not match request",
        ));
    }
    Ok(claims)
}

fn validate_digest(digest: &str) -> Result<(), ApiError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "digest must be a lowercase SHA-256 value",
        ))
    }
}

fn content_length(headers: &HeaderMap) -> Result<u64, ApiError> {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| ApiError::new(StatusCode::LENGTH_REQUIRED, "Content-Length is required"))
}

fn blob_path(state: &AppState, digest: &str) -> PathBuf {
    state
        .object_root
        .join(&digest[..2])
        .join(&digest[2..4])
        .join(digest)
}

async fn commit_blob(
    state: &AppState,
    temporary: &FsPath,
    digest: &str,
    size: u64,
) -> Result<bool, ApiError> {
    let final_path = blob_path(state, digest);
    let parent = final_path
        .parent()
        .ok_or_else(|| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "invalid blob path"))?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(ApiError::io)?;
    match tokio::fs::hard_link(temporary, &final_path).await {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            validate_existing(&final_path, size).await?;
            Ok(true)
        }
        Err(error) => {
            eprintln!(
                "hard-link commit failed for {}: {error}",
                final_path.display()
            );
            Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "blob filesystem must support atomic hard-link commits",
            ))
        }
    }
}

async fn validate_existing(path: &FsPath, size: u64) -> Result<(), ApiError> {
    let metadata = tokio::fs::symlink_metadata(path)
        .await
        .map_err(ApiError::io)?;
    if !metadata.file_type().is_file() || metadata.len() != size {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "existing blob does not match uploaded size",
        ));
    }
    Ok(())
}

fn parse_range(value: &str, size: u64) -> Result<(u64, u64), ApiError> {
    if size == 0 {
        return Err(ApiError::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "empty blob has no byte ranges",
        ));
    }
    let range = value
        .strip_prefix("bytes=")
        .filter(|value| !value.contains(','))
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::RANGE_NOT_SATISFIABLE,
                "only one byte range is supported",
            )
        })?;
    let (start_text, end_text) = range
        .split_once('-')
        .ok_or_else(|| ApiError::new(StatusCode::RANGE_NOT_SATISFIABLE, "invalid byte range"))?;
    let (start, end) = if start_text.is_empty() {
        let suffix = end_text
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                ApiError::new(StatusCode::RANGE_NOT_SATISFIABLE, "invalid suffix range")
            })?;
        let length = suffix.min(size);
        (size - length, size - 1)
    } else {
        let start = start_text.parse::<u64>().map_err(|_| {
            ApiError::new(
                StatusCode::RANGE_NOT_SATISFIABLE,
                "invalid byte range start",
            )
        })?;
        let end = if end_text.is_empty() {
            size - 1
        } else {
            end_text.parse::<u64>().map_err(|_| {
                ApiError::new(StatusCode::RANGE_NOT_SATISFIABLE, "invalid byte range end")
            })?
        };
        (start, end.min(size - 1))
    };
    if start >= size || end < start {
        return Err(ApiError::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "byte range is outside the blob",
        ));
    }
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_normal_open_and_suffix_ranges() {
        assert_eq!(parse_range("bytes=10-19", 100).unwrap(), (10, 19));
        assert_eq!(parse_range("bytes=90-", 100).unwrap(), (90, 99));
        assert_eq!(parse_range("bytes=-10", 100).unwrap(), (90, 99));
        assert!(parse_range("bytes=100-", 100).is_err());
        assert!(parse_range("bytes=1-2,4-5", 100).is_err());
    }
}
