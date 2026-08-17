# Kraken Blob Gateway

Высокопроизводительный файловый контур Kraken. Он принимает и отдаёт только
неизменяемые SHA-256-объекты по краткоживущим HMAC-разрешениям, которые выдаёт
основной Kraken Server. PostgreSQL и пользовательские пароли gateway не получает.

```powershell
$env:KRAKEN_BLOB_ROOT = "D:\Kraken\blobs"
$env:KRAKEN_BLOB_GATEWAY_SECRET = "a-secret-with-at-least-32-bytes"
cargo run --release --manifest-path blob_gateway\Cargo.toml -- --bind 127.0.0.1:8081
```

Публичный сетевой доступ разрешён только через HTTPS. Для локальной разработки
допустим HTTP на loopback-интерфейсе.
