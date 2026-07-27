# Shared frame matrix

`kraken_core.frame_matrix` contains the domain-neutral matrix contracts, the
virtualized Qt widget and a pluggable thumbnail cache. Plugin domain objects
must be translated to `MatrixItem`; they must not be imported by the shared
package.

## Data source

```python
from kraken_core.frame_matrix import MatrixItem, MatrixViewportResult


class PluginMatrixSource:
    def load_viewport(self, request, cancellation=None):
        records = repository.visible_records(request.bounds, request.lod)
        return MatrixViewportResult(
            request=request,
            items=tuple(
                MatrixItem(
                    key=record.key,
                    x=record.x,
                    y=record.y,
                    status=record.status,
                    label=record.name,
                )
                for record in records
            ),
            source_revision=repository.revision,
        )
```

The source is invoked on a worker thread. It should honour the cancellation
token where practical and return only the requested viewport.

## Thumbnail storage

Matrix code receives a `ThumbnailStore`; it never selects a persistence
technology itself.

```python
from kraken_core.frame_matrix import StoreNamespace, ThumbnailStoreFactory

store = ThumbnailStoreFactory().create("sqlite:///D:/cache/kraken-thumbnails")
store.open(StoreNamespace(plugin="contour", project="project-42"))
```

Built-in URIs:

- `memory://` — process-local, non-persistent;
- `sqlite:///path` — BLOBs distributed over at most 64 lazy SQLite shards;
- `files:///path` — explicit hash-partitioned thumbnail files.

Third-party adapters register a callable in the
`kraken.thumbnail_stores` entry-point group. The callable receives the URI
location and returns an object implementing `ThumbnailStore`.

Store capabilities are discovered with `StoreCapability`. Coordinators must
fall back to single-item operations when an adapter does not advertise batch
operations, and cache failures must degrade to a cache miss.

An archive adapter should use append-only segments, a separately replaceable
key-to-offset index, tombstones, recovery of an incomplete final segment and
background compaction. ZIP or TAR details do not belong in the public port.

## Kraken configuration

Kraken uses the SQLite adapter by default. Select another adapter with:

```powershell
kraken-hub --thumbnail-store-uri files:///D:/cache/kraken-thumbnails
```

The same value can be supplied through `KRAKEN_THUMBNAIL_STORE_URI`.
Persistent thumbnails are a disposable cache and are not included in project
backups.
