# Matrix tiles: prepared in the project folder, delivered in blocks

## Intent

Zooming a layer matrix is slow. The server is local, so the network is not the
cause. Measurement (`matrix_tile_timing`, 2026-11-24 17:32–17:34):

- The client sent 4553 tile requests; 4517 (99%) were "not ready", 36 ready.
  The same tiles were asked ~125 times each (retry every 0.75–1.5 s).
- A request is cheap: 10–60 ms on the client, 0.2–3 ms to read the tile on the
  server, the rest is auth and HTTP handling.
- The single builder thread built 4×4-frame urgent tiles one at a time,
  0.5–0.7 s each, reading every frame from storage (`previews_cached=0`):
  ~10–13 ms read + ~20–24 ms shrink per frame. Urgent tiles waited 10.5 s
  (median) and up to 17.9 s in the queue.
- After 17:32:31 no urgent tile was built for two minutes, while the full build
  of a new set kept going. The exact cause of that gap is not confirmed.
- Storage holds 7 tile sets, each with its own full copy of 4900 previews. Every
  new key (layer + image + vector representation) re-reads and shrinks all
  4900 sources (~4 min). A full build of 4900 frames took 285 s.

So time goes to waiting for tiles that are rebuilt per key, by one thread, with
urgent work stuck behind full builds; not to transport.

Intended mechanism (user): the server prepares every tile of a layer ahead of
time, stores them, never recomputes them for an unchanged layer; the client asks
for a level and a range and gets the ready tiles in one block.

## Goals

- Tiles and previews live in **`<source_project_dir>/tile`** of the project
  (decided).
- One preview per frame image for the whole project, shared by every tile set.
  Previews are keyed by image content, so a new layer, representation or
  vector choice reuses them.
- A tile set is keyed by project, layer and **image representation only**.
- Sets are built in the background **30 s after the last image change** of the
  project; deleting the layer before that builds nothing.
- Urgent requests never wait behind a full build.
- Close zoom (< 16 frames per tile) uses frame previews, not 1/2/4/8 tiles.
- One request per screen returns all ready items of a level and range.

## Non-goals

- Rust `KrakenBlobGateway` is not used for tiles (decided: Python server).
- No change to tile pixels or to painting.

## Approaches considered

- Transport: Python server (chosen) vs Rust gateway: the win is fewer round
  trips and no waiting, not throughput.
- Close zoom: stored previews (chosen) vs prebuilt 1/2/4/8 levels: previews
  already exist; extra levels add several tile sets per layer.
- Preview storage: one per project (chosen) vs one per tile set (today):
  today's layout is what makes every new key cost ~4 minutes.

## Design

### Layout

```
<source_project_dir>/tile/
  previews-256/<sha[:2]>/<sha>.png           one per frame image, shared
  sets/<layer_id>/<image_representation_id>/
    current                                  token of the published set
    complete-<TILE_FORMAT>                   marker of a finished set
    <token>/<orientation>/<x1>_<y1>_<x2>_<y2>.png
```

- A project without a workspace binding (no `source_project_dir`) keeps the
  server's persistent `matrix-tiles` root with the same layout. *(My call.)*
- Old sets under `matrix-tiles` are not migrated; each layer is built once in
  the new place. Previews are copied from an old set when one exists for the
  same image digest. *(My call: copying previews avoids the 4-minute re-read.)*

### Building

1. **Previews first.** A set build first makes sure every frame has a preview
   (shared folder), then composes tiles from previews only. Composing a 16-frame
   tile from ready previews costs tens of milliseconds.
2. **Urgent first.** Urgent requests are served between every frame of the
   preview pass and between every tile, so the wait is bounded by one frame or
   one tile, not by a full level.
3. **Robust.** A frame whose source cannot be read is skipped and logged; it
   does not stop the builder. A builder thread that died is started again on the
   next request.
4. **When.** The outbox publisher reports image-affecting events
   (`LayerCreated`, `LayerFilesImported`, `ArtifactVersionCreated`,
   `ArtifactVersionImported`, `ArtifactVersionActivated`,
   `RepresentationActivated`, `RepresentationCreated`) to a tile scheduler. 30 s
   after the last such event of a project, the scheduler checks each active
   layer and image representation and builds those whose content stamp changed.
   `LayerArchived` cancels pending work for that layer.
5. A matrix read never starts a full build of a complete set. If the set is
   missing or incomplete, the read starts it immediately (no 30 s wait).

### Block request

`POST /api/v1/projects/{project_id}/viewport-tiles`, body
`{layer_id, image_representation_id, orientation, kind: "tiles"|"previews",
cells, x1, y1, x2, y2}`; at most 512 items per block.

Response `application/x-kraken-tile-block`: 4-byte big-endian header length,
UTF-8 JSON header `{revision, items: [{x1,y1,x2,y2,offset,length}], missing:
[{x1,y1,x2,y2}]}`, then PNG payloads back to back. Previews are one frame each.

### Client

- The matrix asks one block per screen and level; items go into the thumbnail
  cache under today's keys.
- Close zoom asks `previews` and uses them as per-cell thumbnails.
- The retry backoff doubles once per retry round, not once per missing tile.
- A 404 for the endpoint falls back to per-tile requests.

## Left to the builder

- How the 30 s debounce is held (one timer per project in the server process).
- Splitting large ranges into blocks.
