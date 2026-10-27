# CSliser

CSliser selects numbered frame files from one or more source folders and copies,
moves, or deletes the matching files.

Run the plugin from the workspace:

```powershell
uv run python -m csliser
```

The frame expression accepts inclusive ranges separated by comma or semicolon,
for example `10-20;100:300,500`.

Files may use a numeric stem (`0000.jpg`) or a numbered suffix
(`metal_0000.jpg`). A selected number must identify exactly one file per source
folder. Ambiguous selections and colliding output paths are rejected when the
plan is built. Existing destination files are never overwritten.

The preview records source identity, size and modification time. Replaced or
changed sources require a new plan. Copies use private staging, SHA-256 readback
and exclusive publication; moves and deletes verify the object actually moved
before removing it. Cancellation is checked during copy and verification. Closing
a busy window requests cancellation and waits for the worker to stop.

Windows and Linux source operations are covered by functional tests. Linux uses
[`renameat2(RENAME_NOREPLACE)`](https://man7.org/linux/man-pages/man2/rename.2.html)
for exclusive moves. An unsupported filesystem fails safely; it does not fall
back to replacing a destination. A private verified copy can use exclusive hard
link publication when native exclusive rename is unavailable.

If source removal cannot be rolled back, the error identifies the preserved
quarantine file. There is no durable operation journal or automatic power-loss
recovery yet. Preview metadata is not a content hash or a hostile-host sandbox.
Hub IPC window actions and current installed distributions require separate
acceptance; the standalone-to-Hub managed import test does not certify them.
