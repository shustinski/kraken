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

Before source removal, an immutable sibling receipt records the source identity
and SHA-256 and is fsynced while holding an OS process lock. The quarantined
object is checked again before removal. A failed rollback identifies both the
preserved file and receipt. `--recover-source folder` (repeatable) or the
**Actions → Restore files from quarantine** menu restores verified quarantined
files without overwriting an occupied source name. A live owner, modified data
or invalid receipt is refused; both files and receipt remain available.
Recovery uses a background Qt worker and ordinary confirmation in the GUI.

Receipts cover the source-quarantine boundary of delete and cross-volume move.
Recovery restores the original; it does not resume deletion. A completed
cross-volume copy is preserved, so an interrupted move can recover with both
copies present. Already removed bytes cannot be reconstructed from a receipt.
This is not a whole-batch rollback journal or a physical power-loss guarantee.
POSIX directory metadata is fsynced; Windows receipt fsync is not a directory
durability SLA. Preview metadata alone is not a content hash or a hostile-host sandbox.
The GUI accepts `--source` (repeatable), `--extension` (repeatable), `--frames`,
`--frames-per-row`, `--selection-mode`, `--operation`, and `--destination`.
These options configure the window; copying, moving and deleting still require
the ordinary preview and confirmations. Omitted options preserve saved settings.
Use `--settings-file path.ini` for an isolated instance.

`--plan` prints a JSON summary without executing file operations.
Windows builds include `CSliserCli.exe` for console output, alongside the normal
`CSliser.exe` GUI. Legacy unconnected `open_path`, `open_folder`, and `run_action`
actions are deprecated and return failure; `--get-capabilities` reports discovery.
No Agent job capability or IPC command server is advertised.

Selections are bounded to 4096 ranges, one million unique frames and four million
expansion visits. Excessive selections produce a validation error before expansion.

`--self-test-dir output` runs a bounded Qt worker copy check in a fresh child
directory with its own settings. It verifies source/destination SHA-256 and refusal
to overwrite. Reference input requires `--source` and explicit `--frames`; the probe
accepts at most 64 files / 512 MiB, copies reference inputs, and rejects move/delete
or an external destination. A separate child creates its own diagnostic file,
exits ungracefully after quarantine, and must recover the exact original bytes.
`--self-test-removal-crash-dir` exposes only that bounded generated-file probe;
it accepts no existing source. Notification dialogs are collected without interaction.
This proves Qt/runtime behavior; it does not certify ordinary external GUI input.

The isolated Windows installation and Linux source operation acceptance are
recorded in [the functional audit](../../docs/audits/2026-10-27-csliser-launch-and-install.md).
