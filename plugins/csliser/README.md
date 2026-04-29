# CSliser

CSliser selects numbered frame files from one or more source folders and copies,
moves, or deletes the matching files.

Run the plugin from the workspace:

```powershell
uv run python -m csliser
```

The frame expression accepts inclusive ranges separated by comma or semicolon,
for example `10-20;100:300,500`.
