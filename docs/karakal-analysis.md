# Karakal analysis in Kraken

Karakal has two execution modes backed by the same Qt-free engine:

- standalone Karakal keeps its private history in SQLite;
- Kraken submits immutable, checksummed partitions to `karakal-worker` through Kraken Agent.

Karakal never receives `KRAKEN_DATABASE_URL` and never calls NeuralImage directly. NeuralImage results are registered as project artifact versions, then the operator binds those versions to `A`, `B`, and optionally `C` in **Оценить результат**.

## Local test environment

Python 3.14 and `uv` are required. From the repository root:

```powershell
uv sync --package kraken --extra dev --extra desktop
uv sync --package karakal --extra dev
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest tests plugins/karakal/tests
```

The full workspace also builds Contour's native extension and therefore needs Microsoft C++ Build Tools 14+ on Windows.

Standalone history defaults to `%LOCALAPPDATA%\Karakal\data\analysis.sqlite3`. To isolate a test database:

```powershell
$env:KARAKAL_DATA_DIR = "$PWD\.test_runtime\karakal"
uv run karakal
```

## Agent and Worker

Create an Agent registry such as `.test_runtime\agent-plugins.json`:

```json
{
  "plugins": [
    {
      "operation": "karakal.analyze.v1",
      "command": ["karakal-worker"]
    }
  ]
}
```

Start the Agent:

```powershell
uv run kraken-agent --data-dir .test_runtime\agent --plugins-config .test_runtime\agent-plugins.json
```

Each analysis run is split deterministically into partitions of at most 1000 frames. Completed partitions can be imported immediately. A retry keeps imported partitions and submits only `queued`, `failed`, or interrupted partitions.

Worker inputs are read-only staged files. Worker outputs are:

- `result.json` using `kraken.analysis-partition-result.v1`;
- `outputs/frames.jsonl.gz` containing all finite frame metrics;
- `progress.json`, atomically replaced after each processed frame.

## PostgreSQL server

Install server dependencies and apply both Alembic revisions:

```powershell
uv sync --extra server --extra postgres --extra reports --extra packages
$env:KRAKEN_DATABASE_URL = "postgresql+psycopg://kraken:password@localhost/kraken"
uv run alembic upgrade head
```

Revision `20260901_0002` adds analysis runs, source bindings, partitions, frame results, metric values, scales, and derived-artifact references. Standalone Karakal continues to use SQLite.

## Reproducibility

The run fingerprint excludes `run_id`, but includes the ordered frame selection, source versions, recipe, parameters, metric-registry version, engine build, runtime versions, partition algorithm, and optional seed. Two deliberate reruns therefore have different identities but the same fingerprint when their analytical inputs are identical.
