# Contributing

Thanks for your interest in improving ViaLaNet Polygon Widget!

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,build]"
pre-commit install
```

On Linux / macOS replace the activation line with `source .venv/bin/activate`.

## Everyday workflow

Run the quality gates before opening a pull request:

```powershell
ruff check contour tests examples
ruff format --check contour tests examples
mypy contour
$env:QT_QPA_PLATFORM = "offscreen"
pytest             # fast behavioral suite (default)
pytest -m full     # before a build or on demand: all tests, including vectorization
```

`fast` is the default and runs behavioral checks only. `full` includes every
collected test, including `fast`, GUI regressions, vectorization and benchmarks.
The Windows build script runs `full` before packaging. Benchmarks and real-image
regressions require their SEM fixtures; absent fixture files are reported as skips.
New test modules belong to `full` by default. Add stable behavioral modules to
`FAST_TEST_FILES` in `tests/conftest.py`; mark individual image-processing tests in
mixed modules with `@pytest.mark.vectorization` to keep them out of `fast`.
Profiling is disabled in tests by default to avoid distorting timings; profiling
tests explicitly enable the switches they exercise.

All checks must be green. `pre-commit` is configured to run a subset of these
automatically on every commit.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<optional-scope>): <short summary>

<body explaining the why>
```

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`.

Keep the summary ≤ 72 characters and in the imperative mood.

## Pull requests

1. Fork & branch from `main` (e.g. `feat/ruler-snapping`).
2. Keep changes focused; split unrelated work into separate PRs.
3. Update `CHANGELOG.md` under the `[Unreleased]` section.
4. Ensure CI is green before requesting review.

## Releasing

Releases are automated by `.github/workflows/release.yml` on tag push.

1. Bump `__version__` in [`contour/__version__.py`](contour/__version__.py).
2. Move the `[Unreleased]` section in `CHANGELOG.md` under the new version and
   update the version comparison links.
3. Commit and tag: `git tag v0.4.1 && git push origin v0.4.1`.
4. GitHub Actions builds the Windows installer and attaches it to the release.

## Reporting bugs

Please include:

- Steps to reproduce.
- Expected vs. actual behaviour.
- Contour version (`contour --version`) and Windows build number.
- Relevant log excerpts from
  `%LOCALAPPDATA%\ViaLaNet\Contour\logs\app.log`.
