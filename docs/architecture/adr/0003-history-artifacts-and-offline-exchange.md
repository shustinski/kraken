# ADR 0003: Recorded-time history, immutable artifacts and offline exchange

Status: accepted — 2026-07-17

## Decision

`recorded_at` defines what Kraken knew at a point in time. `effective_at` is
displayed as a claimed business timestamp but never rewrites history. Current
and temporal projections provide read-only `as_of` state across all subsystems.

Managed artifacts are immutable and content-addressed by SHA-256. A changed
review return creates a candidate child version. Byte-identical CIF content is
unchanged; no semantic normalization is applied. If the active base changed
after issue, the candidate is a conflict branch and is not activated.

Offline review manifests are signed with Ed25519. Optional `.kraken-review`
envelopes use scrypt-derived AES-256-GCM. Legacy folders without a manifest are
never treated as trusted and require an ambiguity screen.

## Consequences

No successful workflow overwrites managed bytes. External URI history can prove
only the fingerprint observed by Kraken, so the UI must display a warning when
showing historical external references.

