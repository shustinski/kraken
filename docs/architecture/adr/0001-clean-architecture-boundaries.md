# ADR 0001: Clean architecture boundaries

Status: accepted — 2026-07-17

## Decision

Project-management code lives under `kraken_manager` with the dependency
direction `domain ← application ← infrastructure/presentation`. `kraken_core`
contains only technical runtime, themes and public plugin transport schemas.
The three executable packages are composition roots and may select adapters.

Architecture tests inspect Python imports and reject framework, filesystem,
database, HTTP, GitLab and plugin dependencies in domain/application packages.

## Consequences

PyQt models use presentation DTOs rather than domain aggregates. SQLAlchemy and
FastAPI types do not cross application ports. New storage types can be tested
against common semantic contracts and registered without modifying domain code.

