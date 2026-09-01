"""Schema dispatch shared by the Agent store, HTTP service, and runner."""

from __future__ import annotations

import json
from typing import Mapping, TypeAlias

from kraken_core.analysis_run_protocol import (
    ANALYSIS_PARTITION_JOB_SCHEMA,
    ANALYSIS_PARTITION_RESULT_SCHEMA,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    canonical_json,
)
from kraken_core.plugin_protocol import PluginJobManifest, PluginResultManifest


AgentManifest: TypeAlias = PluginJobManifest | AnalysisPartitionJobManifest
AgentResult: TypeAlias = PluginResultManifest | AnalysisPartitionResultManifest


def manifest_to_json(manifest: AgentManifest) -> str:
    if isinstance(manifest, PluginJobManifest):
        return manifest.to_json()
    return canonical_json(manifest.to_payload())


def result_to_json(result: AgentResult) -> str:
    if isinstance(result, PluginResultManifest):
        return result.to_json()
    return canonical_json(result.to_payload())


def parse_manifest_payload(payload: Mapping[str, object]) -> AgentManifest:
    if payload.get("schema") == ANALYSIS_PARTITION_JOB_SCHEMA:
        return AnalysisPartitionJobManifest.from_payload(payload)
    return PluginJobManifest.from_dict(payload)


def parse_manifest_json(raw: str) -> AgentManifest:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Agent job manifest must be a JSON object")
    return parse_manifest_payload(payload)


def parse_result_json(raw: str) -> AgentResult:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Agent result manifest must be a JSON object")
    if payload.get("schema") == ANALYSIS_PARTITION_RESULT_SCHEMA:
        return AnalysisPartitionResultManifest.from_payload(payload)
    return PluginResultManifest.from_dict(payload)


__all__ = [
    "AgentManifest",
    "AgentResult",
    "manifest_to_json",
    "parse_manifest_json",
    "parse_manifest_payload",
    "parse_result_json",
    "result_to_json",
]
