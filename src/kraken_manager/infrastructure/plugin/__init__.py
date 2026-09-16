"""Adapters for isolated, manifest-based plugin execution."""

from .agent_gateway import AgentPluginGateway
from .analysis_gateway import AgentAnalysisGateway
from .result_reader import AgentStagingResultContentReader, domain_result_from_transport

__all__ = [
    "AgentAnalysisGateway",
    "AgentPluginGateway",
    "AgentStagingResultContentReader",
    "domain_result_from_transport",
]
