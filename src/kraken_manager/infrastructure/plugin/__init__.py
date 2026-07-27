"""Adapters for isolated, manifest-based plugin execution."""

from .agent_gateway import AgentPluginGateway
from .result_reader import AgentStagingResultContentReader, domain_result_from_transport

__all__ = ["AgentPluginGateway", "AgentStagingResultContentReader", "domain_result_from_transport"]
