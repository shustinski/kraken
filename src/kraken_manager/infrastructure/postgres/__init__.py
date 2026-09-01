"""Optional PostgreSQL adapters loaded by the ``postgres`` extra."""

from .account_store import PostgresAccountStore
from .event_store import PostgresEventStore
from .identity_store import PostgresIdentityAclStore
from .job_queue import PostgresLeaseJobQueue
from .performer_store import PostgresPerformerStore
from .projection_store import PostgresProjectionStore
from .session_store import PostgresFederatedSessionCache
from .unit_of_work import PostgresUnitOfWork, PostgresUnitOfWorkFactory

__all__ = [
    "PostgresEventStore",
    "PostgresAccountStore",
    "PostgresIdentityAclStore",
    "PostgresLeaseJobQueue",
    "PostgresPerformerStore",
    "PostgresProjectionStore",
    "PostgresFederatedSessionCache",
    "PostgresUnitOfWork",
    "PostgresUnitOfWorkFactory",
]
from .analysis_store import PostgresAnalysisProjectionStore

__all__ = ["PostgresAnalysisProjectionStore"]
