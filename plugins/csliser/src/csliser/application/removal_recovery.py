"""Bounded, explicit recovery of source quarantine after process interruption."""
from dataclasses import dataclass
import os
from pathlib import Path

from kraken_core.safe_files import ensure_regular_directory, open_regular_read
from csliser.domain.models import OperationError, OperationResult
from csliser.infrastructure.removal_receipt import read_receipt, receipt_lock, sync_directory, verified_digest


@dataclass(frozen=True)
class RecoveryPlan:
    operations: tuple[Path, ...]


def discover_removal_receipts(folder: Path, *, limit=10000) -> RecoveryPlan:
    root = ensure_regular_directory(folder.expanduser().absolute())
    paths = []
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= 1_000_000:
                raise ValueError("Recovery source directory exceeds the scan budget")
            if entry.name.startswith(".") and ".csliser-remove-" in entry.name and entry.name.endswith(".json"):
                paths.append(root / entry.name)
                if len(paths) > limit:
                    raise ValueError(f"At most {limit} removal receipts can be recovered in one operation")
    return RecoveryPlan(tuple(sorted(paths)))


def recover_removal(path: Path, *, cancelled=None) -> str:
    from .file_transfer import _rename_new

    # The lock lives on the immutable receipt itself and dies with its owner.
    with open_regular_read(path, root=path.parent) as stream:
        identity = os.fstat(stream.fileno())
        with receipt_lock(stream):
            source, quarantine, snapshot, digest = read_receipt(path, stream)
            if os.path.lexists(quarantine):
                if os.path.lexists(source):
                    raise FileExistsError(f"Source name is occupied; both files preserved: {source}, {quarantine}")
                if verified_digest(quarantine, snapshot, cancelled=cancelled) != digest:
                    raise OSError(f"Quarantine SHA-256 differs; file preserved: {quarantine}")
                if cancelled is not None and cancelled():
                    raise InterruptedError("Recovery cancelled; files preserved")
                _rename_new(quarantine, source)
                sync_directory(source.parent)
                if verified_digest(source, snapshot, cancelled=cancelled) != digest:
                    raise OSError(f"Restored file changed; receipt preserved: {source}")
                status = "restored"
            elif os.path.lexists(source):
                if verified_digest(source, snapshot, cancelled=cancelled) != digest:
                    raise OSError(f"Source changed; receipt preserved: {source}")
                status = "source_unchanged"
            else:
                # Source removal already happened; recovery never invents bytes.
                status = "already_removed"
    current = os.lstat(path)
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        raise OSError(f"Receipt changed; preserved: {path}")
    path.unlink()
    sync_directory(path.parent)
    return status


class RecoverRemovals:
    def __init__(self):
        self.outcomes = []

    def execute(self, plan: RecoveryPlan, *, progress=None, cancelled=None) -> OperationResult:
        self.outcomes = []
        errors = []
        total = len(plan.operations)
        for index, path in enumerate(plan.operations):
            if cancelled is not None and cancelled():
                return OperationResult(total, len(self.outcomes), total-len(self.outcomes), tuple(errors), cancelled=True)
            try:
                status = recover_removal(path, cancelled=cancelled)
                self.outcomes.append({"receipt": str(path), "status": status})
            except InterruptedError:
                return OperationResult(total, len(self.outcomes), total-len(self.outcomes), tuple(errors), cancelled=True)
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append(OperationError(path, None, str(error)))
            if progress is not None:
                progress(index+1, total, str(path))
        return OperationResult(total, len(self.outcomes), total-len(self.outcomes), tuple(errors))
