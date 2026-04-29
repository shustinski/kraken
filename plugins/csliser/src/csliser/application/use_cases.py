from __future__ import annotations

import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from csliser.domain.models import (
    FileOperation,
    OperationError,
    OperationPlan,
    OperationResult,
    PlannedOperation,
    ProcessingConfig,
)
from csliser.domain.planner import build_operation_plan

ProgressCallback = Callable[[int, int, str], None]
CancelPredicate = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class TransferExecutionOptions:
    copy_metadata: bool = False
    max_workers: int | None = None
    progress_batch_size: int = 25


@dataclass(frozen=True, slots=True)
class _TaskOutcome:
    operation: PlannedOperation
    error: OperationError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BuildTransferPlan:
    def execute(self, config: ProcessingConfig) -> OperationPlan:
        return build_operation_plan(config)


class ExecuteTransferPlan:
    def execute(
        self,
        plan: OperationPlan,
        *,
        options: TransferExecutionOptions | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelPredicate | None = None,
    ) -> OperationResult:
        execution_options = options or TransferExecutionOptions()
        errors: list[OperationError] = []
        total = len(plan.operations)
        if total == 0:
            return OperationResult(requested=0, completed=0, skipped=0)
        if _is_cancelled(cancelled):
            return OperationResult(requested=total, completed=0, skipped=total, cancelled=True)

        errors.extend(_prepare_destination_dirs(plan))
        if _is_cancelled(cancelled):
            return OperationResult(
                requested=total,
                completed=0,
                skipped=total,
                errors=tuple(errors),
                cancelled=True,
            )

        workers = resolve_transfer_workers(total, execution_options.max_workers)
        if workers <= 1:
            return _execute_sync(
                plan,
                options=execution_options,
                progress=progress,
                cancelled=cancelled,
                initial_errors=errors,
            )
        return _execute_parallel(
            plan,
            options=execution_options,
            workers=workers,
            progress=progress,
            cancelled=cancelled,
            initial_errors=errors,
        )


def resolve_transfer_workers(operation_count: int, max_workers: int | None = None) -> int:
    if operation_count <= 0:
        return 0
    if max_workers is not None:
        return max(1, min(operation_count, max_workers))
    return min(operation_count, min(8, max(2, (os.cpu_count() or 4) * 2)))


def _execute_sync(
    plan: OperationPlan,
    *,
    options: TransferExecutionOptions,
    progress: ProgressCallback | None,
    cancelled: CancelPredicate | None,
    initial_errors: list[OperationError],
) -> OperationResult:
    errors = list(initial_errors)
    completed = 0
    processed = 0
    total = len(plan.operations)
    cancelled_requested = False

    for item in plan.operations:
        if _is_cancelled(cancelled):
            cancelled_requested = True
            break
        outcome = _execute_one(plan.config.operation, item, copy_metadata=options.copy_metadata)
        processed += 1
        if outcome.ok:
            completed += 1
        elif outcome.error is not None:
            errors.append(outcome.error)
        _maybe_emit_progress(
            progress,
            processed=processed,
            total=total,
            path=str(item.source),
            batch_size=options.progress_batch_size,
            force=processed == total,
        )

    return OperationResult(
        requested=total,
        completed=completed,
        skipped=total - completed,
        errors=tuple(errors),
        cancelled=cancelled_requested,
    )


def _execute_parallel(
    plan: OperationPlan,
    *,
    options: TransferExecutionOptions,
    workers: int,
    progress: ProgressCallback | None,
    cancelled: CancelPredicate | None,
    initial_errors: list[OperationError],
) -> OperationResult:
    errors = list(initial_errors)
    completed = 0
    processed = 0
    total = len(plan.operations)
    next_index = 0
    cancelled_requested = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[_TaskOutcome], PlannedOperation] = {}
        while next_index < total and len(pending) < workers and not _is_cancelled(cancelled):
            item = plan.operations[next_index]
            pending[_submit(executor, plan.config.operation, item, options)] = item
            next_index += 1

        while pending:
            for future in as_completed(tuple(pending)):
                item = pending.pop(future)
                outcome = _future_outcome(future, item)
                processed += 1
                if outcome.ok:
                    completed += 1
                elif outcome.error is not None:
                    errors.append(outcome.error)

                _maybe_emit_progress(
                    progress,
                    processed=processed,
                    total=total,
                    path=str(outcome.operation.source),
                    batch_size=options.progress_batch_size,
                    force=processed == total,
                )

                if _is_cancelled(cancelled):
                    cancelled_requested = True
                while next_index < total and len(pending) < workers and not cancelled_requested:
                    next_item = plan.operations[next_index]
                    pending[_submit(executor, plan.config.operation, next_item, options)] = next_item
                    next_index += 1
                break

            if cancelled_requested and not pending:
                break

        return OperationResult(
            requested=total,
            completed=completed,
            skipped=total - completed,
            errors=tuple(errors),
            cancelled=cancelled_requested,
        )


def _submit(
    executor: ThreadPoolExecutor,
    operation: FileOperation,
    item: PlannedOperation,
    options: TransferExecutionOptions,
) -> Future[_TaskOutcome]:
    return executor.submit(_execute_one, operation, item, copy_metadata=options.copy_metadata)


def _execute_one(operation: FileOperation, item: PlannedOperation, *, copy_metadata: bool) -> _TaskOutcome:
    try:
        if operation == FileOperation.COPY:
            assert item.destination is not None
            if copy_metadata:
                shutil.copy2(item.source, item.destination)
            else:
                shutil.copyfile(item.source, item.destination)
        elif operation == FileOperation.MOVE:
            assert item.destination is not None
            shutil.move(str(item.source), str(item.destination))
        elif operation == FileOperation.DELETE:
            os.remove(item.source)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    except (OSError, shutil.Error) as exc:
        return _TaskOutcome(item, OperationError(item.source, item.destination, str(exc)))
    return _TaskOutcome(item)


def _prepare_destination_dirs(plan: OperationPlan) -> list[OperationError]:
    if plan.config.operation == FileOperation.DELETE:
        return []
    parents = {item.destination.parent for item in plan.operations if item.destination is not None}
    errors: list[OperationError] = []
    for parent in sorted(parents):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(OperationError(Path(), parent, str(exc)))
    return errors


def _future_outcome(future: Future[_TaskOutcome], item: PlannedOperation) -> _TaskOutcome:
    try:
        return future.result()
    except Exception as exc:
        return _TaskOutcome(item, OperationError(item.source, item.destination, str(exc)))


def _maybe_emit_progress(
    progress: ProgressCallback | None,
    *,
    processed: int,
    total: int,
    path: str,
    batch_size: int,
    force: bool = False,
) -> None:
    if progress is None:
        return
    normalized_batch_size = max(1, batch_size)
    if force or processed % normalized_batch_size == 0:
        progress(processed, total, path)


def _is_cancelled(cancelled: CancelPredicate | None) -> bool:
    return bool(cancelled is not None and cancelled())
