from __future__ import annotations

import pytest

from kraken_server.project_deletion import wait_for_lock


class _Connection:
    class dialect:
        name = "postgresql"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def execute(self, *_args, **_kwargs):
        self.calls += 1
        answer = self.answers.pop(0) if self.answers else False
        return type("Result", (), {"scalar": staticmethod(lambda: answer)})()


def test_exclusive_storage_lock_is_retried_while_requests_hold_it_briefly() -> None:
    connection = _Connection([False, False, True])
    slept: list[float] = []

    assert wait_for_lock(connection, "kraken:storage", timeout=5, sleep=slept.append) is True
    assert connection.calls == 3
    assert len(slept) == 2


def test_storage_lock_wait_gives_up_after_the_timeout() -> None:
    connection = _Connection([])
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    assert wait_for_lock(connection, "kraken:storage", timeout=1, interval=0.25,
                         sleep=sleep, clock=lambda: now[0]) is False
    assert connection.calls == pytest.approx(5, abs=1)


def test_storage_barriers_reuse_pooled_connections() -> None:
    import sqlalchemy as sa

    from kraken_server.project_deletion import ProjectOperationMiddleware, _gate_engine

    url = sa.engine.make_url("postgresql+psycopg://u:p@127.0.0.1:5432/kraken")
    assert _gate_engine(url) is _gate_engine(url)
    assert not isinstance(_gate_engine(url).pool, sa.pool.NullPool)
    middleware = ProjectOperationMiddleware(None, sa.create_engine(url))
    assert not isinstance(middleware.engine.pool, sa.pool.NullPool)
    assert middleware.engine is _gate_engine(url)
