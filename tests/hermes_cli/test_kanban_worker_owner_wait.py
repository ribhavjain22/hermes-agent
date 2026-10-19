"""A Kanban worker spawned while `hermes update` drains the owner waits for it instead of failing.

The dispatcher keeps spawning workers during an update window; the owner answers ``draining``
until the update releases the install. Exiting 1 there books a crash against the card, and two
of those trip the breaker and park a card that never ran (the in-flight handoff cell). The wait
must never submit the claim twice, and a non-transient refusal still fails at once.
"""

from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_client, kanban_worker_client


def _owner(states, submitted):
    @asynccontextmanager
    async def connect():
        state = states.pop(0) if states else None
        if state:
            # The exact refusal connect_gateway raises for a draining/starting owner.
            unavailable = getattr(gateway_client, "GatewayUnavailableError", None)
            if unavailable is None:  # pre-fix tree: the same message as a plain client error
                raise gateway_client.GatewayClientError(f"Gateway {state}: update_paused")
            raise unavailable(f"Gateway {state}: update_paused", state)

        class _Client:
            async def rpc(self, method, **params):
                submitted.append(method)
                return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

        yield _Client()

    return connect


@pytest.mark.asyncio
async def test_worker_waits_out_a_draining_owner_and_submits_once(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["draining", "starting"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_worker_fails_at_once_when_the_owner_is_not_transiently_down(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["inaccessible"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert submitted == []


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_worker_retries_a_transient_gateway_connection_failure_before_submitting(monkeypatch, tmp_path):
    submitted = []
    attempts = 0

    @asynccontextmanager
    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise gateway_client.GatewayClientError(
                "Gateway connection failed; no local fallback"
            ) from ConnectionRefusedError(111, "connection refused")
        async with _owner([], submitted)() as client:
            yield client

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    result = await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))

    assert result == 0
    assert attempts == 2
    assert submitted == ["kanban.run"]
