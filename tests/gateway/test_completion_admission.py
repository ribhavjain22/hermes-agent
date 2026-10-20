"""Real adapter admission is the completion acknowledgement boundary."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import async_delegation as delegation



def test_route_identity_ignores_mutable_display_and_actor_metadata():
    from dataclasses import replace

    from gateway.session_automation import _route_identity

    first = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        user_id="user-1",
        user_name="First name",
        chat_name="Original thread name",
        chat_topic="Original topic",
        scope_id="guild-1",
        profile="main",
        auto_thread_created=True,
        auto_thread_initial_name="Initial name",
    )
    later = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        user_id="user-2",
        user_name="Second name",
        chat_name="Renamed thread",
        chat_topic="Updated topic",
        scope_id="guild-1",
        profile="main",
        auto_thread_created=False,
    )

    assert _route_identity(first) == _route_identity(later)
    assert _route_identity(first) != _route_identity(replace(later, scope_id="guild-2"))
    assert _route_identity(first) != _route_identity(replace(later, thread_id="thread-2"))
    assert _route_identity(first) != _route_identity(replace(later, chat_id="thread-2"))
    assert _route_identity(first) != _route_identity(replace(later, profile="other"))


def pending(key, name):
    evt = {"type": "async_delegation", "session_key": key, "delegation_id": name,
           "summary": name, "status": "completed", "dispatched_at": time.time()}
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {"status": "completed", "summary": name})
    return evt


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))
        # gather() over tasks that are ALL already done completes without yielding (3.12+), so a
        # finished task whose queued _background_tasks.discard callback has not run yet would spin
        # this loop forever; yield once so those done-callbacks run before re-checking.
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_completion_ack_requires_admission_and_replay_never_repeats(tmp_path, caplog):
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    events = [pending(key, f"admission-{i}") for i in range(2)]
    received = []
    release, started = asyncio.Event(), asyncio.Event()

    async def handler(event):
        received.append(event.text)
        started.set()
        await release.wait()
        if key not in adapter._pending_messages:
            queued = runner._promote_queued_event(key, adapter, None)
            if queued is not None:
                adapter._pending_messages[key] = queued

    try:
        # Missing handler must not acknowledge either durable sibling.
        for _ in range(10):
            assert await runner._deliver_async_delegation_group(events) is False
        for event in events:
            row = delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        assert not runner._completion_deliveries_delivered
        adapter.set_message_handler(handler)
        await adapter.handle_message(MessageEvent(text="human-active", source=source))
        await asyncio.wait_for(started.wait(), 2)
        await adapter.handle_message(MessageEvent(text="human-pending", source=source))
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        runner._BUSY_QUEUE_MAX_PENDING = 1
        caplog.set_level(logging.WARNING, logger="gateway.run")
        for _ in range(10):
            assert await runner._deliver_async_delegation_group(events) is False
        refusals = [record for record in caplog.records if "admission_conflict" in record.message]
        assert len(refusals) <= 1
        assert adapter._pending_messages[key].text == "human-pending"
        assert not runner._completion_deliveries_delivered
        for event in events:
            row = delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        # An explicitly mismatched adapter key must fail closed too.
        wrong = dict(events[0], session_key="agent:main:discord:dm:other",
                     platform="discord", chat_type="dm", chat_id="42")
        assert await runner._inject_watch_notification("wrong-route", wrong) is False
        runner._BUSY_QUEUE_MAX_PENDING = 4
        assert await runner._deliver_async_delegation_group(events) is True
        assert await runner._deliver_async_delegation_group(events) is None
        release.set()
        await drain(adapter)
        assert received[:2] == ["human-active", "human-pending"]
        assert len(received) == 3 and all(event["summary"] in received[-1] for event in events)
        for event in events:
            assert delegation.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
        idle = pending(key, "idle-admitted")
        assert await runner._deliver_async_delegation_group([idle]) is True
        await drain(adapter)
        assert len(received) == 4 and "idle-admitted" in received[-1]
    finally:
        release.set()
        await drain(adapter)
        await runner._cancel_process_completion_batch_tasks()


@pytest.mark.asyncio
async def test_busy_completion_refusal_is_suppressed_and_backed_off(tmp_path, caplog):
    from gateway.wake import WakeNotAccepted

    runner = GatewayRunner(GatewayConfig())
    claim = SimpleNamespace(
        proceed=True, early_result=None, defer=False, claim_id=None,
        delegation_id=None, owner_session_id="", proven_session_id="",
    )
    runner._preflight_completion_delivery = AsyncMock(return_value=claim)
    runner._completion_delivery_identity = lambda _evt: None

    @asynccontextmanager
    async def scope(_evt):
        yield

    runner._completion_event_scope = scope
    attempts = 0

    async def refuse_busy(*_args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert kwargs["log_refusal"] is False
        raise WakeNotAccepted("admission_conflict")

    runner._inject_watch_notification = refuse_busy
    event = {"type": "completion", "session_key": "agent:main:discord:dm:42",
             "platform": "discord", "chat_id": "42"}
    caplog.set_level(logging.WARNING, logger="gateway.run")

    assert await runner._deliver_completion_notification("done", event) is False
    assert await runner._deliver_completion_notification("done", event) is False

    assert attempts == 1
    assert not any("Watch notification not admitted" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_unavailable_raw_route_is_quiet_without_hiding_invalid_routes(tmp_path, caplog):
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {}
    evt = pending("opaque-client-session", "raw-admission")
    caplog.set_level(logging.WARNING, logger="gateway.run")
    for _ in range(3):
        assert await runner._deliver_async_delegation_group([evt]) is False
    assert not caplog.records
    row = delegation.get_durable_delegation(evt["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert await runner._inject_watch_notification("watch", {"type": "watch_match", "session_key": "agent:broken"}) is None
    assert any("unresolvable" in record.message for record in caplog.records)
    # API recovery writes only the delivery row, never starts a model turn.
    from gateway.platforms.api_server import APIServerAdapter
    api = APIServerAdapter(PlatformConfig())
    db = SessionDB(tmp_path / "api.db")
    db.create_session(evt["session_key"], "api_server")
    api._ensure_session_db = lambda: db
    runner.adapters = {Platform.API_SERVER: api}
    try:
        caplog.clear()
        assert await runner._deliver_async_delegation_group([evt]) is True
        assert await runner._deliver_async_delegation_group([evt]) is None
        rows = db.get_messages(evt["session_key"])
        assert len(rows) == 1 and rows[0]["display_kind"] == "async_delegation_complete"
        assert not api._background_tasks and not caplog.records
    finally:
        db.close()
