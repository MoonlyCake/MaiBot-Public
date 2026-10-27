"""画像和图片预热发生在真实消息登记时，不等待门控或外部服务。"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast


def load_runtime():
    path = Path(__file__).resolve().parents[2] / "src/maisaka/runtime.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MaisakaHeartFlowChatting")
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_register_message"]
    namespace = {"is_bot_self": lambda platform, user_id: user_id == "bot"}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=future + [node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[node.name]


Runtime = load_runtime()


def message(message_id, *, notify=False, command=False, user="user"):
    return NS(message_id=message_id, received_at=datetime.now(), is_notify=notify, is_command=command,
              platform="qq", message_info=NS(user_info=NS(user_id=user)))


def runtime():
    obj = Runtime()
    obj._focus_group = None
    obj._running = True
    obj._agent_state = "stop"
    obj._STATE_RUNNING = "running"
    obj.session_id = "chat"
    obj._last_message_received_at = 0
    obj._last_processed_index = 1
    obj.message_cache = [message("processed"), message("pending")]
    obj._quote_counter = Mock()
    obj._chat_history = []
    for name in ("_ensure_background_tasks_running", "_record_external_message_interval",
                 "_update_message_trigger_state", "_observe_message_debounce", "_trigger_ingested_jargon_learning",
                 "_emit_monitor_message_ingested", "_prune_processed_message_cache",
                 "_request_planner_interrupt_for_message", "_schedule_message_turn"):
        setattr(obj, name, Mock())
    obj._reasoning_engine = NS(prepare_message_memory=Mock(), _build_history_message=AsyncMock(return_value=None))
    obj._message_turn_scheduler = NS(observe_message=Mock())
    return obj


class MessageMemoryIngressTests(IsolatedAsyncioTestCase):
    async def test_receipt_starts_preparation_before_scheduling_without_consuming_batch(self):
        obj = runtime()
        incoming = message("new")
        events = []
        obj._reasoning_engine.prepare_message_memory.side_effect = lambda **kw: events.append("prepare")
        obj._schedule_message_turn.side_effect = lambda: events.append("schedule")
        await obj._register_message(incoming)
        self.assertEqual(events, ["prepare", "schedule"])
        obj._reasoning_engine.prepare_message_memory.assert_called_once_with(
            profile_message=incoming, source_messages=[obj.message_cache[1], incoming])
        self.assertEqual(obj._last_processed_index, 1)

    async def test_notifications_commands_and_bot_messages_do_not_start_preparation(self):
        for incoming in (message("notice", notify=True), message("command", command=True), message("self", user="bot")):
            with self.subTest(message_id=incoming.message_id):
                obj = runtime()
                await obj._register_message(incoming)
                obj._reasoning_engine.prepare_message_memory.assert_not_called()

    async def test_focus_routes_preparation_through_its_own_batch(self):
        obj = runtime()
        obj._focus_group = NS(has_message=Mock(return_value=False), register_message=AsyncMock())
        incoming = message("focus")
        await obj._register_message(incoming)
        obj._reasoning_engine.prepare_message_memory.assert_not_called()
        obj._focus_group.register_message.assert_awaited_once_with(obj, incoming, None)

