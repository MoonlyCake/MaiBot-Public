"""Focus 消息预热与首轮共享等待：只装载真实方法，不初始化应用。"""

from html import escape
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import asyncio
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]


class SessionBackedMessage:
    def __init__(self, original):
        self.original_message = original
        self.message_id = original.message_id


class Entry:
    def __init__(self, messages, session_ids=None, sequence=0, *, inputs=None):
        self.messages, self.inputs = messages, inputs or {}
        self.progress = {sid: max(sequence for sequence, _ in records) for sid, records in self.inputs.items()}


def message(sid, name, *, notify=False, command=False, user="user"):
    return SimpleNamespace(session_id=sid, message_id=name, is_notify=notify, is_command=command,
                           is_at=False, platform="offline",
                           message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user)))


def entry(original, sequence):
    return Entry([SessionBackedMessage(original)], inputs={original.session_id: [(sequence, original.message_id)]})


def load_group():
    path = ROOT / "src/maisaka/focus/group_runtime.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "FocusGroupRuntime")
    methods = {"register_message", "_queue_entry", "_group_reference_messages", "_build_references", "_run_turn"}
    node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name in methods]
    # 局部导入仅提供常量；测试注入同一接口，避免启动运行时模块。
    for method in node.body:
        method.body = [item for item in method.body if not isinstance(item, ast.ImportFrom)]
    namespace = dict(asyncio=asyncio, time=time, uuid=uuid, escape=escape, logger=Mock(),
                     SessionBackedMessage=SessionBackedMessage, FocusContextEntry=Entry,
                     is_bot_self=lambda platform, user_id: user_id == "bot",
                     MAX_INTERNAL_ROUNDS=2, MESSAGE_MEMORY_WAIT_SECONDS=1.5)
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + [node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace["FocusGroupRuntime"]


class FocusMemoryPreparationTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.group = object.__new__(load_group())
        self.group.entries, self.group._pending, self.group._considered = [], {}, {}
        self.group._sequence, self.group._wait_until = 0, None
        self.group._ready, self.group._debounce_tasks = set(), {}
        self.group.members = {}
        for sid in ("a", "b"):
            engine = SimpleNamespace(prepare_message_memory=Mock(),
                                     _build_planner_injected_user_messages=AsyncMock(return_value=[sid]),
                                     _refresh_jargon_reference_message=Mock(return_value=None))
            self.group.members[sid] = SimpleNamespace(session_id=sid, session_name=sid, _reasoning_engine=engine)
            self.group._debounce_tasks[sid] = SimpleNamespace(done=lambda: False)
        self.group._remember, self.group._ensure_worker = Mock(return_value=True), Mock()
        self.group.checkpoint = AsyncMock()

    async def test_receipt_prepares_own_unconsidered_sources_and_filters_incoming(self):
        runtime = self.group.members["a"]
        self.group._considered["a"] = 1
        self.group.entries = [entry(message("a", "old"), 1), entry(message("a", "current"), 2),
                              entry(message("a", "notice", notify=True), 3), entry(message("b", "foreign"), 4)]
        self.group._pending["a"] = [entry(message("a", "pending"), 5)]
        self.group._sequence = 5
        incoming = message("a", "incoming")
        await self.group.register_message(runtime, incoming, SessionBackedMessage(incoming))
        anchor, sources = runtime._reasoning_engine.prepare_message_memory.call_args.args
        self.assertIs(anchor, incoming)
        # 保留旧 Planner 分组语义；过滤只约束触发预热的新入站消息。
        self.assertEqual([source.message_id for source in sources], ["current", "notice", "pending", "incoming"])
        for invalid in (message("a", "notify", notify=True), message("a", "command", command=True),
                        message("a", "self", user="bot")):
            await self.group.register_message(runtime, invalid, SessionBackedMessage(invalid))
        runtime._reasoning_engine.prepare_message_memory.assert_called_once()

    async def test_members_share_deadline_and_one_input_snapshot(self):
        self.group.entries = [entry(message("a", "a1"), 1), entry(message("b", "b1"), 2)]
        entered, release, calls = asyncio.Event(), asyncio.Event(), []

        async def build(**kwargs):
            sources = kwargs["source_messages"]
            calls.append((kwargs["message_memory_deadline"], [source.message_id for source in sources]))
            if len(calls) == 2:
                entered.set()
            await release.wait()
            return [sources[-1].message_id]

        for member in self.group.members.values():
            member._reasoning_engine._build_planner_injected_user_messages.side_effect = build
        task = asyncio.create_task(self.group._build_references(message_memory_deadline=123.0))
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.5)
            self.group.entries.append(entry(message("b", "future"), 3))
        finally:
            release.set()
        references = await task
        self.assertEqual(calls, [(123.0, ["a1"]), (123.0, ["b1"])])
        self.assertEqual(len(references), 2)

    async def test_only_first_round_has_wait_budget(self):
        group = self.group
        group._closed, group._generation, group._cycle = False, 0, 0
        group._sent, group.history, group.max_context_size = [], [], 2
        group._take_ready_inputs, group._set_stage, group._roster = Mock(), Mock(), Mock(return_value="roster")
        group._compact, group._emit_response = AsyncMock(), AsyncMock()
        group._build_references = AsyncMock(return_value=[])
        group._execute_tools = AsyncMock(return_value=([], False))
        group.tools = SimpleNamespace(refresh=AsyncMock(), definitions=lambda: [], deferred_notice=lambda: "")
        group._service = SimpleNamespace(chat_loop_step=AsyncMock(side_effect=[
            SimpleNamespace(raw_messages=[], tool_calls=[SimpleNamespace(call_id="tool")]),
            SimpleNamespace(raw_messages=[], tool_calls=[]),
        ]))
        before = asyncio.get_running_loop().time()
        await group._run_turn()
        after = asyncio.get_running_loop().time()
        deadlines = [call.kwargs["message_memory_deadline"] for call in group._build_references.await_args_list]
        self.assertEqual(len(deadlines), 2)
        self.assertGreaterEqual(deadlines[0], before + 1.5)
        self.assertLessEqual(deadlines[0], after + 1.5)
        self.assertIsNone(deadlines[1])
