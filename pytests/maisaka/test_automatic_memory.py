"""Exercise real memory workers with controlled I/O, without importing the app."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import ast
import asyncio
import sys
import time

import unittest


@dataclass
class Message:
    message_id: str
    content: str = ""


class Summary(Message):
    pass


def extract(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names or (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                            for target in node.targets))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def make_memory_engine():
    module = ModuleType(__name__ + "_production")
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update(asyncio=asyncio, time=time, deque=deque, dataclass=dataclass, field=field,
               MemorySummaryMessage=Summary,
               global_config=NS(chat=NS(mid_term_memory=True, mid_term_memory_lenth=3)),
               logger=NS(**{level: lambda *a, **kw: None for level in
                            ("debug", "info", "warning", "error", "exception")}),
               is_mid_term_memory_reference_message=lambda message: False)
    root = Path(__file__).resolve().parents[2]
    extract(root / "src/maisaka/memory/mid_term.py", {
        "is_mid_term_memory_message", "insert_mid_term_memory_message",
        "_find_last_mid_term_memory_index", "_trim_mid_term_memory_messages"}, env)
    extract(root / "src/maisaka/reasoning_engine.py", {
        "AutomaticMemoryBatch", "MessageMemoryBatch", "AUTOMATIC_MEMORY_TIMEOUT_SECONDS",
        "MESSAGE_MEMORY_WAIT_SECONDS", "MaisakaReasoningEngine"}, env)
    runtime = NS(session_id="test-session", log_prefix="test", _chat_history=[], _max_context_size=10,
                 chat_stream=NS(is_group_session=True),
                 _chat_loop_service=NS(select_llm_context_messages=lambda history, **kw: (list(history), 0)))
    engine = env["MaisakaReasoningEngine"](runtime)
    gate, entered, calls, branch_gates = asyncio.Event(), asyncio.Event(), [], {}
    gate.set()

    def backend(name):
        async def call(**kwargs):
            messages = kwargs.get("source_messages", kwargs.get("pending_messages", []))
            key = tuple(message.message_id for message in messages)
            calls.append((name, key))
            entered.set()
            await branch_gates.get(name, gate).wait()
            return ["profile:" + ",".join(key)] if name == "profile" else (
                name + ":" + ",".join(key) if name in {"image", "knowledge"}
                else ("heuristic" if name == "heuristic" else None))
        return call

    env.update(heuristic_memory_injector=NS(build_injection_message=backend("heuristic"),
                                          get_cached_reference_expires_at=lambda sid: time.time() + 60),
               image_memory_injector=NS(build_injection_message=backend("image")),
               knowledge_memory_injector=NS(build_injection_message=backend("knowledge")),
               build_person_profile_injection_messages=backend("profile"),
               build_mid_term_memory_reference_message=backend("mid_term"))

    async def inject(message_id, deadline=None):
        message = Message(message_id)
        return await engine._build_planner_injected_user_messages(
            profile_message=message, source_messages=[message], deferred_tools_reminder="tools",
            message_memory_deadline=deadline)

    def prepare(message_id):
        message = Message(message_id)
        engine.prepare_message_memory(message, [message])
        return engine._latest_message_memory_batch

    return NS(env=env, runtime=runtime, engine=engine, gate=gate, entered=entered, calls=calls,
              inject=inject, prepare=prepare, branch_gates=branch_gates, module_name=module.__name__)


class AutomaticMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = make_memory_engine()

    async def asyncTearDown(self):
        try:
            await self.fixture.engine.stop_memory_preparation()
        finally:
            sys.modules.pop(self.fixture.module_name, None)

    async def test_same_batch_starts_once_and_reuses_ready_results(self):
        fixture = self.fixture
        assert await fixture.inject("first") == ["tools"]
        await asyncio.gather(fixture.engine._automatic_memory_task, fixture.engine._message_memory_task)
        expected = ["tools", "heuristic", "profile:first", "image:first", "knowledge:first"]
        assert await fixture.inject("first") == expected
        assert await fixture.inject("first") == expected
        assert sorted(name for name, _ in fixture.calls) == ["heuristic", "image", "knowledge", "mid_term", "profile"]


    async def test_active_batch_coalesces_latest_and_keeps_image_sources(self):
        fixture = self.fixture
        fixture.gate.clear()
        await fixture.inject("active")
        await asyncio.wait_for(fixture.entered.wait(), 1)
        task = fixture.engine._automatic_memory_task
        assert await fixture.inject("skipped") == ["tools"]
        assert await fixture.inject("latest") == ["tools"]
        assert fixture.engine._automatic_memory_task is task
        fixture.gate.set()
        await asyncio.gather(task, fixture.engine._message_memory_task)
        profile_keys = [key for name, key in fixture.calls if name == "profile"]
        assert profile_keys == [("active",), ("latest",)]
        assert "image:latest" in await fixture.inject("latest")
        unrelated = await fixture.inject("unrelated")
        assert "heuristic" in unrelated
        assert not any(text.startswith(("image:", "profile:", "knowledge:")) for text in unrelated)


    async def test_summary_fifo_merges_into_latest_history_without_losing_sources(self):
        fixture = self.fixture
        started, release, observed = asyncio.Event(), asyncio.Event(), []

        async def summarize(messages, **kwargs):
            ids = tuple(message.message_id for message in messages)
            observed.append(ids)
            started.set()
            await release.wait()
            return NS(message=Summary("summary:" + ",".join(ids)), model_name="offline", total_tokens=0)

        fixture.env["build_mid_term_memory_message"] = summarize
        source = [Message("removed-1"), Message("removed-2")]
        fixture.engine._schedule_mid_term_summary(source)
        source.clear()
        await asyncio.wait_for(started.wait(), 1)
        fixture.engine._schedule_mid_term_summary([Message("removed-3")])
        fixture.runtime._chat_history.append(Message("arrived-during-summary"))
        release.set()
        await fixture.engine._mid_term_summary_task
        assert [message.message_id for message in fixture.runtime._chat_history] == ["arrived-during-summary"]
        fixture.engine._merge_completed_memory()
        assert observed == [("removed-1", "removed-2"), ("removed-3",)]
        assert [message.message_id for message in fixture.runtime._chat_history] == [
            "summary:removed-1,removed-2", "summary:removed-3", "arrived-during-summary"]


    async def test_stop_cancels_both_workers_and_their_children(self):
        fixture = self.fixture
        fixture.gate.clear()
        summary_started = asyncio.Event()

        async def summarize(messages, **kwargs):
            summary_started.set()
            await fixture.gate.wait()

        fixture.env["build_mid_term_memory_message"] = summarize
        before = asyncio.all_tasks()
        await fixture.inject("active")
        fixture.engine._schedule_mid_term_summary([Message("preserved")])
        await asyncio.wait_for(fixture.entered.wait(), 1)
        await asyncio.wait_for(summary_started.wait(), 1)
        owned = asyncio.all_tasks() - before - {asyncio.current_task()}
        await fixture.engine.stop_memory_preparation()
        assert owned and all(task.done() for task in owned)
        assert fixture.engine._automatic_memory_task is None
        assert fixture.engine._mid_term_summary_task is None
        assert fixture.engine._pending_automatic_memory_batch is None
        assert fixture.engine._mid_term_summary_queue[0][0].message_id == "preserved"

    async def test_receipt_starts_profile_image_and_knowledge_before_planner_without_heuristic(self):
        fixture = self.fixture
        fixture.gate.clear()
        batch = fixture.prepare("received")
        await asyncio.wait_for(fixture.entered.wait(), 1)
        assert sorted(name for name, _ in fixture.calls) == ["image", "knowledge", "profile"]
        assert fixture.engine._active_message_memory_batch is batch
        assert fixture.engine._automatic_memory_task is None

    async def test_first_wait_uses_its_batch_even_when_future_receipt_arrives(self):
        fixture = self.fixture
        fixture.gate.clear()
        fixture.branch_gates["heuristic"] = asyncio.Event()
        current = fixture.prepare("current")
        await asyncio.wait_for(fixture.entered.wait(), 1)
        waiter = asyncio.create_task(fixture.inject("current", asyncio.get_running_loop().time() + 1))
        await asyncio.sleep(0)
        future = fixture.prepare("future")
        fixture.gate.set()
        result = await asyncio.wait_for(waiter, 1)
        assert "profile:current" in result and "image:current" in result and "knowledge:current" in result
        assert not any(text in result for text in ["profile:future", "image:future", "knowledge:future"])
        assert fixture.engine._planner_message_memory_batch is current
        assert fixture.engine._latest_message_memory_batch is future
        assert not fixture.engine._automatic_memory_task.done()

    async def test_timeout_keeps_worker_and_partial_current_result_without_renewing_budget(self):
        fixture = self.fixture
        fixture.prepare("old")
        await fixture.engine._message_memory_task
        fixture.gate.clear()
        fixture.entered.clear()
        profile_gate = asyncio.Event()
        profile_gate.set()
        fixture.branch_gates["profile"] = profile_gate
        current = fixture.prepare("current")
        await asyncio.wait_for(fixture.entered.wait(), 1)
        worker = fixture.engine._message_memory_task
        deadline = asyncio.get_running_loop().time() + 0.01
        waiter = asyncio.create_task(fixture.inject("current", deadline))
        await asyncio.sleep(0)
        fixture.prepare("future")
        result = await asyncio.wait_for(waiter, 1)
        assert "profile:current" in result and not any(text.startswith("image:") for text in result)
        assert "profile:old" not in result and "profile:future" not in result
        assert not worker.done() and not worker.cancelled()
        retry = await asyncio.wait_for(fixture.inject("current", asyncio.get_running_loop().time() + 1.5), 0.1)
        assert retry == result and current.wait_deadline == deadline
        fixture.gate.set()
        await worker
        assert "image:current" in await fixture.inject("current")

    async def test_later_planner_round_does_not_wait_for_message_memory(self):
        fixture = self.fixture
        fixture.gate.clear()
        fixture.runtime._cycle_counter = 1
        fixture.runtime._update_stage_status = lambda *args, **kwargs: None
        fixture.env["resolve_enable_visual_planner"] = lambda: True
        deadlines = []
        build_messages = fixture.engine._build_planner_injected_user_messages

        async def capture_messages(**kwargs):
            deadlines.append(kwargs["message_memory_deadline"])
            return await build_messages(**kwargs)

        async def definitions():
            return [], "tools"

        async def complete(*args, **kwargs):
            return None

        fixture.engine._build_planner_injected_user_messages = capture_messages
        fixture.engine._build_action_tool_definitions = definitions
        fixture.engine._refresh_jargon_reference_message = lambda: None
        fixture.engine._emit_flow_step = complete
        fixture.engine._run_interruptible_planner = complete
        message = Message("later")
        await asyncio.wait_for(fixture.engine._run_planner_request(
            trigger_message=message, source_messages=[message], round_index=1, round_text="second", state=NS()), 0.1)
        assert deadlines == [None]
        assert not fixture.engine._message_memory_task.done()

    async def test_stop_clears_message_batch_ready_and_cancels_receipt_children(self):
        fixture = self.fixture
        fixture.gate.clear()
        profile_gate = asyncio.Event()
        profile_gate.set()
        fixture.branch_gates["profile"] = profile_gate
        before = asyncio.all_tasks()
        batch = fixture.prepare("receipt")
        await asyncio.wait_for(fixture.entered.wait(), 1)
        assert batch.ready["profile"] == ["profile:receipt"]
        owned = asyncio.all_tasks() - before - {asyncio.current_task()}
        await fixture.engine.stop_memory_preparation()
        assert owned and all(task.done() for task in owned)
        assert batch.ready == {}
        assert fixture.engine._message_memory_task is None
        assert fixture.engine._latest_message_memory_batch is None
        assert fixture.engine._active_message_memory_batch is None
        assert fixture.engine._planner_message_memory_batch is None


if __name__ == "__main__":
    unittest.main()
