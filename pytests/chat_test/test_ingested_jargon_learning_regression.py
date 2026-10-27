"""Run the ingestion scheduler in isolation; never import MaiBot or call a model."""

from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import ast
import asyncio
import logging
import unittest


class _ContextMessage:
    @classmethod
    def from_session_message(cls, message, **kwargs):
        return SimpleNamespace(original_message=message, **kwargs)


class _PendingMessages:
    def __init__(self):
        self.items = {}

    def extend(self, messages):
        for message in messages:
            self.items.setdefault(message.original_message.message_id, message)

    def clear(self):
        self.items.clear()

    def __len__(self):
        return len(self.items)


def _load_scheduler_methods():
    path = Path(__file__).resolve().parents[2] / "src/maisaka/runtime.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    runtime = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MaisakaHeartFlowChatting")
    selected = {"_trigger_ingested_jargon_learning", "_cancel_trimmed_history_learning_task"}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + [node for node in runtime.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in selected],
        type_ignores=[],
    )
    namespace = {"asyncio": asyncio, "SessionBackedMessage": _ContextMessage, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class IngestedJargonLearningRegression(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.methods = _load_scheduler_methods()
        self.pending = _PendingMessages()
        self.block = asyncio.Event()
        self.runs = []
        self.tasks = []

        async def run_learning(messages, **kwargs):
            self.runs.append((messages, kwargs))
            await self.block.wait()

        self.runtime = SimpleNamespace(
            _enable_jargon_learning=True,
            _jargon_learner=SimpleNamespace(
                pending_messages=self.pending,
                min_messages_for_extraction=10,
                collect_context_messages=self.pending.extend,
            ),
            _ingested_jargon_learning_task=None,
            _trimmed_history_learning_task=None,
            _queued_trimmed_learning_progress=None,
            _run_trimmed_history_learning=run_learning,
            _handle_trimmed_history_learning_done=lambda task: None,
            log_prefix="isolated",
        )
        self.trigger = MethodType(self.methods["_trigger_ingested_jargon_learning"], self.runtime)

    async def asyncTearDown(self):
        task = self.runtime._ingested_jargon_learning_task
        if task is not None:
            self.tasks.append(task)
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    @staticmethod
    def messages(start, stop):
        return [
            SimpleNamespace(message_id=str(index), raw_message=object(), processed_plain_text=f"message {index}")
            for index in range(start, stop)
        ]

    async def test_under_threshold_only_accumulates(self):
        with patch.object(asyncio, "create_task", wraps=asyncio.create_task) as create_task:
            self.trigger(self.messages(0, 9))
            self.trigger(self.messages(0, 9))
        self.assertEqual(len(self.pending), 9)
        self.assertIsNone(self.runtime._ingested_jargon_learning_task)
        create_task.assert_not_called()

    async def test_threshold_starts_without_trim_and_active_task_is_reused(self):
        with patch.object(asyncio, "create_task", wraps=asyncio.create_task) as create_task:
            self.trigger(self.messages(0, 10))
            task = self.runtime._ingested_jargon_learning_task
            await asyncio.sleep(0)
            self.trigger(self.messages(10, 13))
        self.assertIsNotNone(task)
        self.assertIs(self.runtime._ingested_jargon_learning_task, task)
        self.assertFalse(task.done())
        self.assertEqual(len(self.pending), 13)
        create_task.assert_called_once()
        self.assertEqual(self.runs, [([], {
            "enable_expression_learning": False,
            "enable_jargon_learning": True,
            "enable_high_frequency_learning": False,
            "publish_queued_progress": False,
        })])

    async def test_disabled_learning_clears_without_starting(self):
        self.trigger(self.messages(0, 9))
        self.runtime._enable_jargon_learning = False
        with patch.object(asyncio, "create_task", wraps=asyncio.create_task) as create_task:
            self.trigger(self.messages(9, 10))
        self.assertEqual(len(self.pending), 0)
        self.assertIsNone(self.runtime._ingested_jargon_learning_task)
        create_task.assert_not_called()

    async def test_shutdown_cancels_trimmed_and_ingested_learning(self):
        first = asyncio.create_task(self.block.wait())
        second = asyncio.create_task(self.block.wait())
        self.tasks.extend((first, second))
        self.runtime._trimmed_history_learning_task = first
        self.runtime._ingested_jargon_learning_task = second
        self.runtime._queued_trimmed_learning_progress = object()
        await self.methods["_cancel_trimmed_history_learning_task"](self.runtime)
        self.assertTrue(first.cancelled())
        self.assertTrue(second.cancelled())
        self.assertIsNone(self.runtime._trimmed_history_learning_task)
        self.assertIsNone(self.runtime._ingested_jargon_learning_task)
        self.assertIsNone(self.runtime._queued_trimmed_learning_progress)


if __name__ == "__main__":
    unittest.main()
