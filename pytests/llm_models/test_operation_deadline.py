"""离线验证真实编排方法；AST 装载避免配置、数据库和客户端初始化。"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import asyncio
import inspect
import random


ROOT = Path(__file__).resolve().parents[2]


def _load(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if methods is not None:
        for node in nodes:
            node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and item.name in methods]
    future = ast.parse("from __future__ import annotations").body
    unit = ast.Module(body=future + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), namespace)
    return namespace


class OperationDeadlineTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = SimpleNamespace(get_client_class_instance_async=AsyncMock(return_value=object()))
        provider = SimpleNamespace(name="provider", client_type="offline")
        models = {name: SimpleNamespace(name=name, model_identifier=name, api_provider=provider.name)
                  for name in ("a", "b", "c")}
        self.ns = _load("src/llm_models/exceptions.py", {
            "ModelAttemptFailed", "LLMTaskTimeoutError", "ReqAbortException", "RespNotOkException",
        }, {"error_code_mapping": {}})
        self.ns.update({
            "asyncio": asyncio, "datetime": datetime, "inspect": inspect, "random": random,
            "RequestType": SimpleNamespace(RESPONSE="response"), "EMBEDDING_TASK_NAMES": set(),
            "RequestTraceContext": lambda **kwargs: SimpleNamespace(current_attempt_started_at=0, **kwargs),
            "LLMExecutionResult": SimpleNamespace, "logger": Mock(), "client_registry": self.registry,
            "TempMethodsLLMUtils": SimpleNamespace(get_model_info_by_name=models.__getitem__,
                                                   get_provider_by_name=lambda name: provider),
            "has_request_snapshot": lambda error: True,
            "save_failed_request_snapshot": Mock(return_value="offline-snapshot"),
            "serialize_client_request_snapshot": lambda request: {},
            "attach_request_snapshot": Mock(), "mark_request_final_failure": Mock(),
            "update_failed_request_attempt": Mock(),
        })
        _load("src/llm_models/utils_model.py", {"LLMOrchestrator"}, self.ns,
              {"_execute_request", "_select_model", "_adjust_model_usage"})
        self.bot = object.__new__(self.ns["LLMOrchestrator"])
        self.bot.task_name, self.bot.request_type = "utils", "offline"
        self.bot.model_for_task = SimpleNamespace(hard_timeout=0.08, model_list=list(models),
                                                 selection_strategy="sequential")
        self.bot.model_usage = dict.fromkeys(models, (0, 0, 0))
        self.bot._explicit_model_usage = {}
        self.bot._refresh_task_config = Mock(return_value=self.bot.model_for_task)
        self.bot._resolve_effective_session_id = lambda session_id: session_id
        self.bot._build_client_request = lambda **kwargs: SimpleNamespace(
            model_info=kwargs["model_info"], trace_context=kwargs["trace_context"])
        self.bot._schedule_llm_error_event = Mock()
        self.bot._attempt_request_on_model = AsyncMock()

    async def execute(self, **kwargs):
        return await self.bot._execute_request(self.ns["RequestType"].RESPONSE, **kwargs)

    def assert_no_slots(self):
        self.assertEqual([usage[2] for usage in self.bot.model_usage.values()], [0, 0, 0])

    async def test_candidates_share_one_deadline(self):
        self.bot.model_for_task.hard_timeout = 0.15
        calls = []

        async def attempt(provider, client, request):
            calls.append(request.model_info.name)
            await asyncio.sleep(0.1)
            if len(calls) == 1:
                raise self.ns["ModelAttemptFailed"]("first candidate failed")
            return SimpleNamespace(usage=None)

        self.bot._attempt_request_on_model.side_effect = attempt
        with self.assertRaises(self.ns["LLMTaskTimeoutError"]) as caught:
            await self.execute()
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(caught.exception.model_name, "b")
        self.assert_no_slots()
        self.bot._schedule_llm_error_event.assert_called_once()

    async def test_fast_failure_can_switch_and_succeed(self):
        self.bot._attempt_request_on_model.side_effect = [
            self.ns["ModelAttemptFailed"]("fast failure"), SimpleNamespace(usage=None),
        ]
        result = await self.execute()
        self.assertEqual(result.model_info.name, "b")
        self.assert_no_slots()
        self.assertEqual(self.bot.model_usage["a"][1], 1)
        self.bot._schedule_llm_error_event.assert_not_called()

    async def test_external_cancellation_is_unchanged(self):
        entered, cancelled = asyncio.Event(), []

        async def attempt(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                cancelled.append(exc)
                raise

        self.bot._attempt_request_on_model.side_effect = attempt
        task = asyncio.create_task(self.execute())
        await entered.wait()
        task.cancel("external stop")
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        self.assertIs(caught.exception, cancelled[0])
        self.assertEqual(str(caught.exception), "external stop")
        self.assert_no_slots()
        self.bot._schedule_llm_error_event.assert_not_called()

    async def test_abort_is_unchanged(self):
        error = self.ns["ReqAbortException"]("steering")
        self.bot._attempt_request_on_model.side_effect = error
        with self.assertRaises(type(error)) as caught:
            await self.execute()
        self.assertIs(caught.exception, error)
        self.assert_no_slots()
        self.bot._schedule_llm_error_event.assert_not_called()

    async def test_context_timeout_is_unchanged_and_releases_slot(self):
        error = TimeoutError("context failure")

        def context_factory(client):
            raise error

        with self.assertRaises(TimeoutError) as caught:
            await self.execute(context_factory=context_factory)
        self.assertIs(caught.exception, error)
        self.assert_no_slots()
        self.bot._attempt_request_on_model.assert_not_called()
        self.bot._schedule_llm_error_event.assert_not_called()

    async def test_selection_timeout_has_no_model_or_leaked_slot(self):
        async def select_client(provider):
            await asyncio.Event().wait()

        self.registry.get_client_class_instance_async.side_effect = select_client
        with self.assertRaises(self.ns["LLMTaskTimeoutError"]) as caught:
            await self.execute()
        self.assertEqual(caught.exception.model_name, "")
        self.assertNotIn("模型", str(caught.exception))
        self.assert_no_slots()
        self.bot._attempt_request_on_model.assert_not_called()
        self.ns["save_failed_request_snapshot"].assert_not_called()

    def test_heuristic_attempt_requires_interval_and_new_messages(self):
        ns = _load("src/maisaka/memory/heuristic_injector.py", {"HeuristicMemoryInjector"}, {},
                   {"_can_trigger"})
        can_trigger = ns["HeuristicMemoryInjector"]._can_trigger
        options = dict(state=SimpleNamespace(last_attempt_at=100, last_attempt_message_count=20),
                       min_interval_seconds=30, min_new_messages=5)
        self.assertFalse(can_trigger(total_message_count=25, now=129, **options))
        self.assertFalse(can_trigger(total_message_count=24, now=130, **options))
        self.assertTrue(can_trigger(total_message_count=25, now=130, **options))
