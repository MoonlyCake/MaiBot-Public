"""回复活性离线回归：不启动应用、不发送消息、不调用模型。"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import runpy
import sys
import time
import uuid

from src.llm_models.payload_content.context_item import ContextItemMeta, ContextToolCall, FunctionCallItem
from src.llm_models.payload_content.tool_option import ToolCall


ROOT = Path(__file__).resolve().parents[2]
is_bare_ai_label = runpy.run_path(str(ROOT / "src/chat/utils/identity_behavior.py"))["is_bare_ai_label"]


def _load_class(path, name, namespace, methods=None, *, decorate=False):
    """直接编译生产类，隔离模块级配置初始化与数据库连接。"""
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = []
    if methods is not None:
        node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in methods]
    if not decorate:
        node.decorator_list = []
    module = ModuleType(f"reply_liveness_regression_{name}")
    sys.modules[module.__name__] = module
    module.__dict__.update(namespace)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), module.__dict__)
    return module.__dict__[name]


CycleEnd = _load_class("src/maisaka/reasoning_engine.py", "CycleEnd", {"dataclass": dataclass}, decorate=True)
CycleRuntimeState = _load_class("src/maisaka/reasoning_engine.py", "CycleRuntimeState", {"dataclass": dataclass, "field": field, "CycleEnd": CycleEnd}, decorate=True)
ModelOutputContextMessage = _load_class("src/maisaka/context/messages.py", "ModelOutputContextMessage", {"dataclass": dataclass}, decorate=True)
TextComponent = _load_class("src/common/data_models/message_component_data_model.py", "TextComponent", {})
CONFIG = SimpleNamespace(
    chat=SimpleNamespace(reply_timing=SimpleNamespace(inevitable_at_reply=True, mentioned_bot_reply=False)),
    capabilities=SimpleNamespace(tool_result_condense=False),
)
Engine = _load_class("src/maisaka/reasoning_engine.py", "MaisakaReasoningEngine", {
    "global_config": CONFIG, "ToolCall": ToolCall, "ContextToolCall": ContextToolCall,
    "FunctionCallItem": FunctionCallItem, "ContextItemMeta": ContextItemMeta,
    "ModelOutputContextMessage": ModelOutputContextMessage, "CycleEnd": CycleEnd,
    "LLMContextMessage": ModelOutputContextMessage, "time": time, "uuid": uuid,
    "logger": Mock(), "deque": deque, "is_bot_self": lambda platform, user_id: user_id == "bot",
    "is_bare_ai_label": is_bare_ai_label, "TextComponent": TextComponent,
    "is_mentioned_bot_in_message": lambda message: (message.is_mentioned, message.is_at, 0.0),
    "STOP_AFTER_EXECUTION_PAUSE_REASON": "stop_after_execution",
}, {
    "__init__", "_begin_reply_accounting", "_start_turn_reply_accounting", "_consume_ingested_forced_turn_reason",
    "_complete_unanswered_user_turn", "_handle_tool_calls",
    "_handle_planner_response_actions", "_end_planner_no_tool_cycle", "_cycle_end_for_pause_tool",
    "_get_planner_content",
})


def _message(message_id="user-message", *, user_id="user"):
    return SimpleNamespace(message_id=message_id, is_notify=False, is_command=False, is_at=False,
                           is_mentioned=False, platform="qq", processed_plain_text="正常问题", raw_message=SimpleNamespace(components=[TextComponent("正常问题")]),
                           message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user_id), group_info=object()))


def _result(name, *, success=True):
    metadata = {"pause_execution": True} if name == "stop" else {"reply_sent": success} if name == "reply" else {}
    return SimpleNamespace(tool_name=name, success=success, metadata=metadata,
                           stop_after_execution=False, post_history_messages=[], get_history_content=lambda: "ok" if success else "failed")


def _engine():
    runtime = SimpleNamespace(
        _message_turn_scheduler=SimpleNamespace(allows_tool=lambda spec: True),
        _chat_history=[], _tool_registry=SimpleNamespace(list_tools=AsyncMock(return_value=[]), invoke=AsyncMock()),
        _is_reply_frequency_silent=lambda: False, _uses_reply_window_quota=lambda: False,
        _forced_turn_message_id=None,
        _consume_forced_turn_reason=Mock(return_value=None), _enter_running_state=Mock(),
        _enter_stop_state=Mock(), _end_planner_continuation=Mock(), _update_stage_status=Mock(),
        _reset_consecutive_wait_count=Mock(), is_action_tool_currently_available=lambda name: True, log_prefix="test",
    )
    engine = Engine(runtime)
    engine._log_tool_call_source = Mock()
    engine._build_tool_invocation = lambda call, thought: SimpleNamespace(tool_name=call.func_name, arguments=call.args)
    engine._build_tool_execution_context = lambda thought: None
    engine._build_tool_availability_context = lambda: None
    engine._record_tool_execution_effects = AsyncMock()
    engine._append_tool_execution_result = Mock()
    engine._append_tool_post_history_messages = Mock()
    engine._emit_planner_progress = AsyncMock()
    engine._should_replace_reasoning = lambda thought: False
    async def display(**kwargs):
        kwargs["tool_monitor_results"].append({"tool_name": kwargs["tool_call"].func_name, "success": kwargs["result"].success})
    engine._append_tool_display_results = display
    async def invoke(invocation, context):
        return _result(invocation.tool_name)
    runtime._tool_registry.invoke.side_effect = invoke
    return engine, runtime


def _start(engine, message=None, *, turn="turn", after_unanswered=False):
    """after_unanswered 模拟上一轮真实用户消息未被回复，使本轮安静收尾时触发一次补答。"""
    messages = [] if message is None else [message]
    engine._start_turn_reply_accounting(SimpleNamespace(logical_turn_id=turn, cached_messages=messages,
        trigger_message=message, proactive_triggered=False))
    if after_unanswered:
        engine._unanswered_user_turns = 1


async def _action(engine, names):
    state = CycleRuntimeState()
    await engine._handle_planner_response_actions(
        response=SimpleNamespace(content="根据当前对话判断。", raw_messages=[],
                                 tool_calls=[ToolCall(f"call-{i}", name, {}) for i, name in enumerate(names)]),
        cycle_detail=SimpleNamespace(time_records={}), state=state, planner_extra_lines=state.planner_extra_lines,
    )
    return state


class TestReplyLiveness(IsolatedAsyncioTestCase):
    async def test_dynamic_window_never_uses_legacy_liveness_compensation(self):
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                engine, runtime = _engine()
                runtime._uses_reply_window_quota = lambda: True
                _start(engine, _message(), after_unanswered=True)
                engine._reply_liveness_prepared = prepared
                state = CycleRuntimeState()
                await engine._complete_unanswered_user_turn(
                    SimpleNamespace(time_records={}), state, "此前已准备回复",
                )
                runtime._tool_registry.invoke.assert_not_awaited()
                self.assertFalse(engine._reply_liveness_attempted)
                self.assertEqual(engine._unanswered_user_turns, 1)

    async def test_first_stop_is_normal_but_second_user_turn_gets_one_reply(self):
        engine, runtime = _engine()
        _start(engine, _message("first"), turn="first")
        await _action(engine, ["stop"])
        self.assertEqual(runtime._tool_registry.invoke.await_count, 1)
        _start(engine, _message("second"), turn="second")
        state = await _action(engine, ["stop"])
        names = [call.args[0].tool_name for call in runtime._tool_registry.invoke.await_args_list]
        self.assertEqual(names, ["stop", "stop", "reply"])
        self.assertEqual(state.tool_monitor_results[-1]["tool_name"], "reply")
        self.assertTrue(state.tool_monitor_results[-1]["success"])
        self.assertEqual(engine._unanswered_user_turns, 0)

    async def test_emoji_preparation_then_stop_does_not_replay_the_tool(self):
        engine, runtime = _engine()
        _start(engine, _message())
        await _action(engine, ["show_emoji_list"])
        state = await _action(engine, ["stop"])
        invocations = [call.args[0] for call in runtime._tool_registry.invoke.await_args_list]
        self.assertEqual([call.tool_name for call in invocations], ["show_emoji_list", "stop", "reply"])
        self.assertEqual(invocations[-1].arguments["msg_id"], "user-message")
        self.assertNotIn("attach_emoji", invocations[-1].arguments)
        self.assertTrue(invocations[-1].arguments["reply_reference"])
        self.assertEqual(state.tool_monitor_results[-1]["tool_name"], "reply")
        self.assertIsInstance(runtime._chat_history[-1].output_item, FunctionCallItem)

    async def test_compensation_can_finish_a_no_tool_response(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        await _action(engine, [])
        self.assertEqual(runtime._tool_registry.invoke.await_args.args[0].tool_name, "reply")

    async def test_wait_continuation_preserves_one_attempt(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        await _action(engine, ["wait"])
        self.assertFalse(engine._reply_liveness_attempted)
        _start(engine, turn="turn")
        self.assertEqual(engine._reply_liveness_messages[0].message_id, "user-message")
        await _action(engine, ["stop"])
        await _action(engine, [])
        names = [call.args[0].tool_name for call in runtime._tool_registry.invoke.await_args_list]
        self.assertEqual(names, ["wait", "stop", "reply"])

    async def test_failed_reply_is_exposed_once_and_never_accounted_success(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        runtime._tool_registry.invoke.side_effect = lambda invocation, context: _result(invocation.tool_name, success=invocation.tool_name != "reply")
        state = await _action(engine, ["stop"])
        await _action(engine, [])
        names = [call.args[0].tool_name for call in runtime._tool_registry.invoke.await_args_list]
        self.assertEqual(names, ["stop", "reply"])
        self.assertFalse(state.tool_monitor_results[-1]["success"])
        self.assertFalse(engine._reply_liveness_succeeded)
        self.assertEqual(engine._unanswered_user_turns, 2)

    async def test_existing_reply_attempt_is_not_duplicated_even_when_it_fails(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        runtime._tool_registry.invoke.side_effect = lambda invocation, context: _result(invocation.tool_name, success=invocation.tool_name != "reply")
        await _action(engine, ["reply"])
        await _action(engine, ["stop"])
        self.assertEqual([call.args[0].tool_name for call in runtime._tool_registry.invoke.await_args_list], ["reply", "stop"])

    async def test_no_user_or_silent_turn_never_generates_a_reply(self):
        for source in (None, _message(user_id="bot"), _message()):
            engine, runtime = _engine()
            _start(engine, source, after_unanswered=True)
            if source is not None and source.message_info.user_info.user_id == "user":
                runtime._is_reply_frequency_silent = lambda: True
            await _action(engine, ["stop"])
            self.assertEqual(runtime._tool_registry.invoke.await_count, 1)

    async def test_successful_reply_then_stop_does_not_add_another(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        await _action(engine, ["reply", "stop"])
        self.assertEqual(runtime._tool_registry.invoke.await_count, 2)
        self.assertEqual(engine._unanswered_user_turns, 0)

    async def test_steered_bare_ai_label_is_not_a_compensation_target(self):
        engine, runtime = _engine()
        bare = _message("bare")
        bare.processed_plain_text = "AI"
        _start(engine, bare, after_unanswered=True)
        await _action(engine, ["stop"])
        self.assertEqual(runtime._tool_registry.invoke.await_count, 1)
        _start(engine, _message("real"), turn="next", after_unanswered=True)
        engine._reply_liveness_messages.append(bare)
        await _action(engine, ["stop"])
        self.assertEqual(runtime._tool_registry.invoke.await_args.args[0].arguments["msg_id"], "real")
