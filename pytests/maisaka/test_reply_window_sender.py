"""发送前预留、实际发送记账及补答强制引用的离线回归。"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import asyncio
import json
import traceback


ROOT = Path(__file__).resolve().parents[2]


class Text:
    def __init__(self, text):
        self.text = text


class Extension:
    def __init__(self, **kwargs):
        self.context = kwargs["context"]
        self.reply_id, self.targets, self.parameters = "offline-reply", [], {}
        self.prepare = AsyncMock()


def fixture():
    path = ROOT / "src/maisaka/builtin_tool/reply.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"handle_tool", "_handle_reserved_reply", "_normalize_reply_arguments", "_use_expression_intent",
             "_build_monitor_metadata", "_build_send_result", "_resolve_segment_reply_context"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    nodes += [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "_REPLY_TOOL_INTERNAL_ARGUMENTS" for target in node.targets)]
    target = NS(message_id="original-message", message_info=NS(user_info=NS(
        user_id="user", user_cardname="", user_nickname="user")))
    reservation = NS(hourly_generation=None)
    runtime = NS(session_id="chat", log_prefix="test", _chat_history=[],
        chat_stream=NS(platform="qq", is_group_session=True),
        find_source_message_by_id=Mock(return_value=target),
        reserve_planner_reply=Mock(return_value=reservation), finish_planner_reply=Mock(),
        _update_stage_status=Mock(), _reasoning_engine=NS(_refresh_chat_history_visual_placeholders=AsyncMock()))
    generation = NS(completion=NS(response_text="第一段。第二段。"), monitor_detail={},
                    selected_expression_ids=[], error_message="")
    replyer = NS(generate_reply_with_context=AsyncMock(return_value=(True, generation)))
    sender = AsyncMock(side_effect=[NS(message_id="sent-1"), NS(message_id="sent-2")])
    tool_ctx = NS(runtime=runtime,
        post_process_reply_message_items_async=AsyncMock(return_value=[
            NS(sequence=NS(components=[Text("第一段。")]), quote_previous=False),
            NS(sequence=NS(components=[Text("第二段。")]), quote_previous=True),
        ]), append_guided_reply_to_chat_history=Mock())
    tool_ctx.build_failure_result = lambda name, content, **kwargs: NS(
        success=False, metadata=kwargs.get("metadata", {}), content=content)
    tool_ctx.build_success_result = lambda name, content, **kwargs: NS(
        success=True, metadata=kwargs.get("metadata", {}), content=content)
    quote = AsyncMock(return_value=(False, 0.0))
    namespace = dict(asyncio=asyncio, json=json, traceback=traceback, deepcopy=deepcopy, logger=Mock(),
        config_module=NS(global_config=NS(expression=NS(use_vector_expression=False),
            chat=NS(enable_reply_at=True, reply_style=NS(group_quote_frequency="never", private_quote_frequency="never")),
            emoji=NS(use_new_send_logic=True), bot=NS(nickname="麦麦"))),
        ReplyExtensionExecution=Extension, ToolAvailabilityContext=lambda **kwargs: NS(**kwargs),
        replyer_manager=NS(get_replyer=Mock(return_value=replyer)),
        knowledge_memory_injector=NS(build_injection_message=AsyncMock(return_value="")),
        _with_jargon_reply_reference=lambda arguments, *_args: arguments,
        _find_recent_reply_to_target=Mock(return_value=""),
        _invoke_before_post_process_hook=AsyncMock(return_value=("第一段。第二段。", {})),
        _resolve_reply_quote=quote, build_reply_monitor_detail=lambda result: {},
        build_visible_text_from_sequence=lambda sequence: "".join(item.text for item in sequence.components),
        TextComponent=Text, send_service=NS(_send_to_target_with_message=sender),
        CLI_PLATFORM_NAME="cli", render_cli_message=Mock(), is_bot_self=lambda *args: False,
        to_display_message_id=str)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    arguments = {"msg_id": target.message_id, "reply_reference": "结合当前话题简短作答"}
    invocation = NS(arguments=arguments, tool_name="reply", call_id="offline-call")
    return NS(run=lambda: namespace["handle_tool"](tool_ctx, invocation), runtime=runtime, target=target,
              reservation=reservation, generation=generation, replyer=replyer, sender=sender, quote=quote)


class ReplyWindowSenderTests(IsolatedAsyncioTestCase):


    async def test_exhausted_window_is_blocked_before_generation_or_sending(self):
        case = fixture()
        case.runtime.reserve_planner_reply.return_value = None
        result = await case.run()
        self.assertFalse(result.success)
        self.assertTrue(result.metadata["reply_quota_blocked"])
        case.replyer.generate_reply_with_context.assert_not_awaited()
        case.sender.assert_not_awaited()
        case.runtime.finish_planner_reply.assert_not_called()


    async def test_regular_reply_preserves_quote_configuration_and_counts_split_once(self):
        case = fixture()
        result = await case.run()
        self.assertTrue(result.success)
        self.assertTrue(result.metadata["reply_sent"])
        self.assertEqual(case.sender.await_count, 2)
        self.assertFalse(case.sender.await_args_list[0].kwargs["set_reply"])
        case.quote.assert_awaited_once()
        case.runtime.finish_planner_reply.assert_called_once_with(case.reservation, success=True)

    async def test_empty_generation_releases_the_reserved_slot(self):
        case = fixture()
        case.generation.completion.response_text = ""
        result = await case.run()
        self.assertFalse(result.success)
        self.assertFalse(result.metadata["reply_sent"])
        case.sender.assert_not_awaited()
        case.runtime.finish_planner_reply.assert_called_once_with(case.reservation, success=False)

    async def test_partial_delivery_counts_once_even_when_later_segment_fails(self):
        case = fixture()
        case.sender.side_effect = [NS(message_id="sent-1"), None]
        result = await case.run()
        self.assertFalse(result.success)
        self.assertTrue(result.metadata["reply_sent"])
        case.runtime.finish_planner_reply.assert_called_once_with(case.reservation, success=True)

    async def test_cancel_after_a_visible_segment_keeps_one_successful_reply(self):
        case = fixture()
        case.sender.side_effect = [NS(message_id="sent-1"), asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):
            await case.run()
        case.runtime.finish_planner_reply.assert_called_once_with(case.reservation, success=True)
