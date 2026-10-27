"""实际回复入口把知识召回结果交给 Replyer 的离线回归。"""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import ast

import pytest


ROOT = Path(__file__).resolve().parents[2]
FACT = "灰石剧场计划于2026年10月26日04:00开放，替换默祷圣祠。"


class Extension:
    def __init__(self, **_kwargs):
        self.reply_id = "offline-knowledge-reply"
        self.targets = []
        self.prepare = AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize("knowledge", [FACT, ""])
async def test_reserved_reply_passes_current_knowledge_to_replyer_without_sending(knowledge):
    path = ROOT / "src/maisaka/builtin_tool/reply.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    method = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_handle_reserved_reply")
    target = NS(message_id="arknights-message", processed_plain_text="灰石剧场何时开放？")
    generation = NS(error_message="offline-stop-before-send", monitor_detail=None)
    replyer = NS(generate_reply_with_context=AsyncMock(return_value=(False, generation)))
    injector = NS(build_injection_message=AsyncMock(return_value=knowledge))
    sender = AsyncMock()
    runtime = NS(
        session_id="group-arknights", log_prefix="offline", _chat_history=[],
        chat_stream=NS(platform="qq", is_group_session=True), _update_stage_status=Mock(),
    )
    tool_ctx = NS(
        runtime=runtime,
        build_failure_result=lambda _name, content, **kwargs: NS(
            success=False, content=content, metadata=kwargs.get("metadata", {}),
        ),
    )
    namespace = {
        "ReplyExtensionExecution": Extension,
        "ToolAvailabilityContext": lambda **kwargs: NS(**kwargs),
        "replyer_manager": NS(get_replyer=Mock(return_value=replyer)),
        "knowledge_memory_injector": injector,
        "_with_jargon_reply_reference": lambda arguments, *_args: arguments,
        "_find_recent_reply_to_target": Mock(return_value=""),
        "build_reply_monitor_detail": lambda _result: {},
        "_build_monitor_metadata": lambda _result, reply_id: {"reply_id": reply_id},
        "send_service": NS(_send_to_target_with_message=sender),
        "logger": Mock(),
    }
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    planner_reference = "已检索但没有资料，不能确定具体时间。"
    arguments = {"reply_reference": planner_reference}
    send_results = []

    result = await namespace[method.name](
        tool_ctx, NS(tool_name="reply", call_id="offline-call"), {}, arguments,
        target.message_id, target, object(), send_results,
    )

    injector.build_injection_message.assert_awaited_once_with(
        session_id="group-arknights", source_messages=[target],
    )
    replyer.generate_reply_with_context.assert_awaited_once()
    actual_request = replyer.generate_reply_with_context.await_args.kwargs
    assert actual_request["reply_message"] is target
    assert actual_request["stream_id"] == "group-arknights"
    expected_reference = f"{planner_reference}\n\n{FACT}" if knowledge else planner_reference
    assert actual_request["reply_tool_args"]["reply_reference"] == expected_reference
    assert result.success is False
    assert send_results == []
    sender.assert_not_awaited()
