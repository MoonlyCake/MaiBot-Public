"""黑话释义进入正文生成指导的定向离线回归。"""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest import TestCase

import ast


ROOT = Path(__file__).resolve().parents[2]


class Text:
    def __init__(self, text):
        self.text = text


class Reference:
    def __init__(self, *entries, kind="jargon", content="内部参考正文"):
        self.reference_type, self.jargon_entries, self.content = kind, entries, content


class RealMessage:
    def __init__(self, text, *, source="user", notify=False, user_id="other"):
        self.source_kind = source
        self.original_message = NS(
            is_notify=notify, platform="qq", message_info=NS(user_info=NS(user_id=user_id)),
            raw_message=NS(components=[Text(text)]),
        )


def load_helper():
    path = ROOT / "src/maisaka/builtin_tool/reply.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_with_jargon_reply_reference")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[])
    namespace = dict(ReferenceMessage=Reference, ReferenceMessageType=NS(JARGON="jargon"),
                     SessionBackedMessage=RealMessage, TextComponent=Text,
                     is_bot_self=lambda _platform, user_id: user_id == "bot")
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


class ReplyJargonReferenceTests(TestCase):
    def setUp(self):
        self.append_reference = load_helper()
        self.arguments = {"reply_reference": "结合本次话题简短作答", "reply_style": "简短表达"}
        self.target = RealMessage("继续聊这个").original_message

    def test_matched_meaning_appended_without_mutating_original_direction(self):
        history = [RealMessage("今天又玉玉了"), Reference(("玉玉", "郁闷低落"))]
        result = self.append_reference(self.arguments, history, self.target)
        self.assertTrue(result["reply_reference"].startswith(self.arguments["reply_reference"] + "\n\n"))
        self.assertIn("玉玉：郁闷低落", result["reply_reference"])
        self.assertEqual(result["reply_style"], "简短表达")
        self.assertEqual(self.arguments["reply_reference"], "结合本次话题简短作答")

    def test_target_text_uses_matcher_case_and_whitespace_rules(self):
        history = [Reference(("GG WP", "打得好"))]
        target = RealMessage("这局 gG\n  Wp 啊").original_message
        result = self.append_reference(self.arguments, history, target)
        self.assertIn("GG WP：打得好", result["reply_reference"])

    def test_internal_references_planner_tools_and_old_unmatched_terms_are_excluded(self):
        history = [
            Reference(("旧黑话", "旧释义"), content="旧黑话以及 Planner 思考"),
            Reference(("内部黑话", "不该进入正文"), kind="memory"),
            NS(content="旧黑话 内部黑话 工具秘密"),
            RealMessage("旧黑话", source="guided_reply", user_id="bot"),
        ]
        self.assertIs(self.append_reference(self.arguments, history, self.target), self.arguments)

    def test_notifications_and_bot_messages_do_not_activate_jargon(self):
        history = [Reference(("玉玉", "郁闷低落")), RealMessage("玉玉", notify=True),
                   RealMessage("玉玉", user_id="bot")]
        self.assertIs(self.append_reference(self.arguments, history, self.target), self.arguments)

    def test_latest_definition_wins_deduplication_and_only_ten_terms_are_added(self):
        terms = [f"term{i:02d}" for i in range(12)]
        history = [RealMessage(" ".join(terms)), Reference(("term00", "旧释义")),
                   Reference(("TERM00", "新释义"), (" term00 ", "重复释义"),
                             *((term, f"释义{i}") for i, term in enumerate(terms[1:], 1)))]
        reference = self.append_reference(self.arguments, history, self.target)["reply_reference"]
        self.assertIn("TERM00：新释义", reference)
        self.assertNotIn("旧释义", reference)
        self.assertNotIn("重复释义", reference)
        self.assertEqual(sum("：" in line for line in reference.splitlines()[4:]), 10)
        self.assertNotIn("term10：", reference)

