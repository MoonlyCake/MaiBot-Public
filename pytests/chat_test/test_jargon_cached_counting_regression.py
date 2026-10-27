"""Keep old-candidate receipts independent of new-candidate model output."""

from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

import ast
import logging
import re


def load_learner(namespace):
    path = Path(__file__).resolve().parents[2] / "src/learners/jargon_learner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    learner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JargonLearner")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), learner],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["JargonLearner"]


class CachedJargonCountingRegression(IsolatedAsyncioTestCase):
    def setUp(self):
        self.receipts = set()
        self.cached = []
        self.miner = SimpleNamespace(
            session_id="original",
            session_name="original name",
            get_cached_jargons=lambda: self.cached,
        )

        async def initialize():
            self.cached.append("测试黑话")

        async def persist(entries):
            self.assertEqual(self.miner.session_id, "actual")
            self.assertEqual(self.miner.session_name, "actual name")
            before = len(self.receipts)
            for entry in entries:
                for group in entry["evidence_messages"]:
                    for ref in group:
                        if ref.get("is_occurrence"):
                            self.receipts.add((ref["platform"], ref["message_id"]))
            return 0, int(len(self.receipts) > before)

        self.miner.initialize = AsyncMock(side_effect=initialize)
        self.miner.process_extracted_entries = AsyncMock(side_effect=persist)
        template = SimpleNamespace(add_context=lambda *_args: None)
        self.model = SimpleNamespace(generate_response_with_context=AsyncMock())
        namespace = {
            "logger": logging.getLogger(__name__),
            "global_config": SimpleNamespace(bot=SimpleNamespace(nickname="")),
            "prompt_manager": SimpleNamespace(
                get_prompt=lambda _name: template,
                render_prompt=AsyncMock(return_value="extract candidates"),
            ),
            "jargon_learn_model": self.model,
            "parse_jargon_response": lambda _response: [],
            "MESSAGE_OPEN_TAG_PATTERN": re.compile(r"^<message\b[^>]*>", re.I | re.S),
            "jargon_occurs_in_text": lambda content, text: content in text,
        }
        Learner = load_learner(namespace)
        self.learner = Learner.__new__(Learner)
        self.learner.session_id = "original"
        self.learner.learning_status = "running"
        self.learner.learning_detail = ""
        self.learner._get_session_display_name = lambda _session: "actual name"
        self.learner._build_multi_learning_messages = lambda *_args: []
        self.learner._log_learning_context_preview = lambda *_args, **_kwargs: None
        self.learner._log_jargon_update_process = lambda **_kwargs: None
        self.sources = [
            SimpleNamespace(
                content=f'<message>第 {index} 条测试黑话',
                original_message=SimpleNamespace(platform="qq", message_id=str(index)),
            )
            for index in range(2)
        ]

    async def run_batch(self):
        learned = []
        changed = await self.learner._run_learning_batch(
            self.sources, learning_session_id="actual", jargon_miner=self.miner, learned_contents=learned,
        )
        self.assertTrue(changed)
        self.assertEqual(self.receipts, {("qq", "0"), ("qq", "1")})
        self.assertEqual(self.learner.learning_status, "completed")
        self.assertIn("测试黑话", learned)
        self.assertEqual((self.miner.session_id, self.miner.session_name), ("original", "original name"))
        self.miner.initialize.assert_awaited_once()
        self.miner.process_extracted_entries.assert_awaited_once()

    async def test_existing_candidates_count_before_extraction_fails(self):
        async def fail(*_args, **_kwargs):
            self.assertEqual(len(self.receipts), 2)
            self.assertEqual(self.miner.session_id, "original")
            raise RuntimeError("model unavailable")

        self.model.generate_response_with_context.side_effect = fail
        await self.run_batch()
        self.assertIn("新候选抽取失败", self.learner.learning_detail)

    async def test_existing_candidates_count_when_extraction_is_empty(self):
        async def empty(*_args, **_kwargs):
            self.assertEqual(len(self.receipts), 2)
            return SimpleNamespace(response="[]")

        self.model.generate_response_with_context.side_effect = empty
        await self.run_batch()
