"""Small live batches trigger learning without waiting for history trimming or ten messages."""

from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

import importlib.util
import logging

from pytests.chat_test.test_jargon_cached_counting_regression import load_learner


BUFFER_PATH = Path(__file__).resolve().parents[2] / "src/learners/pending_learning_buffer.py"
BUFFER_SPEC = importlib.util.spec_from_file_location("proactive_jargon_buffer", BUFFER_PATH)
BUFFER_MODULE = importlib.util.module_from_spec(BUFFER_SPEC)
BUFFER_SPEC.loader.exec_module(BUFFER_MODULE)


class TestJargonProactiveLearning(IsolatedAsyncioTestCase):
    async def test_five_real_messages_start_one_learning_batch_and_keep_prior_context(self):
        Learner = load_learner({
            "PendingLearningBuffer": BUFFER_MODULE.PendingLearningBuffer,
            "logger": logging.getLogger(__name__),
        })
        learner = Learner("session")
        learner.collect_context_messages = lambda _messages: None
        learner._learn_from_sources = AsyncMock(return_value=True)
        miner = object()
        sources = [SimpleNamespace(original_message=SimpleNamespace(message_id=str(index))) for index in range(5)]
        learner.pending_messages.extend(sources[:4])

        self.assertFalse(await learner.learn_from_context_messages([], miner))
        learner._learn_from_sources.assert_not_awaited()
        self.assertEqual(len(learner.pending_messages), 4)

        learner.pending_messages.extend(sources[4:])
        self.assertTrue(await learner.learn_from_context_messages([], miner))
        learner._learn_from_sources.assert_awaited_once_with(
            sources, jargon_miner=miner, learned_contents=None,
        )
