"""Offline regressions: real learner/miner methods, isolated from app and LLM startup."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

import ast
import asyncio
import json
import re
import sqlite3
import threading

ROOT = Path(__file__).resolve().parents[2]


class Record(SimpleNamespace):
    def __init__(self, **values):
        super().__init__(is_complete=False, last_inference_count=0)
        self.__dict__.update(values)


class Query:
    def __init__(self, model):
        self.model, self.values = model, {}

    def filter_by(self, **values):
        self.values.update(values)
        return self

    def limit(self, _limit):
        return self


class Store:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.execute("create table records (id integer primary key, value text)")
        self.messages = {}

    @contextmanager
    def session(self, **_kwargs):
        yield self
        self.connection.commit()

    def exec(self, query):
        if query.model is Record:
            rows = [Record(**json.loads(row[0])) for row in self.connection.execute("select value from records")]
        else:
            rows = list(self.messages.values())
        matching = [r for r in rows if all(getattr(r, k, None) == v for k, v in query.values.items())]
        return SimpleNamespace(all=lambda: matching, first=lambda: next(iter(matching), None))

    def add(self, record):
        if getattr(record, "id", None) is None:
            record.id = self.connection.execute("select coalesce(max(id),0)+1 from records").fetchone()[0]
        self.connection.execute("insert or replace into records values (?,?)", (record.id, json.dumps(vars(record), default=str)))

    def flush(self):
        pass

    def row(self):
        return self.exec(Query(Record)).first()


def isolated_class(filename, class_name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in (class_name, "jargon_occurs_in_text")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[class_name]


class TestJargonMessageCounts(IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        logger = SimpleNamespace(**{name: lambda *_a, **_k: None for name in ("debug", "info", "warning", "error")})
        self.ns = dict(asyncio=asyncio, json=json, re=re, datetime=datetime, OrderedDict=OrderedDict,
                       Jargon=Record, Messages=object, select=Query, get_db_session=self.store.session,
                       JargonCreatedBy=SimpleNamespace(AI="AI", MANUAL="MANUAL"), logger=logger,
                       JARGON_INFERENCE_THRESHOLDS=[2, 4, 8, 25, 100], is_single_char_jargon=lambda text: len(text) == 1,
                       JargonConfigUtils=SimpleNamespace(resolve_jargon_group_scope=lambda _: ({"session"}, False)))
        self.Miner = isolated_class("src/learners/jargon_miner.py", "JargonMiner", self.ns)
        self.miner = self.new_miner()

    def tearDown(self):
        self.store.connection.close()

    def new_miner(self):
        miner = self.Miner("session", "test")
        miner._get_runtime_manager = lambda: SimpleNamespace(invoke_hook=AsyncMock(return_value=SimpleNamespace(aborted=False, kwargs={})))
        miner._infer_meaning_by_id = AsyncMock()
        return miner

    def refs(self, *ids, marked=True):
        return [[dict(platform="test", message_id=str(i), **({"is_occurrence": True} if marked else {})) for i in ids]]

    def entry(self, *ids):
        return dict(content="测试黑话", raw_content=set(), evidence_messages=self.refs(*ids))

    def old_record(self, ids, *, manual=False):
        self.store.add(Record(id=1, content="测试黑话", count=1, meaning="manual meaning" if manual else "",
                              evidence_messages=json.dumps(self.refs(*ids, marked=False)), session_id_dict='{"session":1}',
                              is_jargon=manual, is_complete=False, is_global=False, last_inference_count=0,
                              created_by="MANUAL" if manual else "AI"))
        for i in ids:
            self.store.messages[str(i)] = SimpleNamespace(platform="test", message_id=str(i), processed_plain_text="使用测试黑话")

    async def test_first_batch_two_distinct_sources_starts_initial_meaning(self):
        self.assertEqual(await self.miner.process_extracted_entries([self.entry(1, 2)]), (1, 0))
        await asyncio.sleep(0)
        self.assertEqual(self.store.row().count, 2)
        self.miner._infer_meaning_by_id.assert_awaited_once_with(1)

    async def test_legacy_sources_are_backfilled_once_and_survive_new_miner(self):
        self.old_record([1, 2, 3, 4])
        legacy_entry = self.entry(1, 2, 3, 4)
        legacy_entry["evidence_messages"] = self.refs(1, 2, 3, 4, marked=False)
        self.assertEqual(await self.miner.process_extracted_entries([legacy_entry]), (0, 1))
        await asyncio.sleep(0)
        self.assertEqual(self.store.row().count, 4)
        self.assertEqual(json.loads(self.store.row().session_id_dict), {"session": 4})
        restarted = self.new_miner()
        self.assertEqual(await restarted.process_extracted_entries([self.entry(1, 2, 3, 4)]), (0, 0))
        restarted._infer_meaning_by_id.assert_not_called()
        self.assertEqual(self.store.row().count, 4)

    async def test_overlap_counts_only_new_distinct_messages(self):
        await self.miner.process_extracted_entries([self.entry(1, 2)])
        await self.miner.process_extracted_entries([self.entry(2, 3), self.entry(3, 4)])
        self.assertEqual(self.store.row().count, 4)
        await self.miner.process_extracted_entries([self.entry(1, 2, 3, 4)])
        self.assertEqual(self.store.row().count, 4)
        await asyncio.sleep(0)

    async def test_inference_context_cleanup_preserves_durable_receipts(self):
        await self.miner.process_extracted_entries([self.entry(1, 2)])
        obj = SimpleNamespace(item_id=1, evidence_messages=self.store.row().evidence_messages)
        self.miner._cleanup_jargon_evidence_messages(obj, None)
        self.assertEqual(len(self.Miner._occurrence_keys(json.loads(self.store.row().evidence_messages))), 2)
        restarted = self.new_miner()
        await restarted.process_extracted_entries([self.entry(1, 2)])
        self.assertEqual(self.store.row().count, 2)

    async def test_manual_jargon_remains_untouched(self):
        self.old_record([1], manual=True)
        before = vars(self.store.row())
        self.assertEqual(await self.miner.process_extracted_entries([self.entry(1, 2, 3, 4)]), (0, 0))
        self.assertEqual(vars(self.store.row()), before)
        self.miner._infer_meaning_by_id.assert_not_called()

    def add_candidate(self, record_id, content, *, session_id="session", global_=False,
                      count=1, last_inference=0, complete=False, manual=False):
        self.store.add(Record(id=record_id, content=content, count=count, meaning="", evidence_messages=None,
                              session_id_dict=json.dumps({session_id: count}), is_jargon=False,
                              is_global=global_, is_complete=complete, last_inference_count=last_inference,
                              created_by="MANUAL" if manual else "AI"))

    async def test_initialize_restores_all_scoped_candidates_once_off_loop(self):
        self.ns["JargonConfigUtils"].resolve_jargon_group_scope = lambda _: ({"session", "shared"}, False)
        for index in range(60):
            self.add_candidate(index + 1, f"旧候选{index}")
        self.add_candidate(61, "共享候选", session_id="shared")
        self.add_candidate(62, "全局候选", session_id="other", global_=True)
        self.add_candidate(63, "其他群候选", session_id="other")
        self.add_candidate(64, " ")
        self.add_candidate(65, "啊")
        original_load = self.miner._load_scoped_jargons
        worker_threads = []

        def load_candidates():
            worker_threads.append(threading.get_ident())
            return original_load()

        self.miner._load_scoped_jargons = load_candidates
        await asyncio.gather(self.miner.initialize(), self.miner.initialize())
        cached = self.miner.get_cached_jargons()
        self.assertEqual(len(cached), 62)
        self.assertIn("旧候选0", cached)
        self.assertIn("共享候选", cached)
        self.assertIn("全局候选", cached)
        self.assertNotIn("其他群候选", cached)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], threading.get_ident())
        self.miner._infer_meaning_by_id.assert_not_called()

    async def test_initialize_recovers_only_unprocessed_inference_thresholds(self):
        cases = [(1, 2, 0), (2, 4, 4), (3, 8, 4), (4, 24, 8), (5, 25, 8), (6, 100, 25)]
        for record_id, count, last in cases:
            self.add_candidate(record_id, f"阈值候选{record_id}", count=count, last_inference=last)
        self.add_candidate(7, "已完成候选", count=100, last_inference=100, complete=True)
        self.add_candidate(8, "手动候选", count=100, manual=True)
        self.add_candidate(9, "不足阈值", count=1)
        await self.miner.initialize()
        await asyncio.sleep(0)
        await self.miner.initialize()
        self.assertEqual([call.args[0] for call in self.miner._infer_meaning_by_id.await_args_list], [1, 3, 4, 5, 6])

    def test_missing_meaning_retries_with_new_evidence_before_next_large_milestone(self):
        candidate = Record(created_by="AI", count=9, last_inference_count=8, meaning="")
        self.assertFalse(self.miner._should_infer_meaning(candidate))
        candidate.count = 10
        self.assertTrue(self.miner._should_infer_meaning(candidate))
        candidate.last_inference_count = 25
        candidate.count = 26
        self.assertFalse(self.miner._should_infer_meaning(candidate))
        candidate.count = 27
        self.assertTrue(self.miner._should_infer_meaning(candidate))

    def test_existing_meaning_keeps_milestones_and_manual_terms_are_not_rewritten(self):
        candidate = Record(created_by="AI", count=10, last_inference_count=8, meaning="已有可靠释义")
        self.assertFalse(self.miner._should_infer_meaning(candidate))
        candidate.count = 25
        self.assertTrue(self.miner._should_infer_meaning(candidate))
        candidate.created_by = "MANUAL"
        self.assertFalse(self.miner._should_infer_meaning(candidate))

    async def test_restored_inference_queue_runs_at_most_two_requests_at_once(self):
        for index in range(3):
            self.add_candidate(index + 1, f"等待释义{index}", count=2)
        self.ns["MaiJargon"] = SimpleNamespace(from_db_instance=lambda record: record)
        release, started = asyncio.Event(), asyncio.Event()
        active = 0
        peak = 0

        async def infer(_record):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
            await release.wait()
            active -= 1

        self.miner.infer_meaning = infer
        tasks = [asyncio.create_task(self.Miner._infer_meaning_by_id(self.miner, index + 1)) for index in range(3)]
        await started.wait()
        self.assertEqual(peak, 2)
        release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(peak, 2)

    async def test_existing_ai_candidate_enters_cache_after_update(self):
        self.old_record([1])
        await self.miner.process_extracted_entries([self.entry(2)])
        self.assertIn("测试黑话", self.miner.get_cached_jargons())
        self.assertEqual(self.store.row().count, 2)

    async def test_known_candidates_are_not_evicted_after_fifty_words(self):
        for index in range(70):
            self.miner._add_to_cache(f"候选{index}")
        self.assertEqual(len(self.miner.get_cached_jargons()), 70)
        self.assertIn("候选0", self.miner.get_cached_jargons())

    async def test_learner_counts_body_occurrences_even_with_one_model_candidate(self):
        namespace = dict(self.ns, global_config=SimpleNamespace(bot=SimpleNamespace(nickname="")),
                         MESSAGE_OPEN_TAG_PATTERN=re.compile(r"^<message\b[^>]*>", re.I | re.S))
        Learner = isolated_class("src/learners/jargon_learner.py", "JargonLearner", namespace)
        learner = Learner.__new__(Learner)
        learner._log_jargon_update_process = lambda **_kwargs: None
        miner = SimpleNamespace(process_extracted_entries=AsyncMock(return_value=(1, 0)))
        sources = [SimpleNamespace(content='<message nickname="测试黑话">\n' + ("测试黑话" if i < 4 else "其他正文"),
                                   original_message=SimpleNamespace(platform="test", message_id=str(i))) for i in range(10)]
        await learner._process_jargon_entries([("测试黑话", "1")], sources, miner)
        entries = miner.process_extracted_entries.await_args.args[0]
        self.assertEqual(self.Miner._occurrence_keys(entries[0]["evidence_messages"]), {("test", str(i)) for i in range(4)})

    async def test_latin_word_boundaries_do_not_count_substrings(self):
        match = self.ns["jargon_occurs_in_text"]
        self.assertFalse(match("foo", "food"))
        self.assertTrue(match("foo", "FOO!"))
