from pathlib import Path
from types import SimpleNamespace

import pytest

from src.A_memorix.core.storage.metadata_store import MetadataStore
from src.A_memorix.core.retrieval.sparse_bm25 import SparseBM25Config, SparseBM25Index
from src.maisaka.memory import knowledge_injector


class RecordingSparseIndex:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> list[dict[str, str]]:
        self.calls.append(kwargs)
        # 同时返回私有行，确认注入端只接受明确列入公开目录的候选。
        return self.rows


def message(text: str, *, user: str = "external", notify: bool = False, command: bool = False):
    return SimpleNamespace(
        processed_plain_text=text, platform="qq", is_notify=notify, is_command=command,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user)),
    )


@pytest.fixture
def public_knowledge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = MetadataStore(data_dir=tmp_path / "metadata")
    store.connect()
    content = (
        "【明日方舟｜国服简体中文｜核实日期：2026-10-24｜主题：灰石剧场剿灭委托预告】"
        "「灰石剧场」计划2026年10月26日04:00开放，通关主线2-8解锁。"
    )
    public_id = store.add_paragraph(
        content, source="https://ak.hypergryph.com/news/1454", metadata={"scope_type": "global"}
    )
    private_id = store.add_paragraph(
        "【明日方舟｜主题：灰石剧场】某人的私有聊天记忆，不应泄漏。",
        source="chat_summary:private-other",
        metadata={"scope_type": "chat", "chat_id": "private-other"},
    )
    sparse = RecordingSparseIndex([
        {"hash": public_id, "content": content},
        {"hash": private_id, "content": "私有聊天记忆，不应泄漏"},
    ])
    kernel = SimpleNamespace(metadata_store=store, sparse_index=sparse)
    monkeypatch.setattr(knowledge_injector, "get_runtime_kernel", lambda: kernel)
    monkeypatch.setattr(knowledge_injector, "is_bot_self", lambda _platform, user: user == "bot")
    try:
        yield knowledge_injector.KnowledgeMemoryInjector(), sparse, public_id, private_id
    finally:
        store.close()


@pytest.mark.asyncio
async def test_topic_chat_without_question_or_mention_recalls_and_reuses_result(public_knowledge) -> None:
    injector, sparse, public_id, _ = public_knowledge
    incoming = message("灰石剧场好像后天开放")
    reference = await injector.build_injection_message(session_id="group-1", source_messages=[incoming])
    again = await injector.build_injection_message(session_id="group-1", source_messages=[incoming])

    assert "2026年10月26日04:00" in reference
    assert "https://ak.hypergryph.com/news/1454" in reference
    assert knowledge_injector.KNOWLEDGE_MEMORY_REFERENCE_MARKER in reference
    assert again == reference
    assert len(sparse.calls) == 1
    assert sparse.calls[0]["allowed_ids"] == {public_id}


@pytest.mark.asyncio
async def test_unrelated_chat_does_not_search(public_knowledge) -> None:
    injector, sparse, _, _ = public_knowledge
    reference = await injector.build_injection_message(
        session_id="group-1", source_messages=[message("晚上一起吃点什么")]
    )

    assert reference == ""
    assert sparse.calls == []


@pytest.mark.asyncio
async def test_domain_chat_retrieves_only_global_knowledge(public_knowledge) -> None:
    injector, sparse, public_id, private_id = public_knowledge
    reference = await injector.build_injection_message(
        session_id="group-2", source_messages=[message("明日方舟最近有什么更新")]
    )

    assert sparse.calls[0]["allowed_ids"] == {public_id}
    assert private_id not in sparse.calls[0]["allowed_ids"]
    assert "私有聊天记忆" not in reference
    assert "灰石剧场" in reference


@pytest.mark.asyncio
async def test_topic_discussion_retrieves_real_local_fts_results(public_knowledge, monkeypatch) -> None:
    injector, _, _, _ = public_knowledge
    kernel = knowledge_injector.get_runtime_kernel()
    index = SparseBM25Index(kernel.metadata_store, SparseBM25Config(enable_ngram_fallback_index=False))
    monkeypatch.setattr(kernel, "sparse_index", index)
    try:
        reference = await injector.build_injection_message(
            session_id="group-fts", source_messages=[
                message("灰石剧场后天开，终于换剿灭了")
            ],
        )
        assert "2026年10月26日04:00" in reference
        assert "通关主线2-8" in reference
        assert "私有聊天记忆" not in reference
    finally:
        index.unload()


@pytest.mark.asyncio
async def test_bot_echoes_and_notices_do_not_trigger_knowledge(public_knowledge) -> None:
    injector, sparse, _, _ = public_knowledge
    reference = await injector.build_injection_message(
        session_id="group-1", source_messages=[
            message("灰石剧场", user="bot"), message("明日方舟", notify=True),
            message("明日方舟", command=True), message("晚上吃什么"),
        ],
    )
    assert reference == ""
    assert sparse.calls == []
