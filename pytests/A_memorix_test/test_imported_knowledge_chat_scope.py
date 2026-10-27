from pathlib import Path

import pytest

from src.A_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel
from src.A_memorix.core.runtime.services.search_hit_processing_service import MemorySearchHitProcessingService
from src.A_memorix.core.storage.metadata_store import MetadataStore


@pytest.mark.parametrize("chat_id", ["session-current", "session-unrelated"])
def test_imported_global_knowledge_is_visible_without_exposing_other_chats(
    tmp_path: Path,
    chat_id: str,
) -> None:
    store = MetadataStore(data_dir=tmp_path / "metadata")
    store.connect()
    try:
        global_hash = store.add_paragraph(
            "明日方舟灰石剧场计划于2026年10月26日开放。",
            source="https://ak.hypergryph.com/news/1454",
            metadata={"scope_type": "global"},
            knowledge_type="factual",
        )
        current_hash = store.add_paragraph(
            "当前聊天的私有资料。",
            source="person_fact:current",
            metadata={"scope_type": "chat", "chat_id": "session-current"},
        )
        other_hash = store.add_paragraph(
            "其他聊天的私有资料。",
            source="person_fact:other",
            metadata={"scope_type": "chat", "chat_id": "session-other"},
        )
        kernel = SDKMemoryKernel(plugin_root=tmp_path, config={})
        kernel.metadata_store = store
        service = MemorySearchHitProcessingService(kernel)

        scope = service._resolve_retrieval_scope(chat_id)

        assert scope is not None
        expected = {global_hash, current_hash} if chat_id == "session-current" else {global_hash}
        assert scope.paragraph_ids == expected
        assert other_hash not in scope.paragraph_ids
    finally:
        store.close()
