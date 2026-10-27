"""按当前话题从本机公开知识库准备参考，不调用聊天模型或向量接口。"""

from __future__ import annotations

from collections import OrderedDict
from time import monotonic
from typing import Any, Sequence

import asyncio
import re

from src.A_memorix.runtime_registry import get_runtime_kernel
from src.chat.message_receive.message import SessionMessage
from src.common.logger import get_logger
from src.services.bot_account_service import is_bot_self

logger = get_logger("maisaka_knowledge_memory")

KNOWLEDGE_MEMORY_REFERENCE_MARKER = "【知识库资料-内部参考】"
_TOPIC = re.compile(r"主题：([^｜】]+)")
_QUOTED_TITLE = re.compile(r"[「《]([^「」《》]{2,80})[」》]")
_TOPIC_SUFFIX = re.compile(r"(?:剿灭委托预告|更新状态|官方预告|前瞻直播|预告)$")
_DOMAIN = re.compile(r"明日方舟|(?<![A-Za-z0-9_])Arknights(?![A-Za-z0-9_])|合成玉|黄票|绿票|干员", re.I)
_CATALOG_TTL = 60.0
_RESULT_TTL = 90.0
_MAX_CACHE_ENTRIES = 256
_MAX_REFERENCE_CHARS = 3500


class KnowledgeMemoryInjector:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._catalog_store: Any = None
        self._catalog_expires_at = 0.0
        self._paragraphs: dict[str, dict[str, Any]] = {}
        self._topic_ids: dict[str, set[str]] = {}
        self._topic_pattern: re.Pattern[str] | None = None
        self._results: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()

    async def build_injection_message(
        self,
        *,
        session_id: str,
        source_messages: Sequence[SessionMessage],
    ) -> str:
        texts = [
            " ".join(message.processed_plain_text.split())[:1200]
            for message in reversed(source_messages[-6:])
            if not message.is_notify and not message.is_command
            and not is_bot_self(message.platform, message.message_info.user_info.user_id)
            and message.processed_plain_text.strip()
        ]
        kernel = get_runtime_kernel()
        if not texts or kernel is None or kernel.metadata_store is None or kernel.sparse_index is None:
            return ""
        async with self._lock:
            return await asyncio.to_thread(self._build_reference, kernel, session_id, texts)

    def _load_catalog(self, store: Any, now: float) -> None:
        if self._catalog_store is store and now < self._catalog_expires_at:
            return
        rows = store.query(
            """
            SELECT hash, content, source FROM paragraphs
            WHERE (is_deleted IS NULL OR is_deleted = 0)
              AND content LIKE '【明日方舟｜%'
              AND json_extract(
                    CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END,
                    '$.scope_type'
                  ) = 'global'
            """
        )
        self._paragraphs = {row["hash"]: row for row in rows}
        self._topic_ids = {}
        for row in rows:
            match = _TOPIC.search(row["content"])
            if match is None:
                continue
            topic = match.group(1).strip()
            aliases = {topic, _TOPIC_SUFFIX.sub("", topic)}
            aliases.update(
                title for title in _QUOTED_TITLE.findall(row["content"])
                if topic.startswith(title) or title.startswith(topic)
            )
            for alias in aliases:
                if len(alias) >= 2:
                    self._topic_ids.setdefault(alias.casefold(), set()).add(row["hash"])
        alternatives = []
        for topic in sorted(self._topic_ids, key=len, reverse=True):
            escaped = re.escape(topic)
            if topic[0].isascii() and topic[0].isalnum():
                escaped = r"(?<![A-Za-z0-9_])" + escaped
            if topic[-1].isascii() and topic[-1].isalnum():
                escaped += r"(?![A-Za-z0-9_])"
            alternatives.append(escaped)
        self._topic_pattern = re.compile("|".join(alternatives), re.I) if alternatives else None
        self._catalog_store = store
        self._catalog_expires_at = now + _CATALOG_TTL
        self._results.clear()

    def _build_reference(self, kernel: Any, session_id: str, texts: Sequence[str]) -> str:
        now = monotonic()
        self._load_catalog(kernel.metadata_store, now)
        if not self._paragraphs:
            return ""
        for text in texts:
            topics = list(dict.fromkeys(
                match.group(0).casefold()
                for match in self._topic_pattern.finditer(text)
            )) if self._topic_pattern is not None else []
            if not topics and _DOMAIN.search(text) is None:
                continue
            key = (session_id, text)
            cached = self._results.get(key)
            if cached is not None and now < cached[0]:
                self._results.move_to_end(key)
                return cached[1]
            allowed_ids = (
                set().union(*(self._topic_ids[topic] for topic in topics))
                if topics else set(self._paragraphs)
            )
            # 主题已限定候选范围，检索正文中的细节，不拼接每条资料都有的领域前缀。
            rows = kernel.sparse_index.search(query=text, k=4, allowed_ids=allowed_ids)
            reference = self._format_reference(rows, allowed_ids)
            self._results[key] = (now + _RESULT_TTL, reference)
            self._results.move_to_end(key)
            while len(self._results) > _MAX_CACHE_ENTRIES:
                self._results.popitem(last=False)
            if reference:
                logger.info(f"公开知识自动召回: session_id={session_id} topics={topics} hits={len(rows)}")
            return reference
        return ""

    def _format_reference(self, rows: Sequence[dict[str, Any]], allowed_ids: set[str]) -> str:
        reference = (
            KNOWLEDGE_MEMORY_REFERENCE_MARKER
            + "\n以下是本机已导入的公开知识，可用作事实依据并自然简短接话。"
            "以资料标注日期为准，区分预告与已实装；不要复述这段内部说明。"
        )
        count = 0
        for row in rows:
            paragraph_id = row["hash"]
            if paragraph_id not in allowed_ids:
                continue
            paragraph = self._paragraphs[paragraph_id]
            block = f"\n{count + 1}. {paragraph['content']}\n来源：{paragraph['source']}"
            remaining = _MAX_REFERENCE_CHARS - len(reference)
            if len(block) > remaining:
                if count == 0:
                    reference += block[:remaining]
                    count += 1
                break
            reference += block
            count += 1
        return reference if count else ""


knowledge_memory_injector = KnowledgeMemoryInjector()
