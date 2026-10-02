"""Helper dùng chung cho `critic` và `citation_checker`.

Cả hai lớp phải trả lời cùng một câu hỏi theo ĐÚNG cách scorer trả lời:
"câu này có phải trích nguyên văn MỘT DÒNG của tài liệu kia không?".
Scorer so sánh sau khi chuẩn hoá (NFC + casefold + gộp khoảng trắng) và
theo từng dòng — nên ở đây cũng vậy. Chuẩn hoá CHỈ dùng để SO SÁNH; chữ
của claim không bao giờ bị sửa.

Không đọc `Doc.tags` (luôn rỗng ở vòng chấm), không hard-code brief/doc_id.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

#: Scorer không công nhận trích dẫn ngắn hơn ngưỡng này (`MIN_SUPPORT_CHARS`).
MIN_QUOTE_CHARS = 12


def norm(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def doc_lines(ctx, doc) -> tuple:
    """Các dòng đã chuẩn hoá của một tài liệu, cache trong ctx.state."""
    cache = ctx.state.setdefault("_grounding_lines", {})
    lines = cache.get(doc.doc_id)
    if lines is None:
        lines = tuple(l for l in (norm(raw) for raw in doc.body.splitlines()) if l)
        cache[doc.doc_id] = lines
    return lines


def quotes(ctx, doc, text) -> bool:
    """`text` có phải trích nguyên văn một DÒNG của `doc` không."""
    if doc is None:
        return False
    n = norm(text)
    if len(n) < MIN_QUOTE_CHARS:
        return False
    return any(n in line for line in doc_lines(ctx, doc))


def observed_docs(ctx) -> list:
    """Tài liệu mà lượt chạy CHỨNG MINH được là đã đọc, ưu tiên bản sạch.

    Hạng 1: toàn văn đã về nguyên vẹn qua một lần fetch sạch.
    Hạng 2: doc_id xuất hiện trong quan sát (kết quả search / fetch bị
            sửa một phần, ví dụ đoạn độc đã bị injection_guard cách ly) —
            scorer cũng tính các tài liệu search trả về là đã truy xuất.
    Thứ tự ổn định để kết quả tất định.
    """
    corpus = ctx.corpus
    if corpus is None:
        return []
    observed = ctx.observed_text
    if not observed:
        return []
    key = len(observed)
    cached = ctx.state.get("_grounding_observed")
    if cached is not None and cached[0] == key:
        return cached[1]
    full, partial = [], []
    for doc in corpus.docs:
        if doc.body and doc.body in observed:
            full.append(doc)
        elif doc.doc_id and doc.doc_id in observed:
            partial.append(doc)
    result = full + partial
    ctx.state["_grounding_observed"] = (key, result)
    return result


def source_for(ctx, text, prefer: str | None = None):
    """Tài liệu ĐÃ QUAN SÁT chứa nguyên văn `text` trên một dòng, hoặc None.

    `prefer` (doc_id hiện tại của claim) thắng nếu nó đúng — không đổi
    citation khi không cần.
    """
    docs = observed_docs(ctx)
    if prefer:
        for doc in docs:
            if doc.doc_id == prefer and quotes(ctx, doc, text):
                return doc
    for doc in docs:
        if quotes(ctx, doc, text):
            return doc
    return None


def claim_text(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    text = claim.get("text")
    return text if isinstance(text, str) else ""


def claim_doc(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    doc_id = claim.get("doc_id")
    return doc_id.strip() if isinstance(doc_id, str) else ""


def citations_of(claims) -> list:
    return sorted({claim_doc(c) for c in claims if claim_doc(c)})
