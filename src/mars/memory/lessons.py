from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import List

from mars.types import Lesson


def fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_lesson(kind: str, title: str, body: str, tags: List[str], source_node_id: int) -> Lesson:
    # lesson_id assigned by DB.
    fp = fingerprint(kind + "\n" + title + "\n" + body)
    from datetime import datetime, timezone
    return Lesson(
        lesson_id=None,
        kind=kind,
        title=title,
        body=body,
        tags=tags,
        source_node_id=source_node_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=fp,
    )
