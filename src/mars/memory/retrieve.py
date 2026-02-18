from __future__ import annotations
from typing import List
import sqlite3

from mars.types import Lesson
from mars.store import recent_lessons


def get_recent_lessons(conn: sqlite3.Connection, kind: str, k: int) -> List[Lesson]:
    return recent_lessons(conn, kind=kind, k=k)
