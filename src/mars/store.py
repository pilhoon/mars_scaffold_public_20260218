from __future__ import annotations
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

from mars.types import ActionType, NodeStatus, NodeRecord, Lesson


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS nodes (
  node_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id     INTEGER,
  action        TEXT NOT NULL,
  status        TEXT NOT NULL,
  repo_path     TEXT NOT NULL,
  metric        REAL,
  runtime_sec   REAL,
  reward        REAL,
  visits        INTEGER NOT NULL DEFAULT 0,
  value_sum     REAL NOT NULL DEFAULT 0.0,
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);

CREATE TABLE IF NOT EXISTS best_state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
  lesson_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL,
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,
  tags_json   TEXT NOT NULL,
  source_node_id INTEGER NOT NULL,
  created_at  REAL NOT NULL,
  fingerprint TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lessons_kind ON lessons(kind);
CREATE INDEX IF NOT EXISTS idx_lessons_fp ON lessons(fingerprint);

CREATE TABLE IF NOT EXISTS llm_calls (
  call_seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   REAL NOT NULL,
  process      TEXT,
  worker_id    TEXT,
  iter_index   INTEGER,
  stage        TEXT,
  job_id       TEXT,
  job_type     TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at ON llm_calls(created_at);
"""


def connect(run_dir: str | Path) -> sqlite3.Connection:
    db_path = Path(run_dir) / "state" / "mars.sqlite"
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def _now() -> float:
    return time.time()


# ---- nodes ----

def create_node(conn: sqlite3.Connection, parent_id: Optional[int], action: ActionType, status: NodeStatus, repo_path: str) -> int:
    now = _now()
    cur = conn.execute(
        """INSERT INTO nodes(parent_id, action, status, repo_path, created_at, updated_at)
             VALUES(?, ?, ?, ?, ?, ?)""",
        (parent_id, action.value, status.value, repo_path, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_node_status(conn: sqlite3.Connection, node_id: int, status: NodeStatus) -> None:
    conn.execute("UPDATE nodes SET status=?, updated_at=? WHERE node_id=?", (status.value, _now(), node_id))
    conn.commit()


def update_node_result(conn: sqlite3.Connection, node_id: int, metric: Optional[float], runtime_sec: Optional[float], reward: Optional[float], status: NodeStatus) -> None:
    conn.execute(
        """UPDATE nodes
             SET metric=?, runtime_sec=?, reward=?, status=?, updated_at=?
             WHERE node_id=?""",
        (metric, runtime_sec, reward, status.value, _now(), node_id),
    )
    conn.commit()


def inc_visit_and_value(conn: sqlite3.Connection, node_id: int, reward: float) -> None:
    # value_sum += reward, visits += 1
    conn.execute(
        """UPDATE nodes
             SET visits = visits + 1,
                 value_sum = value_sum + ?,
                 updated_at = ?
             WHERE node_id=?""",
        (reward, _now(), node_id),
    )
    conn.commit()


def get_node(conn: sqlite3.Connection, node_id: int) -> NodeRecord:
    row = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    if row is None:
        raise KeyError(f"node_id not found: {node_id}")
    return NodeRecord(
        node_id=int(row["node_id"]),
        parent_id=row["parent_id"],
        action=ActionType(row["action"]),
        status=NodeStatus(row["status"]),
        repo_path=str(row["repo_path"]),
        metric=row["metric"],
        runtime_sec=row["runtime_sec"],
        reward=row["reward"],
        visits=int(row["visits"]),
        value_sum=float(row["value_sum"]),
    )


def list_children(conn: sqlite3.Connection, parent_id: int) -> List[int]:
    rows = conn.execute("SELECT node_id FROM nodes WHERE parent_id=? ORDER BY node_id ASC", (parent_id,)).fetchall()
    return [int(r[0]) for r in rows]


def list_nodes(conn: sqlite3.Connection, limit: int = 1000) -> List[NodeRecord]:
    rows = conn.execute("SELECT * FROM nodes ORDER BY node_id ASC LIMIT ?", (limit,)).fetchall()
    out: List[NodeRecord] = []
    for row in rows:
        out.append(NodeRecord(
            node_id=int(row["node_id"]),
            parent_id=row["parent_id"],
            action=ActionType(row["action"]),
            status=NodeStatus(row["status"]),
            repo_path=str(row["repo_path"]),
            metric=row["metric"],
            runtime_sec=row["runtime_sec"],
            reward=row["reward"],
            visits=int(row["visits"]),
            value_sum=float(row["value_sum"]),
        ))
    return out


# ---- best_state ----

def set_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """INSERT INTO best_state(key, value) VALUES(?, ?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (key, value),
    )
    conn.commit()


def get_kv(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM best_state WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0])


def record_llm_call(
    conn: sqlite3.Connection,
    *,
    process: str = "",
    worker_id: str = "",
    iter_index: Optional[int] = None,
    stage: str = "",
    job_id: str = "",
    job_type: str = "",
) -> int:
    cur = conn.execute(
        """INSERT INTO llm_calls(created_at, process, worker_id, iter_index, stage, job_id, job_type)
             VALUES(?, ?, ?, ?, ?, ?, ?)""",
        (_now(), process, worker_id, iter_index, stage, job_id, job_type),
    )
    conn.commit()
    return int(cur.lastrowid)


def reserve_rate_limit_slot(
    conn: sqlite3.Connection,
    *,
    limiter_name: str,
    interval_sec: float,
) -> tuple[float, float]:
    """Reserve the next global rate-limit slot.

    Returns:
        (slot_unix_ts, wait_sec_from_now)
    """
    interval = max(0.0, float(interval_sec))
    now = _now()
    if interval <= 0.0:
        return now, 0.0

    key = f"rate_limit.{limiter_name}.next_allowed_at"
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT value FROM best_state WHERE key=?", (key,)).fetchone()
            next_allowed = 0.0
            if row is not None:
                try:
                    next_allowed = float(row[0])
                except Exception:
                    next_allowed = 0.0

            slot_ts = max(now, next_allowed)
            new_next_allowed = slot_ts + interval
            conn.execute(
                """INSERT INTO best_state(key, value) VALUES(?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (key, f"{new_next_allowed:.6f}"),
            )
            conn.commit()
            wait_sec = max(0.0, slot_ts - now)
            return slot_ts, wait_sec
        except sqlite3.OperationalError as e:
            conn.rollback()
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(0.05)
                now = _now()
                continue
            raise


# ---- lessons ----

def add_lesson(conn: sqlite3.Connection, lesson: Lesson) -> int:
    import json
    cur = conn.execute(
        """INSERT INTO lessons(kind, title, body, tags_json, source_node_id, created_at, fingerprint)
             VALUES(?, ?, ?, ?, ?, ?, ?)""",
        (lesson.kind, lesson.title, lesson.body, json.dumps(lesson.tags), lesson.source_node_id, _now(), lesson.fingerprint),
    )
    conn.commit()
    return int(cur.lastrowid)


def lesson_exists_by_fingerprint(conn: sqlite3.Connection, fingerprint: str) -> bool:
    row = conn.execute("SELECT 1 FROM lessons WHERE fingerprint=? LIMIT 1", (fingerprint,)).fetchone()
    return row is not None


def recent_lessons(conn: sqlite3.Connection, kind: str, k: int) -> List[Lesson]:
    import json
    rows = conn.execute(
        """SELECT * FROM lessons WHERE kind=? ORDER BY lesson_id DESC LIMIT ?""",
        (kind, k),
    ).fetchall()
    out: List[Lesson] = []
    for r in rows:
        out.append(Lesson(
            lesson_id=int(r["lesson_id"]),
            kind=str(r["kind"]),
            title=str(r["title"]),
            body=str(r["body"]),
            tags=list(json.loads(r["tags_json"])),
            source_node_id=int(r["source_node_id"]),
            created_at=str(r["created_at"]),
            fingerprint=str(r["fingerprint"]),
        ))
    return out
