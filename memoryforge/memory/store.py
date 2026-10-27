"""Persistence for attempts, lessons and retrieval events.

One class, two dialects: PostgreSQL + pgvector (production, Docker) and
SQLite + numpy (laptop, CI). `open_store()` picks Postgres when
MF_DATABASE_URL is set and reachable, and falls back to SQLite otherwise.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger(__name__)

JSON_COLS = {"meta", "retrieved", "replay", "history", "features"}
VEC_COLS = {"embedding", "context"}

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, meta TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, namespace TEXT NOT NULL,
    condition TEXT NOT NULL, seed INT NOT NULL, phase TEXT NOT NULL, step INT NOT NULL,
    task_id TEXT NOT NULL, trial INT NOT NULL, code TEXT NOT NULL, passed INT NOT NULL,
    n_passed INT NOT NULL, n_tests INT NOT NULL, feedback TEXT, pattern TEXT, pattern_conf REAL,
    true_pattern TEXT, reflection TEXT, retrieved TEXT NOT NULL DEFAULT '[]', tokens INT NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS attempts_run_idx ON attempts (run_id, condition, phase);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, text TEXT NOT NULL,
    pattern TEXT NOT NULL, embedding BLOB NOT NULL, context BLOB NOT NULL, embedder TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate', source_task TEXT NOT NULL, source_attempt INT,
    created_step INT NOT NULL DEFAULT 0, alpha REAL NOT NULL DEFAULT 1, beta REAL NOT NULL DEFAULT 1,
    alpha0 REAL NOT NULL DEFAULT 1, beta0 REAL NOT NULL DEFAULT 1, uses INT NOT NULL DEFAULT 0,
    replay TEXT NOT NULL DEFAULT '{}', history TEXT NOT NULL DEFAULT '[]', merged_into INT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS lessons_ns_idx ON lessons (namespace, status);
CREATE TABLE IF NOT EXISTS retrieval_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id INT NOT NULL, lesson_id INT NOT NULL,
    namespace TEXT NOT NULL, rank INT NOT NULL, score REAL NOT NULL, features TEXT NOT NULL,
    withheld INT NOT NULL DEFAULT 0, passed INT, pattern_hit INT);
CREATE INDEX IF NOT EXISTS retrieval_ns_idx ON retrieval_events (namespace);
"""

BOOL_COLS = {"passed", "withheld", "pattern_hit"}


class Store:
    def __init__(self, conn: Any, dialect: str) -> None:
        self.conn = conn
        self.dialect = dialect
        self._lock = threading.RLock()
        self._ph = "?" if dialect == "sqlite" else "%s"

    # -- plumbing -----------------------------------------------------------
    def _sql(self, sql: str) -> str:
        return sql if self.dialect == "sqlite" else sql.replace("?", "%s")

    def _encode(self, col: str, value: Any) -> Any:
        if col in JSON_COLS:
            return json.dumps(value)
        if col in VEC_COLS:
            arr = np.asarray(value, dtype=np.float32)
            return arr.tobytes() if self.dialect == "sqlite" else arr
        if col in BOOL_COLS and value is not None and self.dialect == "sqlite":
            return int(bool(value))
        return value

    def _decode(self, row: dict) -> dict:
        for col in list(row):
            v = row[col]
            if col in JSON_COLS and isinstance(v, str):
                row[col] = json.loads(v)
            elif col in VEC_COLS and v is not None:
                row[col] = np.frombuffer(v, dtype=np.float32) if isinstance(v, (bytes, memoryview)) else np.asarray(v, dtype=np.float32)
            elif col in BOOL_COLS and v is not None:
                row[col] = bool(v)
        return row

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> Any:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(self._sql(sql), tuple(params))
            self.conn.commit()
            return cur

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(self._sql(sql), tuple(params))
            cols = [d[0] for d in cur.description]
            return [self._decode(dict(zip(cols, r))) for r in cur.fetchall()]

    def _insert(self, table: str, pk: str, fields: dict) -> int:
        cols = list(fields)
        vals = [self._encode(c, fields[c]) for c in cols]
        marks = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks})"
        if self.dialect == "sqlite":
            return int(self._exec(sql, vals).lastrowid)
        return int(self._exec(sql + f" RETURNING {pk}", vals).fetchone()[0])

    def _update(self, table: str, pk: str, key: Any, fields: dict) -> None:
        if not fields:
            return
        sets = ", ".join(f"{c} = ?" for c in fields)
        self._exec(f"UPDATE {table} SET {sets} WHERE {pk} = ?", [*(self._encode(c, v) for c, v in fields.items()), key])

    # -- runs ---------------------------------------------------------------
    def create_run(self, run_id: str, meta: dict) -> None:
        self._exec("DELETE FROM runs WHERE run_id = ?", [run_id])
        self._insert("runs", "run_id", {"run_id": run_id, "meta": meta})

    def update_run(self, run_id: str, meta: dict) -> None:
        self._update("runs", "run_id", run_id, {"meta": meta})

    def runs(self) -> list[dict]:
        return self._query("SELECT * FROM runs ORDER BY created_at DESC")

    # -- attempts -----------------------------------------------------------
    def add_attempt(self, **fields: Any) -> int:
        return self._insert("attempts", "attempt_id", fields)

    def update_attempt(self, attempt_id: int, **fields: Any) -> None:
        self._update("attempts", "attempt_id", attempt_id, fields)

    def attempts(self, **where: Any) -> list[dict]:
        clause, params = self._where(where)
        return self._query(f"SELECT * FROM attempts{clause} ORDER BY attempt_id", params)

    # -- lessons ------------------------------------------------------------
    def add_lesson(self, **fields: Any) -> int:
        return self._insert("lessons", "lesson_id", fields)

    def update_lesson(self, lesson_id: int, **fields: Any) -> None:
        self._update("lessons", "lesson_id", lesson_id, fields)

    def get_lesson(self, lesson_id: int) -> dict:
        rows = self._query("SELECT * FROM lessons WHERE lesson_id = ?", [lesson_id])
        if not rows:
            raise KeyError(lesson_id)
        return rows[0]

    def lessons(self, *, with_vectors: bool = False, statuses: Iterable[str] | None = None, **where: Any) -> list[dict]:
        cols = "*" if with_vectors else (
            "lesson_id, namespace, text, pattern, embedder, status, source_task, source_attempt, created_step, "
            "alpha, beta, alpha0, beta0, uses, replay, history, merged_into, created_at"
        )
        clause, params = self._where(where)
        if statuses is not None:
            statuses = list(statuses)
            if not statuses:
                return []
            marks = ", ".join("?" for _ in statuses)
            clause += (" AND" if clause else " WHERE") + f" status IN ({marks})"
            params += statuses
        return self._query(f"SELECT {cols} FROM lessons{clause} ORDER BY lesson_id", params)

    def candidates(self, namespace: str, statuses: Iterable[str], query: np.ndarray, limit: int = 50) -> list[dict]:
        """Nearest lessons by max(cos(query, text), cos(query, source context))."""
        rows = self.lessons(with_vectors=True, statuses=statuses, namespace=namespace)
        if not rows:
            return []
        q = np.asarray(query, dtype=np.float32)
        E = np.stack([r["embedding"] for r in rows])
        C = np.stack([r["context"] for r in rows])
        cos_text, cos_ctx = E @ q, C @ q
        order = np.argsort(-np.maximum(cos_text, cos_ctx))[:limit]
        out = []
        for i in order:
            r = rows[i]
            r["cos_text"], r["cos_ctx"] = float(cos_text[i]), float(cos_ctx[i])
            out.append(r)
        return out

    # -- retrieval events ---------------------------------------------------
    def add_retrieval(self, **fields: Any) -> int:
        return self._insert("retrieval_events", "event_id", fields)

    def resolve_retrievals(self, attempt_id: int, passed: bool, pattern: str | None) -> list[dict]:
        events = self._query("SELECT * FROM retrieval_events WHERE attempt_id = ?", [attempt_id])
        for e in events:
            lesson = self.get_lesson(e["lesson_id"])
            hit = (not passed) and pattern == lesson["pattern"]
            self._update("retrieval_events", "event_id", e["event_id"], {"passed": passed, "pattern_hit": hit})
            e["passed"], e["pattern_hit"], e["lesson"] = passed, hit, lesson
        return events

    def retrieval_events(self, **where: Any) -> list[dict]:
        clause, params = self._where(where)
        return self._query(f"SELECT * FROM retrieval_events{clause} ORDER BY event_id", params)

    # -- helpers ------------------------------------------------------------
    def _where(self, where: dict) -> tuple[str, list]:
        if not where:
            return "", []
        parts, params = [], []
        for k, v in where.items():
            parts.append(f"{k} = ?")
            params.append(self._encode(k, v))
        return " WHERE " + " AND ".join(parts), params

    def delete_namespace_prefix(self, prefix: str) -> None:
        like = prefix + "%"
        self._exec("DELETE FROM retrieval_events WHERE namespace LIKE ?", [like])
        self._exec("DELETE FROM lessons WHERE namespace LIKE ?", [like])
        self._exec("DELETE FROM attempts WHERE namespace LIKE ?", [like])


class PgStore(Store):
    def candidates(self, namespace: str, statuses: Iterable[str], query: np.ndarray, limit: int = 50) -> list[dict]:
        statuses = list(statuses)
        if not statuses:
            return []
        q = np.asarray(query, dtype=np.float32)
        marks = ", ".join("%s" for _ in statuses)
        sql = f"""
            SELECT *, 1 - (embedding <=> %s) AS cos_text, 1 - (context <=> %s) AS cos_ctx
            FROM lessons WHERE namespace = %s AND status IN ({marks})
            ORDER BY LEAST(embedding <=> %s, context <=> %s) LIMIT %s"""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(sql, (q, q, namespace, *statuses, q, q, limit))
            cols = [d[0] for d in cur.description]
            return [self._decode(dict(zip(cols, r))) for r in cur.fetchall()]


def open_sqlite(path: str | Path) -> Store:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.executescript(SQLITE_DDL)
    conn.execute("PRAGMA journal_mode=WAL")
    return Store(conn, "sqlite")


def open_postgres(dsn: str) -> Store:
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn, connect_timeout=5)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    conn.execute((Path(__file__).with_name("schema.sql")).read_text())
    conn.commit()
    return PgStore(conn, "postgres")


def open_store(database_url: str | None = None, sqlite_path: str | None = None) -> Store:
    from ..config import settings

    url = settings.database_url if database_url is None else database_url
    if url:
        try:
            return open_postgres(url)
        except Exception as e:  # noqa: BLE001
            log.warning("postgres unavailable (%s); falling back to sqlite", e)
    return open_sqlite(sqlite_path or settings.sqlite_path)
