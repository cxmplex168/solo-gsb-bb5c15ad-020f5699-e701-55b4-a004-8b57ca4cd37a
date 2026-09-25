"""SQLite 持久化层。

数据库文件路径由环境变量 DATABASE_PATH 指定（默认 data/digest.db），
应用启动时自动创建目录与表结构。所有写入在单个事务中完成，
任一失败即整体回滚，不会产生半截数据。

results 表除完整结果 result_json 外，另持久化摘要列 cut_count /
fragment_count：GET /digests 列表只读摘要列，不再逐条加载、解析
完整结果。启动时对旧版库执行**可重复迁移**（缺列则 ALTER 补列、
NULL 行由 result_json 回填），重复执行不会重复改动或回填。

读取完整记录时（get_submission），对修复前保存的**线性**记录在内存中
依据已存切口校正分子端点末端（digest.repair_linear_fragment_ends）：
端点切口（top_cut 为 0 或 L）的末端类型与突出端曾未被写入首片段左端 /
末片段右端。校正只作用于返回值，不重算位点、不改写数据库。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from .digest import repair_linear_fragment_ends

# 全新建库时的表结构：摘要列直接随表创建。
SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence    TEXT NOT NULL,
    topology    TEXT NOT NULL CHECK (topology IN ('linear', 'circular')),
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enzymes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    pattern       TEXT NOT NULL,
    cut_top       INTEGER NOT NULL,
    cut_bottom    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    submission_id  INTEGER PRIMARY KEY REFERENCES submissions(id) ON DELETE CASCADE,
    result_json    TEXT NOT NULL,
    cut_count      INTEGER NOT NULL,
    fragment_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_enzymes_submission ON enzymes(submission_id);
"""

# 旧版库的可重复迁移：逐列补缺、按 NULL 回填，全程在一个事务内完成。
# json_valid 守卫保证个别损坏行（应用自身不会写出）不会让启动迁移整体失败：
# 能解析的历史行照常回填，损坏行保持 NULL 待人工修复。
MIGRATIONS = (
    (
        "cut_count",
        "UPDATE results SET cut_count ="
        " (SELECT json_array_length(result_json, '$.cuts'))"
        " WHERE cut_count IS NULL AND json_valid(result_json)",
    ),
    (
        "fragment_count",
        "UPDATE results SET fragment_count ="
        " (SELECT json_array_length(result_json, '$.fragments'))"
        " WHERE fragment_count IS NULL AND json_valid(result_json)",
    ),
)


class Database:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """把旧版库补齐到当前表结构并回填历史摘要。

        幂等：列已存在则跳过 ALTER；摘要已回填的行不为 NULL，不会被
        重复更新。补列与回填在同一事务内提交，中途失败整体回滚。
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(results)")
            }
            for column, backfill_sql in MIGRATIONS:
                if column not in existing:
                    # 先以可空列补入旧表
                    conn.execute(
                        f"ALTER TABLE results ADD COLUMN {column} INTEGER"
                    )
                # 每次启动都执行：WHERE ... IS NULL 使其幂等——
                # 新补列的历史行全部为 NULL 会被回填；已回填的行不再命中。
                conn.execute(backfill_sql)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_submission(self, sequence: str, topology: str, enzymes, result: dict):
        """在一次事务中保存提交、酶列表与计算结果，返回 (id, created_at)。

        摘要计数与完整结果在同一事务、同一条 results 记录写入，
        任一失败即整体回滚，二者不会出现不一致。
        """
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:  # 异常时自动回滚
            cur = conn.execute(
                "INSERT INTO submissions (sequence, topology, created_at) VALUES (?, ?, ?)",
                (sequence, topology, created_at),
            )
            submission_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO enzymes (submission_id, name, pattern, cut_top, cut_bottom)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (submission_id, e.name, e.pattern, e.cut_top, e.cut_bottom)
                    for e in enzymes
                ],
            )
            conn.execute(
                "INSERT INTO results (submission_id, result_json,"
                " cut_count, fragment_count) VALUES (?, ?, ?, ?)",
                (
                    submission_id,
                    json.dumps(result, ensure_ascii=False),
                    len(result["cuts"]),
                    len(result["fragments"]),
                ),
            )
        return submission_id, created_at

    def get_submission(self, submission_id: int):
        """按 id 查询完整记录，不存在时返回 None。"""
        with self._connect() as conn:
            sub = conn.execute(
                "SELECT id, sequence, topology, created_at FROM submissions WHERE id = ?",
                (submission_id,),
            ).fetchone()
            if sub is None:
                return None
            enzymes = conn.execute(
                "SELECT name, pattern, cut_top, cut_bottom FROM enzymes"
                " WHERE submission_id = ? ORDER BY id",
                (submission_id,),
            ).fetchall()
            result = conn.execute(
                "SELECT result_json FROM results WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        parsed = json.loads(result["result_json"])
        if sub["topology"] == "linear":
            # 修复前保存的线性记录可能漏写端点切口的末端信息：读取时
            # 依据已存切口在内存中校正片段末端，数据库内容保持不变
            repair_linear_fragment_ends(parsed, len(sub["sequence"]))
        return {
            "id": sub["id"],
            "sequence": sub["sequence"],
            "topology": sub["topology"],
            "created_at": sub["created_at"],
            "enzymes": [dict(e) for e in enzymes],
            "result": parsed,
        }

    def list_submissions(self) -> list[dict]:
        """列出全部提交的摘要信息，按 id 升序（结果顺序确定）。

        只查询摘要列与必要的提交元数据：序列长度由 SQL LENGTH 计算，
        切口/片段数直接取持久化的摘要列，不读取也不解析 result_json，
        列表成本不再随完整结果体积增长。
        """
        with self._connect() as conn:
            subs = conn.execute(
                "SELECT s.id, s.topology, LENGTH(s.sequence) AS sequence_length,"
                "       s.created_at,"
                "       (SELECT COUNT(*) FROM enzymes e WHERE e.submission_id = s.id) AS enzyme_count,"
                "       r.cut_count AS cut_count,"
                "       r.fragment_count AS fragment_count"
                " FROM submissions s JOIN results r ON r.submission_id = s.id"
                " ORDER BY s.id"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "topology": row["topology"],
                "sequence_length": row["sequence_length"],
                "enzyme_count": row["enzyme_count"],
                "cut_count": row["cut_count"],
                "fragment_count": row["fragment_count"],
                "created_at": row["created_at"],
            }
            for row in subs
        ]
