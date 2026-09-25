"""持久化层测试：摘要列持久化与旧版库的可重复迁移回填。"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.db import Database
from app.main import app

# 旧版（引入摘要列之前）的表结构
LEGACY_SCHEMA = """
CREATE TABLE submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence TEXT NOT NULL,
    topology TEXT NOT NULL CHECK (topology IN ('linear', 'circular')),
    created_at TEXT NOT NULL
);
CREATE TABLE enzymes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    pattern TEXT NOT NULL,
    cut_top INTEGER NOT NULL,
    cut_bottom INTEGER NOT NULL
);
CREATE TABLE results (
    submission_id INTEGER PRIMARY KEY REFERENCES submissions(id) ON DELETE CASCADE,
    result_json TEXT NOT NULL
);
CREATE INDEX idx_enzymes_submission ON enzymes(submission_id);
"""


def _raw_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_legacy_row(conn, sequence, topology, result, enzyme_name="EcoRI"):
    cur = conn.execute(
        "INSERT INTO submissions (sequence, topology, created_at) VALUES (?, ?, ?)",
        (sequence, topology, "2026-01-01T00:00:00+00:00"),
    )
    submission_id = cur.lastrowid
    conn.execute(
        "INSERT INTO enzymes (submission_id, name, pattern, cut_top, cut_bottom)"
        " VALUES (?, ?, 'GAATTC', 1, 5)",
        (submission_id, enzyme_name),
    )
    conn.execute(
        "INSERT INTO results (submission_id, result_json) VALUES (?, ?)",
        (submission_id, json.dumps(result)),
    )
    return submission_id


@pytest.fixture()
def legacy_db(tmp_path):
    """构造一个含两条历史记录的旧版库（results 无摘要列）。"""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    _insert_legacy_row(
        conn,
        "AAAAGAATTCTTTT",
        "linear",
        {"cuts": [{"id": "cut-1"}], "fragments": [{"index": 0}, {"index": 1}]},
    )
    _insert_legacy_row(
        conn,
        "GCGCGCGC",
        "circular",
        {"cuts": [], "fragments": [{"index": 0}]},
    )
    conn.commit()
    conn.close()
    return path


def test_legacy_db_lacks_summary_columns(legacy_db):
    columns = [r[1] for r in sqlite3.connect(legacy_db).execute("PRAGMA table_info(results)")]
    assert columns == ["submission_id", "result_json"]


def test_migration_adds_columns_and_backfills(legacy_db):
    Database(legacy_db)  # 打开旧库即触发迁移
    conn = _raw_connect(legacy_db)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(results)")]
    assert columns == ["submission_id", "result_json", "cut_count", "fragment_count"]
    rows = conn.execute(
        "SELECT submission_id, cut_count, fragment_count FROM results"
        " ORDER BY submission_id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [(1, 1, 2), (2, 0, 1)]
    conn.close()


def test_migration_is_repeatable(legacy_db):
    Database(legacy_db)
    Database(legacy_db)  # 重复执行不报错、不改动已回填值
    conn = _raw_connect(legacy_db)
    rows = conn.execute(
        "SELECT submission_id, cut_count, fragment_count FROM results"
        " ORDER BY submission_id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [(1, 1, 2), (2, 0, 1)]
    conn.close()


def test_migration_backfills_null_rows_even_when_column_exists(tmp_path):
    """列已存在但某行摘要为 NULL（半途迁移遗留）时，再次启动仍能补齐。"""
    path = tmp_path / "partial.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    _insert_legacy_row(
        conn, "AAAAGAATTCTTTT", "linear",
        {"cuts": [], "fragments": [{"index": 0}]},
    )
    conn.commit()
    conn.close()
    # 只补列、不回填，模拟半途中断后的库
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE results ADD COLUMN cut_count INTEGER")
    conn.execute("ALTER TABLE results ADD COLUMN fragment_count INTEGER")
    assert conn.execute("SELECT cut_count FROM results").fetchone()[0] is None
    conn.commit()
    conn.close()

    Database(path)
    conn = sqlite3.connect(path)
    assert conn.execute(
        "SELECT cut_count, fragment_count FROM results"
    ).fetchone() == (0, 1)
    conn.close()


def test_list_on_migrated_db_via_api(legacy_db, monkeypatch):
    """应用在旧库上启动后，列表字段与排序保持既有行为。"""
    monkeypatch.setenv("DATABASE_PATH", str(legacy_db))
    with TestClient(app) as client:
        resp = client.get("/digests")
        assert resp.status_code == 200
        items = resp.json()
        assert [item["id"] for item in items] == [1, 2]
        assert items[0] == {
            "id": 1,
            "topology": "linear",
            "sequence_length": 14,
            "enzyme_count": 1,
            "cut_count": 1,
            "fragment_count": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        assert items[1]["cut_count"] == 0
        assert items[1]["fragment_count"] == 1

        # 完整结果仍可正常读取
        full = client.get("/digests/1")
        assert full.status_code == 200
        assert len(full.json()["cuts"]) == 1
        assert len(full.json()["fragments"]) == 2


def test_migration_tolerates_unparseable_result_json(tmp_path):
    """迁移回填时个别损坏的 result_json 不会让应用启动失败。"""
    path = tmp_path / "corrupt.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    _insert_legacy_row(
        conn,
        "AAAAGAATTCTTTT",
        "linear",
        {"cuts": [{"id": "c"}], "fragments": [{"index": 0}]},
    )
    conn.commit()
    conn.close()
    # 模拟列已补但回填前 result_json 被损坏：迁移不应抛错
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE results ADD COLUMN cut_count INTEGER")
    conn.execute("ALTER TABLE results ADD COLUMN fragment_count INTEGER")
    conn.execute("UPDATE results SET result_json = '<<<broken json>>>'")
    conn.commit()
    conn.close()

    Database(path)  # 不抛异常即为通过


def test_migration_on_empty_legacy_db(tmp_path):
    """空旧库迁移也应成功并补齐列。"""
    path = tmp_path / "empty.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    conn.commit()
    conn.close()

    Database(path)
    conn = sqlite3.connect(path)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(results)")]
    assert columns == ["submission_id", "result_json", "cut_count", "fragment_count"]
    conn.close()


def test_list_does_not_parse_result_json(legacy_db):
    """列表只读摘要列：result_json 损坏后列表仍返回正确计数。"""
    db = Database(legacy_db)
    conn = sqlite3.connect(legacy_db)
    conn.execute("UPDATE results SET result_json = '<<<broken json>>>'")
    conn.commit()
    conn.close()

    items = db.list_submissions()
    assert [(i["cut_count"], i["fragment_count"]) for i in items] == [(1, 2), (0, 1)]


def test_new_submission_persists_summary_in_same_transaction(tmp_path):
    db_path = tmp_path / "new.db"
    db = Database(db_path)
    from app.schemas import EnzymeIn

    enzyme = EnzymeIn(name="EcoRI", pattern="GAATTC", cut_top=1, cut_bottom=5)
    sid, _ = db.save_submission(
        "AAAAGAATTCTTTT",
        "linear",
        [enzyme],
        {"cuts": [{"id": "c1"}], "fragments": [{"index": 0}, {"index": 1}]},
    )
    conn = _raw_connect(db_path)
    row = conn.execute(
        "SELECT result_json, cut_count, fragment_count FROM results WHERE submission_id = ?",
        (sid,),
    ).fetchone()
    assert json.loads(row["result_json"])["cuts"] == [{"id": "c1"}]
    assert row["cut_count"] == 1
    assert row["fragment_count"] == 2

    # 全新建库的摘要列为 NOT NULL
    info = conn.execute("PRAGMA table_info(results)").fetchall()
    not_null = {r["name"]: r["notnull"] for r in info}
    assert not_null["cut_count"] == 1
    assert not_null["fragment_count"] == 1
    conn.close()


def test_new_submission_listed_after_migration(legacy_db, monkeypatch):
    """迁移后的旧库上继续提交，新记录摘要同事务写入并出现在列表末尾。"""
    monkeypatch.setenv("DATABASE_PATH", str(legacy_db))
    with TestClient(app) as client:
        payload = {
            "sequence": "AAAAGAATTCTTTT",
            "topology": "linear",
            "enzymes": [
                {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}
            ],
        }
        created = client.post("/digests", json=payload)
        assert created.status_code == 201
        items = client.get("/digests").json()
        assert [item["id"] for item in items] == [1, 2, 3]
        assert items[2]["cut_count"] == 1
        assert items[2]["fragment_count"] == 2
