"""线性端点切口末端规则测试：端点切口写入首/末片段末端、修复前记录的
读取校正，以及连接兼容性 / 拼接链 / 甲基化分析对正确末端的使用。"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.db import Database
from app.digest import run_digest
from app.main import app
from app.schemas import EnzymeIn


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


# 端点切口酶：识别 AAG，cut_top=0、cut_bottom=1 -> 切口 (0,1)，5' 突出端 A
END5 = {"name": "End5", "pattern": "AAG", "cut_top": 0, "cut_bottom": 1}
# 3' 突出端酶：识别 ATG，cut_top=2、cut_bottom=1 -> 切口 (4,3)，突出端 T
# （反向互补(T)=A，与 5' A 端兼容）
THREE_PRIME_ENZYME = {"name": "F3", "pattern": "ATG", "cut_top": 2, "cut_bottom": 1}

END5_SEQ = "AAGCCCC"  # 长度 7，唯一切口 (0,1) 位于分子端点
F3_SEQ = "GGATGGG"  # 长度 7，唯一切口 (4,3)，f0 右端 / f1 左端为 3prime(T)


def _create(client, sequence, enzymes, topology="linear"):
    resp = client.post(
        "/digests",
        json={"sequence": sequence, "topology": topology, "enzymes": enzymes},
    )
    assert resp.status_code == 201
    return resp.json()


def _insert_prefix_record(db_path, sequence, enzyme):
    """把一条修复前风格的线性记录直接落库：端点切口已保存，
    但片段端点末端按修复前行为丢失为 none。返回记录 id。"""
    db = Database(str(db_path))
    enz = EnzymeIn(**enzyme)
    result = run_digest(sequence, "linear", [enz])
    for frag in result["fragments"]:
        frag["left_end"] = {"type": "none", "overhang": ""}
        frag["right_end"] = {"type": "none", "overhang": ""}
    submission_id, _ = db.save_submission(sequence, "linear", [enz], result)
    return submission_id


def _stored_result(db_path, submission_id):
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT result_json FROM results WHERE submission_id = ?",
        (submission_id,),
    ).fetchone()
    conn.close()
    return json.loads(row[0])


# ---------- 提交与查询 ----------


def test_endpoint_cut_end_written_on_create_and_get(client):
    body = _create(client, END5_SEQ, [END5])
    assert [(c["top_cut"], c["bottom_cut"]) for c in body["cuts"]] == [(0, 1)]
    assert body["cuts"][0]["end_type"] == "5prime"
    assert body["cuts"][0]["overhang"] == "A"
    assert len(body["fragments"]) == 1  # 端点切口不增加片段数
    frag = body["fragments"][0]
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 查询结果与提交时完全一致


def test_no_endpoint_cut_keeps_none_ends(client):
    body = _create(
        client,
        "AAAAGAATTCTTTT",
        [{"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}],
    )
    assert body["fragments"][0]["left_end"] == {"type": "none", "overhang": ""}
    assert body["fragments"][-1]["right_end"] == {"type": "none", "overhang": ""}


# ---------- 修复前记录的读取校正 ----------


def test_read_repairs_prefix_record_without_rewriting_db(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    sid = _insert_prefix_record(db_path, END5_SEQ, END5)
    # 落库内容保持修复前样式：端点切口在 cuts 中，片段末端为 none
    stored_before = _stored_result(db_path, sid)
    assert [(c["top_cut"], c["bottom_cut"]) for c in stored_before["cuts"]] == [(0, 1)]
    assert stored_before["fragments"][0]["left_end"] == {"type": "none", "overhang": ""}

    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    with TestClient(app) as client:
        got = client.get(f"/digests/{sid}")
        assert got.status_code == 200
        frag = got.json()["fragments"][0]
        # 读取时依据已存切口校正片段末端
        assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
        assert frag["right_end"] == {"type": "none", "overhang": ""}
        # 切口与位点原样返回，不重算
        assert got.json()["cuts"] == stored_before["cuts"]

    # 数据库不被改写：再次读取原始结果仍为修复前内容
    assert _stored_result(db_path, sid) == stored_before


def test_repaired_record_ends_used_by_ligation_and_assembly(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    sid = _insert_prefix_record(db_path, END5_SEQ, END5)
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    with TestClient(app) as client:
        rec3 = _create(client, F3_SEQ, [THREE_PRIME_ENZYME])

        # 连接兼容性：修复前该 5' 端为 none 会漏报，校正后能与 3' T 端配对
        data = client.post(
            "/ligation-compatibility", json={"digest_ids": [sid, rec3["id"]]}
        ).json()
        assert data["pair_count"] == 2
        assert all(p["compatibility"] == "sticky" for p in data["pairs"])
        assert all(p["ligated_overhang"] == "A" for p in data["pairs"])
        assert data["pairs"][0]["first"] == {
            "digest_id": sid,
            "fragment_index": 0,
            "end": "left",
            "end_type": "5prime",
            "overhang": "A",
        }
        assert data["pairs"][0]["second"] == {
            "digest_id": rec3["id"],
            "fragment_index": 0,
            "end": "right",
            "end_type": "3prime",
            "overhang": "T",
        }

        # 拼接链：校正后的端点末端使两条记录连成一条链
        plan = client.post(
            "/ligation-assembly",
            json={"digest_ids": [sid, rec3["id"]], "max_chain_length": 8},
        ).json()
        assert plan["node_count"] == 2
        assert plan["connection_count"] == 1
        conn = plan["connections"][0]
        assert conn["compatibility"] == "sticky"
        assert conn["ligated_overhang"] == "A"
        assert plan["used_digest_ids"] == [sid, rec3["id"]]
        assert plan["stop_reason"] == "all_records_used"


def test_fresh_records_endpoint_ends_ligate(client):
    # 新提交的记录同样经连接分析使用端点末端（无需读取校正路径）
    rec5 = _create(client, END5_SEQ, [END5])
    rec3 = _create(client, F3_SEQ, [THREE_PRIME_ENZYME])
    data = client.post(
        "/ligation-compatibility", json={"digest_ids": [rec5["id"], rec3["id"]]}
    ).json()
    assert data["pair_count"] == 2
    assert {p["first"]["end_type"] for p in data["pairs"]} == {"5prime"}


# ---------- 甲基化与端点切口 ----------


def test_methylation_blocking_endpoint_cut_restores_natural_end(client):
    body = _create(client, END5_SEQ, [END5])
    # 位点起于 0，敏感偏移 0 -> 物理坐标 0 被甲基化即阻断端点切口 (0,1)
    resp = client.post(
        f"/digests/{body['id']}/methylation-analysis",
        json={
            "methylated_positions": [0],
            "enzyme_rules": [{"enzyme": "End5", "sensitive_offsets": [0]}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["blocked_sites"]) == 1
    assert data["retained_cuts"] == []
    assert [(c["top_cut"], c["bottom_cut"]) for c in data["removed_cuts"]] == [(0, 1)]
    # 端点切口被阻断后恢复天然末端
    assert len(data["fragments"]) == 1
    frag = data["fragments"][0]
    assert frag["left_end"] == {"type": "none", "overhang": ""}
    assert frag["right_end"] == {"type": "none", "overhang": ""}


def test_methylation_retaining_endpoint_cut_keeps_overhang_end(client):
    body = _create(client, END5_SEQ, [END5])
    # 甲基化落在位点之外（坐标 3）：端点切口保留，片段左端保持 5' 突出端
    resp = client.post(
        f"/digests/{body['id']}/methylation-analysis",
        json={
            "methylated_positions": [3],
            "enzyme_rules": [{"enzyme": "End5", "sensitive_offsets": [0]}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["blocked_sites"] == []
    assert len(data["retained_cuts"]) == 1
    frag = data["fragments"][0]
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}
