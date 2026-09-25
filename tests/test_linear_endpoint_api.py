"""线性端点切口末端规则与旧记录读取校正的接口测试。"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.db import Database
from app.main import app
from app.schemas import EnzymeIn

# 端点切口酶：识别 AAG，cut_top=0、cut_bottom=1 -> 切口 (0,1)，5' 突出端 A
ENDPOINT_ENZYME = {"name": "End5", "pattern": "AAG", "cut_top": 0, "cut_bottom": 1}
# 3' 突出端酶：识别 GT，cut_top=2、cut_bottom=1 -> 切口 (4,3)，3' 突出端 T
# （反向互补(T)=A，可与 5' 突出端 A 配对）
THREE_PRIME_ENZYME = {"name": "End3", "pattern": "GT", "cut_top": 2, "cut_bottom": 1}

# 修复前保存的线性记录：切口 (0,1) 已存，但片段左端被写为 none（端点切口
# 的末端信息丢失），用于验证读取时的内存校正
PREFIX_RESULT = {
    "cuts": [
        {
            "top_cut": 0,
            "bottom_cut": 1,
            "end_type": "5prime",
            "overhang": "A",
            "enzymes": ["End5"],
            "sites": [
                {"enzyme": "End5", "start": 0, "strand": "+", "matched": "AAG"}
            ],
        }
    ],
    "fragments": [
        {
            "index": 0,
            "start": 0,
            "end": 7,
            "length": 7,
            "sequence": "AAGCCCC",
            "wraps_origin": False,
            "left_end": {"type": "none", "overhang": ""},
            "right_end": {"type": "none", "overhang": ""},
        }
    ],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def prefix_client(tmp_path, monkeypatch):
    """含一条修复前线性记录的库：切口 (0,1) 存在但片段左端为 none。"""
    path = tmp_path / "prefix.db"
    db = Database(str(path))
    enzyme = EnzymeIn(name="End5", pattern="AAG", cut_top=0, cut_bottom=1)
    sid, _ = db.save_submission("AAGCCCC", "linear", [enzyme], PREFIX_RESULT)
    monkeypatch.setenv("DATABASE_PATH", str(path))
    with TestClient(app) as c:
        yield c, sid, path


def _create(client, sequence, enzymes, topology="linear"):
    resp = client.post(
        "/digests",
        json={"sequence": sequence, "topology": topology, "enzymes": enzymes},
    )
    assert resp.status_code == 201
    return resp.json()


def _create_three_prime(client):
    """3prime 记录：序列 CCGTCC，单切口 (4,3)，f0 右端 / f1 左端为 3prime(T)。"""
    return _create(client, "CCGTCC", [THREE_PRIME_ENZYME])


# ---------- 端点切口末端规则（新提交） ----------


def test_endpoint_cut_written_to_fragment_end_via_api(client):
    resp = client.post(
        "/digests",
        json={
            "sequence": "AAGCCCC",
            "topology": "linear",
            "enzymes": [ENDPOINT_ENZYME],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    # 切口照常报告（含末端类型与突出端）
    assert len(body["cuts"]) == 1
    assert body["cuts"][0]["top_cut"] == 0
    assert body["cuts"][0]["bottom_cut"] == 1
    assert body["cuts"][0]["end_type"] == "5prime"
    assert body["cuts"][0]["overhang"] == "A"
    # 端点切口不增加片段，但首片段左端携带切口末端
    assert len(body["fragments"]) == 1
    frag = body["fragments"][0]
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 查询结果与提交时完全一致


def test_endpoint_cut_pairs_in_ligation_via_api(client):
    rec5 = _create(client, "AAGCCCC", [ENDPOINT_ENZYME])  # f0 左端 5prime(A)
    rec3 = _create_three_prime(client)
    resp = client.post(
        "/ligation-compatibility",
        json={"digest_ids": [rec5["id"], rec3["id"]]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_count"] == 2
    a, b = rec5["id"], rec3["id"]
    assert data["pairs"] == [
        {
            "first": {
                "digest_id": a, "fragment_index": 0, "end": "left",
                "end_type": "5prime", "overhang": "A",
            },
            "second": {
                "digest_id": b, "fragment_index": 0, "end": "right",
                "end_type": "3prime", "overhang": "T",
            },
            "compatibility": "sticky",
            "ligated_overhang": "A",
        },
        {
            "first": {
                "digest_id": a, "fragment_index": 0, "end": "left",
                "end_type": "5prime", "overhang": "A",
            },
            "second": {
                "digest_id": b, "fragment_index": 1, "end": "left",
                "end_type": "3prime", "overhang": "T",
            },
            "compatibility": "sticky",
            "ligated_overhang": "A",
        },
    ]


# ---------- 修复前记录的读取校正（内存修复，不改写数据库） ----------


def test_prefix_record_repaired_on_read(prefix_client):
    client, sid, path = prefix_client
    resp = client.get(f"/digests/{sid}")
    assert resp.status_code == 200
    frag = resp.json()["fragments"][0]
    # 已存切口 (0,1) 的末端校正到首片段左端
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}

    # 数据库内容不被改写：原始 result_json 仍为修复前的 none 末端
    raw = sqlite3.connect(path).execute(
        "SELECT result_json FROM results WHERE submission_id = ?", (sid,)
    ).fetchone()[0]
    stored = json.loads(raw)
    assert stored["fragments"][0]["left_end"] == {"type": "none", "overhang": ""}


def test_prefix_record_ligation_uses_repaired_end(prefix_client):
    client, sid, _ = prefix_client
    rec3 = _create_three_prime(client)
    resp = client.post(
        "/ligation-compatibility",
        json={"digest_ids": [sid, rec3["id"]]},
    )
    assert resp.status_code == 200
    data = resp.json()
    # 修复前该记录两端均为 none、不产生任何配对；校正后 5prime(A) 参与配对
    assert data["pair_count"] == 2
    firsts = {(p["first"]["digest_id"], p["first"]["end"]) for p in data["pairs"]}
    assert firsts == {(sid, "left")}
    assert all(p["first"]["end_type"] == "5prime" for p in data["pairs"])
    assert all(p["compatibility"] == "sticky" for p in data["pairs"])


def test_prefix_record_assembly_uses_repaired_end(prefix_client):
    client, sid, _ = prefix_client
    rec3 = _create_three_prime(client)
    resp = client.post(
        "/ligation-assembly",
        json={
            "digest_ids": [sid, rec3["id"]],
            "include_blunt": True,
            "max_chain_length": 8,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # 校正后的 5prime(A) 端使两条记录拼接成链；修复前只能报 no_compatible_ends
    assert data["node_count"] == 2
    assert data["connection_count"] == 1
    assert data["used_digest_ids"] == [sid, rec3["id"]]
    assert data["stop_reason"] == "all_records_used"
    connection = data["connections"][0]
    assert connection["from_end"] == {
        "digest_id": sid, "fragment_index": 0, "end": "left",
        "end_type": "5prime", "overhang": "A",
    }
    assert connection["to_end"] == {
        "digest_id": rec3["id"], "fragment_index": 0, "end": "right",
        "end_type": "3prime", "overhang": "T",
    }
    assert connection["compatibility"] == "sticky"
    assert connection["ligated_overhang"] == "A"


def test_circular_record_not_repaired(tmp_path, monkeypatch):
    """读取校正只作用于线性记录：环形记录的内容按原样返回。"""
    path = tmp_path / "circular.db"
    db = Database(str(path))
    enzyme = EnzymeIn(name="End5", pattern="AAG", cut_top=0, cut_bottom=1)
    circular_result = {
        "cuts": [
            {
                "top_cut": 0,
                "bottom_cut": 1,
                "end_type": "5prime",
                "overhang": "A",
                "enzymes": ["End5"],
                "sites": [
                    {"enzyme": "End5", "start": 0, "strand": "+", "matched": "AAG"}
                ],
            }
        ],
        "fragments": [
            {
                "index": 0,
                "start": 0,
                "end": 0,
                "length": 7,
                "sequence": "AAGCCCC",
                "wraps_origin": False,
                "left_end": {"type": "none", "overhang": ""},
                "right_end": {"type": "none", "overhang": ""},
            }
        ],
    }
    sid, _ = db.save_submission("AAGCCCC", "circular", [enzyme], circular_result)
    monkeypatch.setenv("DATABASE_PATH", str(path))
    with TestClient(app) as client:
        frag = client.get(f"/digests/{sid}").json()["fragments"][0]
        # 环形记录不存在"端点切口"概念，存储内容原样返回
        assert frag["left_end"] == {"type": "none", "overhang": ""}
        assert frag["right_end"] == {"type": "none", "overhang": ""}


# ---------- 甲基化阻断端点切口后恢复天然末端 ----------


def test_methylation_blocking_endpoint_cut_restores_natural_end(client):
    rec = _create(client, "AAGCCCC", [ENDPOINT_ENZYME])
    resp = client.post(
        f"/digests/{rec['id']}/methylation-analysis",
        json={
            "methylated_positions": [0],
            "enzyme_rules": [{"enzyme": "End5", "sensitive_offsets": [0]}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["retained_cuts"] == []
    assert len(data["removed_cuts"]) == 1
    assert (data["removed_cuts"][0]["top_cut"], data["removed_cuts"][0]["bottom_cut"]) == (0, 1)
    # 端点切口被阻断移除后，分子末端恢复为天然末端 none
    assert len(data["fragments"]) == 1
    frag = data["fragments"][0]
    assert frag["left_end"] == {"type": "none", "overhang": ""}
    assert frag["right_end"] == {"type": "none", "overhang": ""}


def test_methylation_retaining_endpoint_cut_keeps_sticky_end(client):
    rec = _create(client, "AAGCCCC", [ENDPOINT_ENZYME])
    # 甲基化坐标 3 不在识别位点（0..2）的敏感偏移上，端点切口保留
    resp = client.post(
        f"/digests/{rec['id']}/methylation-analysis",
        json={
            "methylated_positions": [3],
            "enzyme_rules": [{"enzyme": "End5", "sensitive_offsets": [0]}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["removed_cuts"] == []
    assert len(data["retained_cuts"]) == 1
    frag = data["fragments"][0]
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}
