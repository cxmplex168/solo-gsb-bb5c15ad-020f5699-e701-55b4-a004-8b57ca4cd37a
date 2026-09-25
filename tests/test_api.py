"""API 层测试：提交、查询、校验与原子拒绝。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def eco_request(**overrides):
    payload = {
        "sequence": "AAAAGAATTCTTTT",
        "topology": "linear",
        "enzymes": [{"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}],
    }
    payload.update(overrides)
    return payload


def test_create_and_get_digest(client):
    resp = client.post("/digests", json=eco_request())
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == 1
    assert body["topology"] == "linear"
    assert body["sequence_length"] == 14
    assert len(body["cuts"]) == 1
    assert body["cuts"][0]["end_type"] == "5prime"
    assert body["cuts"][0]["overhang"] == "AATT"
    assert [f["length"] for f in body["fragments"]] == [5, 9]

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 查询结果与提交时完全一致


def test_list_digests(client):
    client.post("/digests", json=eco_request())
    client.post("/digests", json=eco_request(sequence="GCGCGCGC", topology="circular"))
    resp = client.get("/digests")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert items[0]["id"] == 1 and items[0]["cut_count"] == 1
    assert items[1]["id"] == 2 and items[1]["cut_count"] == 0
    assert items[1]["fragment_count"] == 1  # 无切口保留整条分子


def test_get_missing_digest_returns_404(client):
    assert client.get("/digests/999").status_code == 404


def test_illegal_offset_rejects_entire_save(client):
    bad = eco_request(
        enzymes=[
            {"name": "Good", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5},
            {"name": "Bad", "pattern": "GATATC", "cut_top": 7, "cut_bottom": 3},
        ]
    )
    resp = client.post("/digests", json=bad)
    assert resp.status_code == 422
    assert client.get("/digests").json() == []  # 整次保存被拒绝，无任何落库


def test_negative_offset_rejected(client):
    bad = eco_request(
        enzymes=[{"name": "Neg", "pattern": "GAATTC", "cut_top": -1, "cut_bottom": 5}]
    )
    assert client.post("/digests", json=bad).status_code == 422
    assert client.get("/digests").json() == []


def test_invalid_sequence_base_rejected(client):
    resp = client.post("/digests", json=eco_request(sequence="ACGTXACGT"))
    assert resp.status_code == 422
    assert client.get("/digests").json() == []


def test_invalid_pattern_base_rejected(client):
    bad = eco_request(
        enzymes=[{"name": "Bad", "pattern": "GAUZ", "cut_top": 1, "cut_bottom": 1}]
    )
    assert client.post("/digests", json=bad).status_code == 422


def test_empty_enzyme_list_rejected(client):
    assert client.post("/digests", json=eco_request(enzymes=[])).status_code == 422


def test_circular_digest_via_api(client):
    payload = eco_request(sequence="ATTCGGGGGA", topology="circular")
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["cuts"][0]["sites"][0]["start"] == 8  # 跨原点命中
    assert len(body["fragments"]) == 1
    assert body["fragments"][0]["length"] == 10


def test_circular_cut_conflict_rejects_entire_save(client):
    # 环形 AAAA 上两种酶的切口取模后落在同一坐标 (0,3)，但末端不一致
    # （5prime/AAA 与 3prime/A）：整个请求以 422 拒绝，且不落库
    payload = eco_request(
        sequence="AAAA",
        topology="circular",
        enzymes=[
            {"name": "E1", "pattern": "AAAA", "cut_top": 0, "cut_bottom": 3},
            {"name": "E2", "pattern": "AAAA", "cut_top": 1, "cut_bottom": 0},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 422
    assert "(0, 3)" in resp.json()["detail"]
    assert client.get("/digests").json() == []  # 整次保存被拒绝，无任何落库


def test_circular_consistent_same_coordinate_merge_via_api(client):
    # 取模后同坐标且末端一致（同为 5prime/AAA）：合法合并，照常保存
    payload = eco_request(
        sequence="AAAA",
        topology="circular",
        enzymes=[
            {"name": "E1", "pattern": "AAAA", "cut_top": 0, "cut_bottom": 3},
            {"name": "E2", "pattern": "AAAA", "cut_top": 1, "cut_bottom": 4},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["cuts"]) == 4
    cut = next(
        c for c in body["cuts"] if (c["top_cut"], c["bottom_cut"]) == (0, 3)
    )
    assert cut["enzymes"] == ["E1", "E2"]
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "AAA"
    assert [f["length"] for f in body["fragments"]] == [1, 1, 1, 1]


def test_circular_shared_top_cut_counts_consistent(client):
    # 两个物理切口共用同一正链坐标：保存两个 cuts，片段按唯一边界生成
    payload = eco_request(
        sequence="AACAA",
        topology="circular",
        enzymes=[
            {"name": "E1", "pattern": "AC", "cut_top": 0, "cut_bottom": 0},
            {"name": "E2", "pattern": "AC", "cut_top": 0, "cut_bottom": 1},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["cuts"]) == 2
    assert len(body["fragments"]) == 1  # 唯一边界开环为一条全长片段
    frag = body["fragments"][0]
    assert frag["length"] == 5
    assert frag["sequence"] == "ACAAA"
    assert frag["left_end"] == {"type": "blunt", "overhang": ""}

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 明细与提交时完全一致

    items = client.get("/digests").json()
    assert len(items) == 1
    # 列表摘要计数与明细一致
    assert items[0]["cut_count"] == len(body["cuts"]) == 2
    assert items[0]["fragment_count"] == len(body["fragments"]) == 1


def test_palindrome_asymmetric_offsets_recorded_via_api(client):
    # 回文模式 ATAT 偏移 (1,1) 不满足镜像对称：正/反结合落在不同物理
    # 坐标，新提交必须同时记录 (2,2) 与 (4,4) 两个切口及三条片段
    payload = eco_request(
        sequence="AATATG",
        enzymes=[{"name": "Pal", "pattern": "ATAT", "cut_top": 1, "cut_bottom": 1}],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert [(c["top_cut"], c["bottom_cut"]) for c in body["cuts"]] == [(2, 2), (4, 4)]
    assert [f["length"] for f in body["fragments"]] == [2, 2, 2]
    assert [s["strand"] for c in body["cuts"] for s in c["sites"]] == ["+", "-"]

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 落库结果与提交响应完全一致

    items = client.get("/digests").json()
    assert len(items) == 1
    # 列表摘要计数与修正后的明细一致
    assert items[0]["cut_count"] == 2
    assert items[0]["fragment_count"] == 3


def test_palindrome_mirror_symmetric_offsets_single_cut_via_api(client):
    # 回文模式偏移镜像对称（cut_top + cut_bottom == len(pattern)）时
    # 两个结合方向产生同一物理切口：只记录一次，不重复计数
    payload = eco_request(
        sequence="AATATG",
        enzymes=[{"name": "PalSym", "pattern": "ATAT", "cut_top": 1, "cut_bottom": 3}],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["cuts"]) == 1
    cut = body["cuts"][0]
    assert (cut["top_cut"], cut["bottom_cut"]) == (2, 4)
    assert len(cut["sites"]) == 1
    assert cut["sites"][0]["strand"] == "+"
    assert [f["length"] for f in body["fragments"]] == [2, 4]


def test_same_input_same_result_order(client):
    payload = eco_request(
        sequence="GAATTCAAAAGATATCTTTTGGTACC",
        topology="circular",
        enzymes=[
            {"name": "KpnI", "pattern": "GGTACC", "cut_top": 5, "cut_bottom": 1},
            {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5},
            {"name": "EcoRV", "pattern": "GATATC", "cut_top": 3, "cut_bottom": 3},
        ],
    )
    r1 = client.post("/digests", json=payload).json()
    r2 = client.post("/digests", json=payload).json()
    assert r1["cuts"] == r2["cuts"]
    assert r1["fragments"] == r2["fragments"]
