"""酶切核心算法测试。"""

import pytest

from app.digest import CutConflictError, run_digest
from app.iupac import reverse_complement


class FakeEnzyme:
    def __init__(self, name, pattern, cut_top, cut_bottom):
        self.name = name
        self.pattern = pattern
        self.cut_top = cut_top
        self.cut_bottom = cut_bottom


# 常用酶：EcoRI G^AATTC（5' 黏端 AATT）
ECORI = FakeEnzyme("EcoRI", "GAATTC", 1, 5)
# EcoRV GAT^ATC（平端）
ECORV = FakeEnzyme("EcoRV", "GATATC", 3, 3)
# KpnI GGTAC^C（3' 黏端 GTAC）
KPNI = FakeEnzyme("KpnI", "GGTACC", 5, 1)


def test_reverse_complement_iupac():
    assert reverse_complement("GAATTC") == "GAATTC"  # 回文
    assert reverse_complement("GCGGCC") == "GGCCGC"
    assert reverse_complement("ACGTYR") == "YRACGT"


def test_ecori_linear_sticky_5prime():
    seq = "AAAAGAATTCTTTT"  # 位点起于 4
    result = run_digest(seq, "linear", [ECORI])
    assert len(result["cuts"]) == 1
    cut = result["cuts"][0]
    assert (cut["top_cut"], cut["bottom_cut"]) == (5, 9)
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "AATT"
    assert cut["enzymes"] == ["EcoRI"]
    assert cut["sites"][0]["start"] == 4
    assert cut["sites"][0]["strand"] == "+"

    frags = result["fragments"]
    assert len(frags) == 2
    assert (frags[0]["start"], frags[0]["end"], frags[0]["length"]) == (0, 5, 5)
    assert frags[0]["sequence"] == "AAAAG"
    assert frags[0]["left_end"]["type"] == "none"
    assert frags[0]["right_end"] == {"type": "5prime", "overhang": "AATT"}
    assert (frags[1]["start"], frags[1]["end"], frags[1]["length"]) == (5, 14, 9)
    assert frags[1]["sequence"] == "AATTCTTTT"
    assert frags[1]["left_end"] == {"type": "5prime", "overhang": "AATT"}
    assert frags[1]["right_end"]["type"] == "none"


def test_ecorv_blunt():
    seq = "CCGATATCGG"  # 位点起于 2
    result = run_digest(seq, "linear", [ECORV])
    cut = result["cuts"][0]
    assert (cut["top_cut"], cut["bottom_cut"]) == (5, 5)
    assert cut["end_type"] == "blunt"
    assert cut["overhang"] == ""
    assert [f["length"] for f in result["fragments"]] == [5, 5]


def test_kpnI_sticky_3prime():
    seq = "AAAGGTACCTTT"  # 位点起于 3
    result = run_digest(seq, "linear", [KPNI])
    cut = result["cuts"][0]
    assert (cut["top_cut"], cut["bottom_cut"]) == (8, 4)
    assert cut["end_type"] == "3prime"
    assert cut["overhang"] == "GTAC"


def test_circular_site_spanning_origin():
    # GAATTC 跨越首尾：位置 8,9,0,1,2,3
    seq = "ATTCGGGGGA"
    result = run_digest(seq, "circular", [ECORI])
    assert len(result["cuts"]) == 1
    cut = result["cuts"][0]
    assert cut["sites"][0]["start"] == 8
    assert (cut["top_cut"], cut["bottom_cut"]) == (9, 3)
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "AATT"
    # 单切口开环：一条全长片段
    assert len(result["fragments"]) == 1
    frag = result["fragments"][0]
    assert frag["length"] == len(seq)
    assert frag["start"] == 9
    assert frag["sequence"] == seq[9:] + seq[:9]
    assert frag["left_end"]["type"] == "5prime"
    assert frag["right_end"]["type"] == "5prime"


def test_circular_two_cuts_two_fragments():
    # 两个 EcoRI 位点，均不跨原点
    seq = "GAATTC" + "CCCC" + "GAATTC" + "GGGG"  # 位点 0 与 10
    result = run_digest(seq, "circular", [ECORI])
    assert len(result["cuts"]) == 2
    frags = result["fragments"]
    assert len(frags) == 2
    assert sorted(f["length"] for f in frags) == [10, 10]
    assert all(f["left_end"]["type"] == "5prime" for f in frags)


def test_ambiguous_bases_in_pattern_and_sequence():
    # 模式含 N：ANNT 匹配 AGTT（偏移 2/2 镜像对称，回文模式只计一次，
    # 使本测试专注于简并匹配；方向折叠见回文系列测试）
    enz = FakeEnzyme("Amb1", "ANNT", 2, 2)
    result = run_digest("CAGTT", "linear", [enz])
    assert len(result["cuts"]) == 1
    assert result["cuts"][0]["sites"][0]["start"] == 1
    # 序列含简并碱基：模式 GCGC 可匹配 GCGN（N 与 C 有交集）
    enz2 = FakeEnzyme("Amb2", "GCGC", 2, 2)
    result2 = run_digest("GGCGCN", "linear", [enz2])
    assert len(result2["cuts"]) == 1
    assert result2["cuts"][0]["sites"][0]["start"] == 1


def test_palindrome_asymmetric_offsets_both_orientations():
    # 回文模式 ATAT（k=4）偏移 (1,1) 不满足镜像对称（1+1 != 4）：
    # 同一识别位点上正向结合切 (2,2)、反向结合切 (4,4)，分别保留
    enz = FakeEnzyme("Pal", "ATAT", 1, 1)
    result = run_digest("AATATG", "linear", [enz])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(2, 2), (4, 4)]
    fwd, rev = result["cuts"]
    assert fwd["end_type"] == "blunt" and rev["end_type"] == "blunt"
    assert fwd["sites"] == [
        {"enzyme": "Pal", "start": 1, "strand": "+", "matched": "ATAT"}
    ]
    assert rev["sites"] == [
        {"enzyme": "Pal", "start": 1, "strand": "-", "matched": "ATAT"}
    ]
    # 两个边界 -> 三条片段
    assert [f["length"] for f in result["fragments"]] == [2, 2, 2]
    assert [f["sequence"] for f in result["fragments"]] == ["AA", "TA", "TG"]


def test_palindrome_mirror_symmetric_offsets_not_double_counted():
    # 回文模式偏移镜像对称（cut_top + cut_bottom == k）时两个方向
    # 产生同一物理切口：只搜一次，切口与位点均不重复计数
    enz = FakeEnzyme("PalSym", "ATAT", 1, 3)  # 1+3 == 4
    result = run_digest("AATATG", "linear", [enz])
    assert len(result["cuts"]) == 1
    cut = result["cuts"][0]
    assert (cut["top_cut"], cut["bottom_cut"]) == (2, 4)
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "TA"
    assert cut["sites"] == [
        {"enzyme": "PalSym", "start": 1, "strand": "+", "matched": "ATAT"}
    ]
    assert [f["length"] for f in result["fragments"]] == [2, 4]


def test_palindrome_asymmetric_sticky_ends():
    # 非镜像对称偏移的回文产生黏端时，两个方向各自换算切口与突出端
    enz = FakeEnzyme("PalSticky", "GANTC", 0, 2)  # k=5，0+2 != 5
    result = run_digest("AAGATTCCAA", "linear", [enz])  # 位点 GATTC 起于 2
    assert [
        (c["top_cut"], c["bottom_cut"], c["end_type"], c["overhang"])
        for c in result["cuts"]
    ] == [(2, 4, "5prime", "GA"), (5, 7, "5prime", "TC")]
    assert [s["strand"] for c in result["cuts"] for s in c["sites"]] == ["+", "-"]
    assert [f["length"] for f in result["fragments"]] == [2, 3, 5]


def test_palindrome_asymmetric_offsets_circular():
    # 环形序列上同一回文位点的正反结合同样落在不同物理坐标
    enz = FakeEnzyme("Pal", "ATAT", 1, 1)
    result = run_digest("AATATG", "circular", [enz])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(2, 2), (4, 4)]
    frags = result["fragments"]
    assert len(frags) == 2
    assert sorted(f["length"] for f in frags) == [2, 4]
    assert sum(f["length"] for f in frags) == 6


def test_overlapping_sites():
    # GATATC 与 TATCGA 重叠于同一区域
    enz1 = FakeEnzyme("E1", "GATATC", 3, 3)
    enz2 = FakeEnzyme("E2", "TATCGA", 2, 2)
    seq = "AAGATATCGAA"
    result = run_digest(seq, "linear", [enz1, enz2])
    assert len(result["cuts"]) == 2
    starts = sorted(s["start"] for c in result["cuts"] for s in c["sites"])
    assert starts == [2, 4]


def test_reverse_strand_site():
    # 非回文模式 GCGGCC，其反向互补 GGCCGC 出现在 top 链上
    enz = FakeEnzyme("Rev", "GCGGCC", 2, 4)
    seq = "AAGGCCGCTT"  # GGCCGC 起于 2
    result = run_digest(seq, "linear", [enz])
    assert len(result["cuts"]) == 1
    cut = result["cuts"][0]
    assert cut["sites"][0]["strand"] == "-"
    # 反向命中：top = 2 + 6 - 4 = 4，bottom = 2 + 6 - 2 = 6
    assert (cut["top_cut"], cut["bottom_cut"]) == (4, 6)
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "CC"  # seq[4:6]


def test_physical_cut_deduplicated():
    # 两个酶产生完全相同的物理切口 -> 只保留一条切口记录
    other = FakeEnzyme("EcoRI-twin", "GAATTC", 1, 5)
    result = run_digest("AAAAGAATTCTTTT", "linear", [ECORI, other])
    assert len(result["cuts"]) == 1
    assert result["cuts"][0]["enzymes"] == ["EcoRI", "EcoRI-twin"]
    assert len(result["fragments"]) == 2


def test_circular_modulo_coordinate_conflict_rejected():
    # 环形 AAAA 上模式 AAAA：偏移 (0,3) 在 start 0 产生原始切口 (0,3)，
    # 偏移 (1,0) 在 start 3 产生原始切口 (4,3)——取模后同为 (0,3)，
    # 但末端分别为 5prime/AAA 与 3prime/A，合并结果会随酶名排序变化
    e1 = FakeEnzyme("E1", "AAAA", 0, 3)
    e2 = FakeEnzyme("E2", "AAAA", 1, 0)
    with pytest.raises(CutConflictError) as excinfo:
        run_digest("AAAA", "circular", [e1, e2])
    conflicts = excinfo.value.conflicts
    coords = [(c["top_cut"], c["bottom_cut"]) for c in conflicts]
    assert (0, 3) in coords
    entry = conflicts[coords.index((0, 3))]
    assert entry["ends"] == [("3prime", "A"), ("5prime", "AAA")]
    assert "(0, 3)" in str(excinfo.value)
    # 与酶传入顺序无关：同样的冲突必然同样被拒绝
    with pytest.raises(CutConflictError):
        run_digest("AAAA", "circular", [e2, e1])


def test_circular_modulo_coordinate_consistent_merge():
    # 取模后同坐标且末端一致：合法合并，酶与位点照常归并。
    # E1 在 start 0 产生原始切口 (0,3)；E2 在 start 3 产生原始切口 (4,7)，
    # 取模后同为 (0,3)，且末端同为 5prime/AAA
    e1 = FakeEnzyme("E1", "AAAA", 0, 3)
    e2 = FakeEnzyme("E2", "AAAA", 1, 4)
    result = run_digest("AAAA", "circular", [e1, e2])
    assert len(result["cuts"]) == 4  # (0,3) (1,0) (2,1) (3,2)，每个坐标两酶合并
    cut = next(c for c in result["cuts"] if (c["top_cut"], c["bottom_cut"]) == (0, 3))
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "AAA"
    assert cut["enzymes"] == ["E1", "E2"]
    assert [(s["enzyme"], s["start"]) for s in cut["sites"]] == [("E1", 0), ("E2", 3)]
    # 环形跨原点片段按已有规则生成：四个唯一边界 -> 四条长度 1 的片段
    assert [f["length"] for f in result["fragments"]] == [1, 1, 1, 1]
    assert sum(f["length"] for f in result["fragments"]) == 4


def test_circular_shared_top_cut_single_boundary():
    # 两种酶识别同一 AC 位点、cut_top 均为 0，cut_bottom 分别为 0/1：
    # 产生两个物理切口 (1,1) 与 (1,2)，但正链上只有唯一断裂点
    e1 = FakeEnzyme("E1", "AC", 0, 0)
    e2 = FakeEnzyme("E2", "AC", 0, 1)
    result = run_digest("AACAA", "circular", [e1, e2])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(1, 1), (1, 2)]
    # 唯一边界开环：一条全长片段，不丢失
    frags = result["fragments"]
    assert len(frags) == 1
    frag = frags[0]
    assert (frag["index"], frag["start"], frag["end"], frag["length"]) == (0, 1, 1, 5)
    assert frag["sequence"] == "ACAAA"  # seq[1:] + seq[:1]
    assert frag["wraps_origin"] is True
    # 同坐标末端信息取稳定切口顺序中的第一条（(1,1) blunt 先于 (1,2)）
    assert frag["left_end"] == {"type": "blunt", "overhang": ""}
    assert frag["right_end"] == {"type": "blunt", "overhang": ""}


def test_circular_shared_top_cut_multiple_boundaries():
    # 共用正链坐标的切口归并为一个边界；多个边界时片段长度之和等于序列长度
    e1 = FakeEnzyme("E1", "AC", 0, 0)
    e2 = FakeEnzyme("E2", "AC", 0, 1)
    e3 = FakeEnzyme("E3", "AA", 0, 0)  # 额外边界 top=0、3、4
    result = run_digest("AACAA", "circular", [e1, e2, e3])
    boundaries = sorted({c["top_cut"] for c in result["cuts"]})
    assert boundaries == [0, 1, 3, 4]
    frags = result["fragments"]
    assert len(frags) == len(boundaries)
    assert sum(f["length"] for f in frags) == 5
    assert [f["index"] for f in frags] == [0, 1, 2, 3]
    assert [f["start"] for f in frags] == boundaries  # 按边界坐标稳定排列
    # 每个片段的右端即下一边界的第一条切口
    for idx, frag in enumerate(frags):
        assert frag["end"] == boundaries[(idx + 1) % len(boundaries)]


def test_no_cut_keeps_whole_molecule():
    seq = "AAAAAAAAAA"
    for topology in ("linear", "circular"):
        result = run_digest(seq, topology, [ECORI])
        assert result["cuts"] == []
        assert len(result["fragments"]) == 1
        frag = result["fragments"][0]
        assert frag["length"] == len(seq)
        assert frag["sequence"] == seq
        assert frag["left_end"]["type"] == "none"
        assert frag["right_end"]["type"] == "none"


def test_linear_endpoint_cut_at_zero_sets_left_end():
    # 切口 (0,1) 位于分子左端点：不产生新边界，但首片段左端携带
    # 该切口的 5' 突出端 A，而不是天然末端 none
    enz = FakeEnzyme("End5", "AAG", 0, 1)
    result = run_digest("AAGCCCC", "linear", [enz])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(0, 1)]
    cut = result["cuts"][0]
    assert cut["end_type"] == "5prime"
    assert cut["overhang"] == "A"
    frags = result["fragments"]
    assert len(frags) == 1  # 端点切口不增加片段
    frag = frags[0]
    assert (frag["start"], frag["end"], frag["length"]) == (0, 7, 7)
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}


def test_linear_endpoint_cut_at_length_sets_right_end():
    # 切口 top_cut == L 位于分子右端点：末片段右端携带该切口的 3' 突出端
    enz = FakeEnzyme("End3", "AAG", 3, 2)  # 位点起于 4，切口 (7,6)
    result = run_digest("CCCCAAG", "linear", [enz])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(7, 6)]
    cut = result["cuts"][0]
    assert cut["end_type"] == "3prime"
    assert cut["overhang"] == "G"
    frags = result["fragments"]
    assert len(frags) == 1
    assert frags[0]["left_end"] == {"type": "none", "overhang": ""}
    assert frags[0]["right_end"] == {"type": "3prime", "overhang": "G"}


def test_linear_endpoint_cuts_at_both_ends():
    # 左右端点同时有切口：首片段左端与末片段右端分别携带各自切口末端
    left = FakeEnzyme("LeftEnd", "AAG", 0, 1)  # 切口 (0,1)：5prime A
    right = FakeEnzyme("RightEnd", "CCC", 3, 3)  # 切口 (6,6) 与端点切口 (7,7)：blunt
    result = run_digest("AAGCCCC", "linear", [left, right])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [
        (0, 1),
        (6, 6),
        (7, 7),
    ]
    frags = result["fragments"]
    assert [f["length"] for f in frags] == [6, 1]  # 端点切口不增加片段
    assert frags[0]["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frags[0]["right_end"] == {"type": "blunt", "overhang": ""}
    assert frags[1]["left_end"] == {"type": "blunt", "overhang": ""}
    assert frags[1]["right_end"] == {"type": "blunt", "overhang": ""}


def test_linear_endpoint_cut_with_internal_cut():
    # 端点切口与内部切口共存：端点末端写入首片段，内部边界照常
    end5 = FakeEnzyme("End5", "AAG", 0, 1)  # 切口 (0,1)
    result = run_digest("AAGCCCCGAATTC", "linear", [end5, ECORI])  # EcoRI 切口 (8,12)
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(0, 1), (8, 12)]
    frags = result["fragments"]
    assert [f["length"] for f in frags] == [8, 5]
    assert frags[0]["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frags[0]["right_end"] == {"type": "5prime", "overhang": "AATT"}
    assert frags[1]["left_end"] == {"type": "5prime", "overhang": "AATT"}
    assert frags[1]["right_end"] == {"type": "none", "overhang": ""}


def test_linear_shared_endpoint_cut_stable_order():
    # 多个切口共用同一端点：沿用稳定切口顺序（(top_cut, bottom_cut) 升序）
    # 的第一条——(0,1) 5prime A 先于 (0,2) 5prime AA，与酶传入顺序无关
    e1 = FakeEnzyme("E1", "AAG", 0, 2)
    e2 = FakeEnzyme("E2", "AAG", 0, 1)
    result = run_digest("AAGCCCC", "linear", [e1, e2])
    assert [(c["top_cut"], c["bottom_cut"]) for c in result["cuts"]] == [(0, 1), (0, 2)]
    assert len(result["fragments"]) == 1
    frag = result["fragments"][0]
    assert frag["left_end"] == {"type": "5prime", "overhang": "A"}
    assert frag["right_end"] == {"type": "none", "overhang": ""}
    # 调换酶传入顺序结果一致
    again = run_digest("AAGCCCC", "linear", [e2, e1])
    assert again == result


def test_linear_no_endpoint_cut_keeps_none_ends():
    # 切口不在端点时，分子两端保持天然末端 none（既有行为不变）
    result = run_digest("AAAAGAATTCTTTT", "linear", [ECORI])
    assert result["fragments"][0]["left_end"]["type"] == "none"
    assert result["fragments"][1]["right_end"]["type"] == "none"


def test_multiple_enzymes_combined():
    seq = "GAATTC" + "AAAA" + "GATATC" + "TTTT"  # EcoRI@0, EcoRV@10
    result = run_digest(seq, "linear", [ECORI, ECORV])
    assert len(result["cuts"]) == 2
    assert [f["length"] for f in result["fragments"]] == [1, 12, 7]
    # 切口按 top 位置排序
    assert [c["top_cut"] for c in result["cuts"]] == [1, 13]


def test_deterministic_output_order():
    seq = "GAATTCAAAAAGATATCTTTTGGTACCGGGG"
    enzymes = [KPNI, ECORI, ECORV]
    first = run_digest(seq, "circular", enzymes)
    second = run_digest(seq, "circular", list(reversed(enzymes)))
    assert first == second  # 与酶传入顺序无关，结果完全一致
