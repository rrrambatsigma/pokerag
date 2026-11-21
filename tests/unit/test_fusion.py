import pytest

from pokerag.index.fusion import reciprocal_rank_fusion


def test_rrf_prefers_documents_found_by_both_sources() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=60)

    assert [hit.chunk_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].ranks == {"dense": 2, "bm25": 1}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_scores_follow_k() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a"]}, k=1)

    assert fused[0].score == pytest.approx(1 / 2)


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=0)


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion({}) == []


def test_rrf_keeps_stable_tie_break_by_id() -> None:
    fused = reciprocal_rank_fusion({"dense": ["z"], "bm25": ["a"]}, k=60)

    assert [hit.chunk_id for hit in fused] == ["a", "z"]
