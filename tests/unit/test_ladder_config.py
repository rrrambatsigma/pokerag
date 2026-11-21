"""B0-B6 ladder configuration and prompt files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from pokerag.llm.config import EXPECTED, load_ladder

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "configs" / "ladder.yaml"


def test_ladder_has_all_variants_in_order() -> None:
    ladder = load_ladder(LADDER)

    assert tuple(ladder.configs) == EXPECTED
    assert len(ladder.sampling.seeds) == 3


def test_ladder_encodes_the_intended_layers() -> None:
    ladder = load_ladder(LADDER)
    configs = ladder.configs

    assert configs["B0"].retrieval.mode == "off"
    assert configs["B1"].retrieval.mode == "off"
    assert configs["B1"].fewshot is not None
    assert configs["B2"].retrieval.mode == "dense"
    assert configs["B2"].retrieval.strategy == "fixed"
    assert not configs["B2"].reranker

    assert configs["B3"].retrieval.mode == "hybrid"
    assert configs["B3"].reranker and not configs["B3"].planner

    for name in ("B4", "B5", "B6"):
        variant = configs[name]
        assert variant.planner and variant.sufficiency and variant.verifier

    assert configs["B5"].adapter.enabled is False
    assert configs["B5"].adapter.reason
    assert configs["B6"].tools == ["document_version", "generation_check"]


def test_revision_must_be_pinned_before_a_run() -> None:
    ladder = load_ladder(LADDER)

    with pytest.raises(ValueError):
        ladder.require_pinned()


def _copy_ladder(tmp_path: Path, mutate) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(ROOT / "configs" / "prompts", root / "configs" / "prompts")
    data = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    mutate(data)
    path = root / "configs" / "ladder.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_missing_variant_is_rejected(tmp_path: Path) -> None:
    path = _copy_ladder(tmp_path, lambda data: data["configs"].pop("B6"))

    with pytest.raises(ValueError, match="B6"):
        load_ladder(path)


def test_two_seeds_are_rejected(tmp_path: Path) -> None:
    path = _copy_ladder(
        tmp_path, lambda data: data["sampling"].update({"seeds": [7, 17]})
    )

    with pytest.raises(ValueError, match="seeds"):
        load_ladder(path)


def test_b2_must_use_dense_retrieval(tmp_path: Path) -> None:
    def mutate(data):
        data["configs"]["B2"]["retrieval"]["mode"] = "hybrid"

    path = _copy_ladder(tmp_path, mutate)

    with pytest.raises(ValueError, match="B2"):
        load_ladder(path)
