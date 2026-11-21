"""Typed configuration for the B0-B6 ladder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

EXPECTED = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
MODES = ("off", "dense", "bm25", "hybrid")
STRATEGIES = ("fixed", "heading", "hierarchical")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelRef(Strict):
    model_id: str
    revision: str | None = None
    dtype: str = "float16"
    device: str = "cuda"


class SamplingRef(Strict):
    temperature: float
    top_p: float
    max_new_tokens: int
    seeds: list[int] = Field(min_length=1)


class RetrievalRef(Strict):
    mode: Literal["off", "dense", "bm25", "hybrid"]
    strategy: Literal["fixed", "heading", "hierarchical"]
    top_k: int


class AdapterRef(Strict):
    enabled: bool = False
    reason: str | None = None


class Variant(Strict):
    system_prompt: str
    user_prompt: str
    fewshot: str | None = None
    retrieval: RetrievalRef
    reranker: bool = False
    planner: bool = False
    sufficiency: bool = False
    verifier: bool = False
    adapter: AdapterRef = Field(default_factory=AdapterRef)
    tools: list[str] = Field(default_factory=list)
    notes: str | None = None


class Ladder(Strict):
    base_model: ModelRef
    sampling: SamplingRef
    configs: dict[str, Variant]

    def require_pinned(self) -> None:
        """Runs may only start once the base model revision is pinned."""
        if not self.base_model.revision:
            raise ValueError(
                "base_model.revision belum di-pin; isi commit sha model sebelum run"
            )


def _prompt_text(root: Path, relative: str, problems: list[str]) -> str:
    path = root / relative
    if not path.is_file():
        problems.append(f"prompt tidak ada: {relative}")
        return ""
    return path.read_text(encoding="utf-8")


def _validate_variant(name: str, variant: Variant, root: Path, problems: list[str]) -> None:
    system = _prompt_text(root, variant.system_prompt, problems)
    user = _prompt_text(root, variant.user_prompt, problems)
    if variant.fewshot:
        _prompt_text(root, variant.fewshot, problems)

    if not system:
        problems.append(f"{name}: system prompt kosong")
    if "{question}" not in user:
        problems.append(f"{name}: user prompt tidak memuat {{question}}")
    if variant.retrieval.mode != "off" and "{evidence}" not in user:
        problems.append(f"{name}: retrieval aktif tetapi user prompt tidak memuat {{evidence}}")
    if name in ("B0", "B1") and variant.retrieval.mode != "off":
        problems.append(f"{name}: retrieval harus off")
    if variant.retrieval.mode == "off" and name not in ("B0", "B1"):
        problems.append(f"{name}: retrieval off hanya sah untuk B0/B1")
    if name == "B2" and variant.retrieval.mode != "dense":
        problems.append("B2: mode harus dense")
    if variant.retrieval.mode == "dense" and name != "B2":
        problems.append(f"{name}: mode dense hanya untuk B2")
    if name in ("B3", "B4", "B5", "B6") and variant.retrieval.mode != "hybrid":
        problems.append(f"{name}: mode harus hybrid")
    if name == "B2" and variant.reranker:
        problems.append("B2: reranker belum boleh aktif")
    if name in ("B3", "B4", "B5", "B6") and not variant.reranker:
        problems.append(f"{name}: reranker harus aktif")
    if name in ("B4", "B5", "B6"):
        for flag in ("planner", "sufficiency", "verifier"):
            if not getattr(variant, flag):
                problems.append(f"{name}: {flag} harus aktif")
    if name == "B5" and not variant.adapter.enabled and not variant.adapter.reason:
        problems.append("B5: adapter nonaktif harus punya alasan")
    if name == "B6" and not variant.tools:
        problems.append("B6: tools tidak boleh kosong")


def load_ladder(path: str | Path) -> Ladder:
    """Load and validate the ladder. Prompt paths resolve from the repo root."""
    ladder_path = Path(path)
    data = yaml.safe_load(ladder_path.read_text(encoding="utf-8"))
    ladder = Ladder.model_validate(data)

    problems: list[str] = []
    if tuple(ladder.configs) != EXPECTED:
        problems.append(
            f"configs harus berurutan {EXPECTED}, ditemukan {tuple(ladder.configs)}"
        )
    if len(ladder.sampling.seeds) < 3:
        problems.append("sampling.seeds minimal tiga untuk tiga run independen")

    root = ladder_path.resolve().parent.parent
    for name in EXPECTED:
        variant = ladder.configs.get(name)
        if variant is None:
            problems.append(f"config {name} tidak ada")
            continue
        _validate_variant(name, variant, root, problems)

    if problems:
        raise ValueError("ladder tidak valid:\n- " + "\n- ".join(problems))
    return ladder
