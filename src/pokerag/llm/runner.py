"""Run one ladder variant over a query set and record predictions + manifest.

The runner is the only place that combines retrieval, generation, evaluation
output and the run manifest. It accepts injected components so the wiring can be
tested without a GPU.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pokerag.bench.manifest import (
    ModelSpec,
    RequestRecord,
    SamplingSpec,
    build_manifest,
    write_run,
)
from pokerag.core.jsonl import write_jsonl
from pokerag.eval.qrels import load_queries
from pokerag.llm.base import Answer, Generator
from pokerag.llm.config import Ladder, load_ladder
from pokerag.llm.prompts import PromptSet, load_prompt_set
from pokerag.llm.qwen import QwenGenerator
from pokerag.retrieval.config import load_retrieval_config
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.types import Hit

DEFAULT_RETRIEVAL = "configs/retrieval.v2.yaml"


class RetrieverLike(Protocol):
    index_version: str
    strategy: str

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[Hit]: ...

    def close(self) -> None: ...


class FakeGenerator(Protocol):
    name: str

    def generate(self, question: str, evidence: Sequence[Hit]) -> Answer: ...


def _citation_payload(answer: Answer, hits: Sequence[Hit]) -> list[dict]:
    by_id = {hit.chunk_id: hit for hit in hits}
    citations: list[dict] = []
    for chunk_id in answer.cited_chunk_ids:
        hit = by_id.get(chunk_id)
        if hit is None:
            continue
        citations.append(
            {
                "chunkId": hit.chunk_id,
                "docId": hit.doc_id,
                "charStart": hit.char_start,
                "charEnd": hit.char_end,
            }
        )
    return citations


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def prompt_set_for(ladder: Ladder, variant_name: str, root: Path) -> PromptSet:
    variant = ladder.configs[variant_name]
    return load_prompt_set(
        root / variant.system_prompt,
        root / variant.user_prompt,
        root / variant.fewshot if variant.fewshot else None,
    )


def run_variant(
    *,
    ladder_path: str | Path,
    variant_name: str,
    queries_path: str | Path,
    run_index: int = 0,
    limit: int | None = None,
    device: str | None = None,
    out_root: str | Path = "artifacts/runs",
    retrieval_config: str | Path = DEFAULT_RETRIEVAL,
    generator: Generator | None = None,
    retriever: RetrieverLike | None = None,
) -> Path:
    ladder_path = Path(ladder_path)
    ladder = load_ladder(ladder_path)
    ladder.require_pinned()
    if variant_name not in ladder.configs:
        raise ValueError(f"variant tidak dikenal: {variant_name}")
    if run_index >= len(ladder.sampling.seeds):
        raise ValueError(
            f"run_index {run_index} melebihi daftar seed {ladder.sampling.seeds}"
        )

    variant = ladder.configs[variant_name]
    root = ladder_path.resolve().parent.parent
    prompts = prompt_set_for(ladder, variant_name, root)
    queries = load_queries(Path(queries_path))
    ordered = [queries[key] for key in sorted(queries)]
    if limit is not None:
        ordered = ordered[:limit]

    cfg = load_retrieval_config(Path(retrieval_config))
    uses_retrieval = variant.retrieval.mode != "off"
    owns_retriever = False
    if uses_retrieval and retriever is None:
        retriever = Retriever(cfg, strategy=variant.retrieval.strategy)
        owns_retriever = True

    seed = ladder.sampling.seeds[run_index]

    # Two passes on purpose. bge-m3 and a 1.5B generator do not fit together in
    # 6 GB: retrieval first, free the embedder, then load the generator. The
    # per-query latency still includes both parts.
    hits_by_query: dict[str, list[Hit]] = {query.query_id: [] for query in ordered}
    retrieval_ms: dict[str, float] = {query.query_id: 0.0 for query in ordered}
    index_version = "off"

    if uses_retrieval and retriever is not None:
        index_version = retriever.index_version
        # Warm the embedder before timing; loading bge-m3 takes ~30 s on this
        # machine and must not be charged to the first user query.
        retriever.search("warmup", mode=variant.retrieval.mode, top_k=1)
        try:
            for query in ordered:
                started = time.perf_counter()
                hits_by_query[query.query_id] = retriever.search(
                    query.text,
                    mode=variant.retrieval.mode,
                    top_k=variant.retrieval.top_k,
                )
                retrieval_ms[query.query_id] = (time.perf_counter() - started) * 1000
        finally:
            if owns_retriever:
                retriever.close()
                retriever = None
                gc.collect()
                _free_cuda()

    if generator is None:
        generator = QwenGenerator(
            ladder.base_model,
            ladder.sampling,
            prompts,
            seed=seed,
            use_evidence=uses_retrieval,
            device=device,
        )

    predictions: list[dict] = []
    records: list[RequestRecord] = []
    for query in ordered:
        hits = hits_by_query[query.query_id]
        started = time.perf_counter()
        answer = generator.generate(query.text, hits)
        generation_ms = (time.perf_counter() - started) * 1000

        predictions.append(
            {
                "query_id": query.query_id,
                "text": answer.text,
                "abstained": answer.abstained,
                "citations": _citation_payload(answer, hits),
            }
        )
        records.append(
            RequestRecord(
                query_id=query.query_id,
                latency_ms=retrieval_ms[query.query_id] + generation_ms,
                input_tokens=answer.prompt_tokens,
                output_tokens=answer.completion_tokens,
                abstained=answer.abstained,
                cited_chunk_ids=answer.cited_chunk_ids,
            )
        )
    manifest = build_manifest(
        run_set=variant_name,
        run_index=run_index,
        snapshot=cfg.index_config().snapshot,
        index_version=index_version,
        strategy=variant.retrieval.strategy,
        mode=variant.retrieval.mode,
        config_path=ladder_path,
        prompt_path=root / variant.system_prompt,
        queries_path=Path(queries_path),
        model=ModelSpec(
            model_id=ladder.base_model.model_id,
            revision=ladder.base_model.revision,
            dtype=ladder.base_model.dtype,
            device=ladder.base_model.device,
        ),
        sampling=SamplingSpec(
            temperature=ladder.sampling.temperature,
            top_p=ladder.sampling.top_p,
            max_new_tokens=ladder.sampling.max_new_tokens,
            seed=seed,
        ),
        extra={
            "variant": variant_name,
            "prompt_set_version": prompts.version,
            "prompt_user": variant.user_prompt,
            "prompt_fewshot": variant.fewshot,
            "retrieval_candidates": variant.retrieval.top_k,
        },
    )
    run_dir = write_run(out_root, manifest, records)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    return run_dir
