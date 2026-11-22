"""Run the API with uvicorn: ``pokerag-api --variant B4 --port 8000``.

The default serves B4 with a 4-bit Qwen generator, so bge-m3, the reranker and
the generator fit together in 6 GB. ``--no-generator`` keeps the stub path for
UI work without a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerag.api.app import DEFAULT_CONFIG, create_app
from pokerag.llm.config import load_ladder
from pokerag.llm.prompts import load_prompt_set
from pokerag.llm.qwen import QwenGenerator
from pokerag.llm.verify import QwenVerifier, VerifyingGenerator
from pokerag.core.logging import get_logger
from pokerag.retrieval.rerank import CrossEncoderReranker

DEFAULT_LADDER = "configs/ladder.yaml"
LOGGER = get_logger("pokerag.api")


def build_app(args: argparse.Namespace):
    if args.no_generator:
        return create_app(config_path=args.config)

    ladder = load_ladder(args.ladder)
    ladder.require_pinned()
    if args.variant not in ladder.configs:
        raise SystemExit(f"variant tidak dikenal: {args.variant}")
    variant = ladder.configs[args.variant]
    root = Path(args.ladder).resolve().parent.parent

    prompts = load_prompt_set(
        root / variant.system_prompt,
        root / variant.user_prompt,
        root / variant.fewshot if variant.fewshot else None,
    )
    base = QwenGenerator(
        ladder.base_model,
        ladder.sampling,
        prompts,
        use_evidence=variant.retrieval.mode != "off",
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    generator = base
    if variant.verifier:
        if ladder.verifier is None:
            raise SystemExit("config memakai verifier tetapi blok verifier tidak ada")
        verifier = QwenVerifier(
            base,
            (root / ladder.verifier.system_prompt).read_text(encoding="utf-8"),
            (root / ladder.verifier.user_prompt).read_text(encoding="utf-8"),
        )
        generator = VerifyingGenerator(
            base,
            verifier,
            system_prompt=base.system_text,
            repair_template=(root / ladder.verifier.repair_user_prompt).read_text(
                encoding="utf-8"
            ),
            max_repairs=ladder.verifier.max_repairs,
        )

    reranker = None
    if variant.reranker:
        if ladder.reranker is None:
            raise SystemExit("config memakai reranker tetapi blok reranker tidak ada")
        reranker = CrossEncoderReranker(ladder.reranker, device=args.device)

    app = create_app(
        config_path=args.config,
        generator=generator,
        reranker=reranker,
        planner=variant.planner,
        variant_name=args.variant,
    )

    # Load bge-m3 before the first user request, otherwise the first chat waits
    # ~30 s while the embedder is materialised.
    if variant.retrieval.mode != "off":
        retriever = app.state.pool.get(app.state.cfg.strategy)
        retriever.search("warmup", mode=variant.retrieval.mode, top_k=1)
        LOGGER.info("retrieval warmup selesai")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-api", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ladder", default=DEFAULT_LADDER)
    parser.add_argument("--variant", default="B4", help="B0 sampai B6")
    parser.add_argument("--no-generator", action="store_true", help="pakai stub")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="4-bit NF4 untuk Qwen (default: aktif)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
