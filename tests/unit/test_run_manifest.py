"""Run manifest and repeatability helpers, without a model or network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerag.bench.manifest import (
    ModelSpec,
    RequestRecord,
    SamplingSpec,
    build_manifest,
    environment,
    file_digest,
    git_state,
    new_run_id,
    summarise,
    write_run,
)


def records() -> list[RequestRecord]:
    return [
        RequestRecord("q1", latency_ms=10.0, input_tokens=100, output_tokens=20),
        RequestRecord("q2", latency_ms=20.0, input_tokens=200, output_tokens=30, abstained=True),
        RequestRecord("q3", latency_ms=30.0, output_tokens=40),
        RequestRecord("q4", latency_ms=40.0, input_tokens=400, output_tokens=50),
        RequestRecord("q5", latency_ms=5.0, error="timeout"),
    ]


def test_new_run_id() -> None:
    assert new_run_id("b3", 0) == "b3-r00"
    assert new_run_id("b3", 12) == "b3-r12"
    with pytest.raises(ValueError):
        new_run_id("b3", -1)


def test_environment_reports_python_and_packages() -> None:
    env = environment()

    assert env["python"]
    assert isinstance(env["packages"], dict)
    assert "pokerag" in env["packages"]


def test_git_state_returns_sha_or_none() -> None:
    sha, dirty = git_state()

    assert sha is None or len(sha) == 40
    assert dirty is None or isinstance(dirty, bool)


def test_file_digest_missing_returns_none(tmp_path: Path) -> None:
    assert file_digest(None) is None
    assert file_digest(tmp_path / "nope") is None


def test_build_manifest_records_hashes_and_specs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    prompt = tmp_path / "prompt.txt"
    qrels = tmp_path / "qrels.jsonl"
    queries = tmp_path / "queries.csv"
    for path in (config, prompt, qrels, queries):
        path.write_text("x", encoding="utf-8")

    manifest = build_manifest(
        run_set="b3",
        run_index=1,
        snapshot="pokerag-corpus-v2",
        index_version="abc123",
        strategy="heading",
        mode="hybrid",
        config_path=config,
        prompt_path=prompt,
        qrels_path=qrels,
        queries_path=queries,
        model=ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "rev", "float16", "cuda"),
        sampling=SamplingSpec(0.2, 0.9, 512, 7),
        extra={"temperature_note": "fixed"},
    )

    assert manifest.run_id == "b3-r01"
    assert manifest.config_sha256 is not None
    assert len(manifest.config_sha256) == 64
    assert manifest.prompt_sha256 is not None
    assert manifest.qrels_sha256 is not None
    assert manifest.queries_sha256 is not None
    payload = manifest.to_dict()
    assert payload["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert payload["sampling"]["seed"] == 7
    assert payload["extra"]["temperature_note"] == "fixed"


def test_summarise_computes_latency_tokens_and_errors() -> None:
    summary = summarise(records())

    assert summary["requests"] == 5
    assert summary["errors"] == 1
    assert summary["abstained"] == 1
    assert summary["abstain_rate"] == pytest.approx(0.2)
    assert summary["latency_ms"]["mean"] == pytest.approx(25.0)
    assert summary["latency_ms"]["p50"] == pytest.approx(20.0)
    assert summary["latency_ms"]["p95"] == pytest.approx(40.0)
    assert summary["latency_ms"]["max"] == pytest.approx(40.0)
    assert summary["tokens"]["input_total"] == 700
    assert summary["tokens"]["output_total"] == 140
    assert summary["tokens"]["with_input"] == 3
    assert summary["tokens"]["with_output"] == 4


def test_summarise_empty_is_none_not_zero() -> None:
    summary = summarise([])

    assert summary["requests"] == 0
    assert summary["latency_ms"]["mean"] is None
    assert summary["tokens"]["input_total"] is None
    assert summary["abstain_rate"] is None


def test_write_run_roundtrip(tmp_path: Path) -> None:
    manifest = build_manifest(
        run_set="b2",
        run_index=0,
        snapshot="pokerag-corpus-v2",
        index_version="abc",
        strategy="fixed",
        mode="dense",
    )
    run_dir = write_run(tmp_path / "runs", manifest, records())

    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    lines = (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()

    assert run_dir.name == "b2-r00"
    assert payload["run_id"] == "b2-r00"
    assert payload["summary"]["requests"] == 5
    assert len(lines) == 5
    assert json.loads(lines[0])["query_id"] == "q1"
