"""Crawl politeness is a requirement, so it is tested without the network."""

from __future__ import annotations

from pathlib import Path

from pokerag.ingest.config import FetchConfig
from pokerag.ingest.transport import MediaWikiTransport


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeSleep:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(round(seconds, 3))
        self.clock.now += seconds


class FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    content = b'{"query":{}}'

    def json(self) -> dict:
        return {"query": {}}


class FakeSession:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = 0

    def get(self, *args, **kwargs) -> FakeResponse:
        self.calls += 1
        self.clock.now += 0.01  # request duration
        return FakeResponse()


def test_requests_are_spaced_by_the_configured_delay(tmp_path: Path):
    clock = FakeClock()
    sleep = FakeSleep(clock)
    cfg = FetchConfig(
        api="https://example.invalid/w/api.php",
        user_agent="PokeRAG-test/0.1",
        delay_seconds=5.0,
        batch_titles=1,
        timeout_seconds=1,
        max_retries=1,
        backoff_seconds=[0],
    )
    transport = MediaWikiTransport(
        cfg,
        tmp_path / "fetch_log.jsonl",
        session=FakeSession(clock),
        sleep=sleep,
        clock=clock,
    )

    transport.get_json({"action": "query"})
    transport.get_json({"action": "query"})
    transport.get_json({"action": "query"})

    assert sleep.calls == [5.0, 5.0]
    assert transport.stats.requests == 3
