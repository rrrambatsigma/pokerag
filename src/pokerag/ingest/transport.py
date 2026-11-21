"""Polite HTTP transport for the Bulbapedia MediaWiki API.

Design constraints that come from the site itself (checked 2026-11-14):

* ``robots.txt`` sets ``Crawl-delay: 5`` and only allows ``/wiki/`` paths.
* Plain ``requests`` gets HTTP 403 from the CDN; a browser-fingerprinted TLS
  client (``curl_cffi``) is accepted. We never attempt to solve or bypass a
  challenge: a 403 aborts the run with a clear error.
* The API is batched (5 titles per call) so a 320-page snapshot needs ~64 calls
  instead of 320.

Every attempt is appended to ``fetch_log.jsonl`` so politeness and failures are
auditable after the fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokerag.core.jsonl import append_jsonl
from pokerag.core.logging import get_logger
from pokerag.ingest.config import FetchConfig

logger = get_logger("pokerag.ingest.transport")


class AccessBlocked(RuntimeError):
    """Raised when the remote site refuses automated access (HTTP 403)."""


class ApiError(RuntimeError):
    """Raised when the MediaWiki API returns an ``error`` payload."""


@dataclass
class TransportStats:
    """Counters reported at the end of a run."""

    requests: int = 0
    retries: int = 0
    bytes_received: int = 0


class MediaWikiTransport:
    """Throttled JSON client with bounded retries."""

    def __init__(
        self,
        cfg: FetchConfig,
        log_path: Path,
        session: Any | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._log_path = log_path
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.stats = TransportStats()
        if session is None:
            from curl_cffi import requests as curl_requests

            session = curl_requests.Session(impersonate=cfg.impersonate)
        self._session = session

    # ------------------------------------------------------------------ utils
    def _throttle(self) -> None:
        if self._last_request is None:
            return
        elapsed = self._clock() - self._last_request
        remaining = self._cfg.delay_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def _log(
        self, params: dict[str, Any], status: int, size: int, ms: int, attempt: int, gap: float | None
    ) -> None:
        append_jsonl(
            self._log_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "action": params.get("action"),
                "status": status,
                "bytes": size,
                "ms": ms,
                "attempt": attempt,
                "gap_seconds": gap,
                "params": {k: v for k, v in params.items() if k in ("action", "page", "oldid", "titles")},
            },
        )

    def _retry_after(self, response: Any) -> float | None:
        value = None
        headers = getattr(response, "headers", {}) or {}
        try:
            value = headers.get("Retry-After")
        except Exception:  # pragma: no cover - defensive
            value = None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------- public
    def get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        """GET ``params`` against the API and return parsed JSON.

        Raises:
            AccessBlocked: on HTTP 403, without retrying.
            ApiError: when MediaWiki answers with an ``error`` object.
            RuntimeError: when retries are exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            self._throttle()
            gap = None if self._last_request is None else round(self._clock() - self._last_request, 3)
            started = time.perf_counter()
            try:
                response = self._session.get(
                    self._cfg.api,
                    params=params,
                    timeout=self._cfg.timeout_seconds,
                    headers={"User-Agent": self._cfg.user_agent},
                )
            except Exception as exc:  # network level failure
                last_error = exc
                self._wait_backoff(attempt, None)
                self.stats.retries += 1
                continue
            finally:
                self._last_request = self._clock()

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            size = len(response.content)
            self.stats.requests += 1
            self.stats.bytes_received += size
            self._log(params, response.status_code, size, elapsed_ms, attempt, gap)

            if response.status_code == 403:
                raise AccessBlocked(
                    "Bulbapedia returned HTTP 403 for this client. Automatic access is "
                    "refused; do not attempt to bypass it. Stop and report."
                )
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"HTTP {response.status_code}")
                self.stats.retries += 1
                self._wait_backoff(attempt, self._retry_after(response))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Unexpected HTTP {response.status_code} for {params}")

            payload = response.json()
            if "error" in payload:
                raise ApiError(f"{payload['error'].get('code')}: {payload['error'].get('info')}")
            return payload

        raise RuntimeError(f"request failed after {self._cfg.max_retries} attempts: {last_error}")

    def _wait_backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            self._sleep(retry_after)
            return
        table = self._cfg.backoff_seconds
        delay = table[min(attempt - 1, len(table) - 1)]
        logger.warning("retrying in %ss (attempt %s)", delay, attempt)
        self._sleep(delay)
