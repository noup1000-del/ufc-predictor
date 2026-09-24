"""Shared HTTP access: one session, global rate limit, retries, raw-HTML cache.

Every network request in the project must go through `HttpClient.fetch`.
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import requests

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(Exception):
    """Raised when a page cannot be fetched after all attempts."""

    def __init__(self, url: str, reason: str):
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason


class RateLimiter:
    """Guarantees at least `min_interval` seconds between consecutive calls to `wait()`."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                remaining = self._last + self.min_interval - now
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last = now


_CHALLENGE_MARKERS = (b"Checking your browser", b"'/__c'", b'"/__c"')


def is_bot_challenge(content: bytes) -> bool:
    head = content[:20000]
    return any(m in head for m in _CHALLENGE_MARKERS)


def decode_html(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


class HttpClient:
    def __init__(
        self,
        cache_dir: str | Path,
        user_agent: str,
        min_interval: float = 1.0,
        timeout: float = 15,
        max_attempts: int = 3,
        backoff_base: float = 2,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        host_intervals: dict[str, float] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._sleep = sleep
        self.limiter = RateLimiter(min_interval, clock=clock, sleep=sleep)
        # Slower per-host limits on top of the global one (e.g. a site's robots.txt crawl-delay).
        self.host_limiters = {host.lower(): RateLimiter(sec, clock=clock, sleep=sleep)
                              for host, sec in (host_intervals or {}).items()}
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.stats = {"network": 0, "cache_hits": 0, "failures": 0}

    def cache_path(self, page_type: str, page_id: str) -> Path:
        for part in (page_type, page_id):
            if not _SAFE_ID.match(part):
                raise ValueError(f"Unsafe cache key component: {part!r}")
        return self.cache_dir / page_type / f"{page_id}.html"

    def is_cached(self, page_type: str, page_id: str) -> bool:
        return self.cache_path(page_type, page_id).exists()

    def fetch(self, page_type: str, page_id: str, url: str, refresh: bool = False) -> str:
        """Return the page HTML, from cache unless `refresh` is set or it isn't cached yet."""
        path = self.cache_path(page_type, page_id)
        if not refresh and path.exists():
            self.stats["cache_hits"] += 1
            logger.debug("cache hit %s/%s", page_type, page_id)
            return decode_html(path.read_bytes())

        content = self._get(url)
        _write_atomic(path, content)
        return decode_html(content)

    def download(self, url: str, dest: str | Path, refresh: bool = False) -> Path:
        """Download `url` to `dest` (atomically) unless it already exists."""
        dest = Path(dest)
        if not refresh and dest.exists():
            self.stats["cache_hits"] += 1
            return dest
        _write_atomic(dest, self._get(url))
        return dest

    def get(self, url: str, params: dict | None = None) -> bytes:
        """Uncached GET (still rate limited and retried)."""
        if params:
            url = requests.Request("GET", url, params=params).prepare().url
        return self._get(url)

    def _get(self, url: str) -> bytes:
        last_reason = "unknown"
        host_limiter = self.host_limiters.get((urlsplit(url).hostname or "").lower())
        for attempt in range(1, self.max_attempts + 1):
            if host_limiter is not None:
                host_limiter.wait()
            self.limiter.wait()
            self.stats["network"] += 1
            retry_after: float | None = None
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_reason = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200 and resp.content:
                    if is_bot_challenge(resp.content):
                        # Never cache or retry: the site is refusing automated access.
                        self.stats["failures"] += 1
                        raise FetchError(url, "site returned a browser-verification challenge page")
                    logger.debug("GET %s -> 200 (%d bytes)", url, len(resp.content))
                    return resp.content
                if resp.status_code == 200:
                    last_reason = "empty response body"
                elif resp.status_code in _RETRY_STATUS:
                    last_reason = f"HTTP {resp.status_code}"
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                else:
                    self.stats["failures"] += 1
                    raise FetchError(url, f"HTTP {resp.status_code}")

            if attempt < self.max_attempts:
                delay = retry_after if retry_after is not None else (
                    self.backoff_base * 2 ** (attempt - 1) + random.uniform(0, 0.5)
                )
                logger.warning("GET %s failed (%s), attempt %d/%d, retrying in %.1fs",
                               url, last_reason, attempt, self.max_attempts, delay)
                self._sleep(delay)

        self.stats["failures"] += 1
        raise FetchError(url, f"gave up after {self.max_attempts} attempts ({last_reason})")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


_client: HttpClient | None = None
_client_lock = threading.Lock()


def get_client() -> HttpClient:
    """Process-wide client, so the rate limit is shared by every caller."""
    global _client
    with _client_lock:
        if _client is None:
            cfg = load_config()
            h = cfg["http"]
            _client = HttpClient(
                cache_dir=resolve_path(cfg["paths"]["html_cache_dir"]),
                user_agent=h["user_agent"],
                min_interval=h["min_interval_sec"],
                timeout=h["timeout_sec"],
                max_attempts=h["max_attempts"],
                backoff_base=h["backoff_base_sec"],
                host_intervals=h.get("host_min_interval_sec"),
            )
        return _client
