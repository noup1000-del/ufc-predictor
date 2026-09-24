import pytest
import requests

from src.http import FetchError, HttpClient, RateLimiter


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeResponse:
    def __init__(self, status=200, content=b"<html>ok</html>", headers=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_client(tmp_path, responses, clock=None):
    clock = clock or FakeClock()
    session = FakeSession(responses)
    client = HttpClient(tmp_path, "test-agent", min_interval=1.0, timeout=15,
                        max_attempts=3, backoff_base=2, session=session,
                        sleep=clock.sleep, clock=clock.clock)
    return client, session, clock


def test_rate_limiter_spaces_calls():
    c = FakeClock()
    rl = RateLimiter(1.0, clock=c.clock, sleep=c.sleep)
    rl.wait()
    rl.wait()
    c.t += 0.3
    rl.wait()
    assert c.sleeps == pytest.approx([1.0, 0.7])


def test_per_host_interval_applies_only_to_that_host(tmp_path):
    clock = FakeClock()
    session = FakeSession([FakeResponse() for _ in range(4)])
    client = HttpClient(tmp_path, "test-agent", min_interval=1.0, session=session, sleep=clock.sleep,
                        clock=clock.clock, host_intervals={"www.ufc.com": 15})
    client.get("https://www.ufc.com/events")
    client.get("https://www.ufc.com/event/a")   # waits for the 15s crawl delay
    client.get("https://example.org/x")         # other hosts: only the global 1s limit (clock is frozen)
    client.get("https://example.org/y")
    assert clock.sleeps == pytest.approx([15.0, 1.0, 1.0])


def test_fetch_caches_and_reuses(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(content=b"<p>x</p>")])
    assert client.fetch("fight", "abc123", "http://x/abc123") == "<p>x</p>"
    assert (tmp_path / "fight" / "abc123.html").read_bytes() == b"<p>x</p>"
    # Second call must not hit the network (FakeSession would raise IndexError).
    assert client.fetch("fight", "abc123", "http://x/abc123") == "<p>x</p>"
    assert len(session.calls) == 1
    assert client.stats == {"network": 1, "cache_hits": 1, "failures": 0}


def test_refresh_refetches(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(content=b"a"), FakeResponse(content=b"b")])
    client.fetch("events_completed", "all", "http://x")
    assert client.fetch("events_completed", "all", "http://x", refresh=True) == "b"
    assert len(session.calls) == 2


def test_timeout_is_passed(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse()])
    client.fetch("event", "e1", "http://x/e1")
    assert session.calls[0][1] == 15


def test_retries_then_succeeds(tmp_path):
    client, session, clock = make_client(
        tmp_path, [requests.Timeout("t"), FakeResponse(503), FakeResponse(content=b"ok")])
    assert client.fetch("event", "e1", "http://x/e1") == "ok"
    assert len(session.calls) == 3
    backoffs = [s for s in clock.sleeps if s >= 2]
    assert 2 <= backoffs[0] < 2.5 and 4 <= backoffs[1] < 4.5


def test_gives_up_after_three_attempts_and_does_not_cache(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(500)] * 3)
    with pytest.raises(FetchError):
        client.fetch("event", "e1", "http://x/e1")
    assert len(session.calls) == 3
    assert not (tmp_path / "event" / "e1.html").exists()
    assert client.stats["failures"] == 1


def test_404_is_not_retried(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(404)])
    with pytest.raises(FetchError):
        client.fetch("event", "e1", "http://x/e1")
    assert len(session.calls) == 1


def test_retry_after_header_is_honoured(tmp_path):
    client, _, clock = make_client(
        tmp_path, [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse()])
    client.fetch("event", "e1", "http://x/e1")
    assert 7 in clock.sleeps


def test_bot_challenge_page_is_rejected_and_not_cached(tmp_path):
    page = b"<html><title>Loading</title><p>Checking your browser</p><script>xhr.open('POST',\"/__c\",true)</script>"
    client, session, _ = make_client(tmp_path, [FakeResponse(content=page)])
    with pytest.raises(FetchError, match="challenge"):
        client.fetch("event", "e1", "http://x/e1")
    assert len(session.calls) == 1
    assert not (tmp_path / "event" / "e1.html").exists()


def test_unsafe_cache_key_rejected(tmp_path):
    client, _, _ = make_client(tmp_path, [])
    with pytest.raises(ValueError):
        client.cache_path("fight", "../etc")
