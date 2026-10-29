"""Guard availability tracking, /metrics gauge and /health session_store (#4512).

Covers:
* ``Guard.available`` flips to False on a connection error (backoff) and back
  to True after a successful call.
* ``headroom_redis_available{feature=...}`` gauge appears in /metrics output.
* ``/health`` reports ``session_store`` as degraded while top-level ``ready``
  stays True.
"""

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from headroom.cache import redis_client  # noqa: E402
from headroom.cache.redis_client import Guard, guards  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_guards():
    """Ensure no guard registrations leak between tests."""
    redis_client._GUARDS.clear()
    yield
    redis_client._GUARDS.clear()


# --- Guard.available ---------------------------------------------------------


class TestGuardAvailable:
    def test_available_initially(self):
        g = Guard("test feature", "test")
        assert g.available is True

    def test_unavailable_after_connection_error(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        g = Guard("test feature", "test")
        import redis

        def failing():
            raise redis.ConnectionError("connection refused")

        result = g.call("read", failing, "default")
        assert result == "default"
        assert g.available is False

    def test_available_again_after_successful_call(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 0.0)
        g = Guard("test feature", "test")
        import redis

        # First: fail with a non-connection error to populate _failing
        def failing():
            raise redis.RedisError("some error")

        g.call("read", failing, "default")
        assert g.available is False  # _failing is non-empty

        # Now succeed
        result = g.call("read", lambda: "ok", "default")
        assert result == "ok"
        assert g.available is True

    def test_available_false_during_backoff_even_without_failing(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        g = Guard("test feature", "test")
        import redis

        def failing():
            raise redis.ConnectionError("timeout")

        g.call("write", failing, None)
        # _failing has "write" AND _skip_until is in the future
        assert g.available is False

    def test_guards_registry(self):
        g1 = Guard("feature A", "alpha")
        g2 = Guard("feature B", "beta")
        snapshot = guards()
        assert snapshot == {"alpha": g1, "beta": g2}

    def test_guards_last_wins(self):
        Guard("feature A", "same")
        g2 = Guard("feature B", "same")
        snapshot = guards()
        assert snapshot["same"] is g2


# --- /metrics gauge ----------------------------------------------------------


class TestMetricsGauge:
    @pytest.mark.asyncio
    async def test_gauge_present_when_guards_exist(self):
        Guard("test feature", "test_label")
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert "# HELP headroom_redis_available" in text
        assert "# TYPE headroom_redis_available gauge" in text
        assert 'headroom_redis_available{feature="test_label"} 1' in text

    @pytest.mark.asyncio
    async def test_gauge_zero_during_outage(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        import redis

        g = Guard("test feature", "outage_label")

        def failing():
            raise redis.ConnectionError("down")

        g.call("read", failing, None)

        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert 'headroom_redis_available{feature="outage_label"} 0' in text

    @pytest.mark.asyncio
    async def test_no_gauge_when_no_guards(self):
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert "headroom_redis_available" not in text


# --- /health session_store ---------------------------------------------------


class TestHealthSessionStore:
    def test_session_store_degraded_does_not_flip_ready(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        import redis
        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        # Register a guard that will be unavailable
        g = Guard("compress sessions", "compress_session")

        def failing():
            raise redis.ConnectionError("down")

        g.call("read", failing, None)
        assert g.available is False

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        # Top-level ready must remain True despite degraded session_store
        assert payload["ready"] is True
        assert payload["status"] == "healthy"

        # session_store shows degraded
        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is True
        assert ss["ready"] is False
        assert ss["status"] == "degraded"
        assert ss["optional"] is True
        assert ss["features"]["compress_session"] is False

    def test_session_store_disabled_when_no_guards(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is False
        assert ss["status"] == "disabled"

    def test_session_store_healthy_when_available(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        g = Guard("CCR store", "ccr")
        assert g.available is True

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is True
        assert ss["ready"] is True
        assert ss["status"] == "healthy"
        assert ss["features"]["ccr"] is True
