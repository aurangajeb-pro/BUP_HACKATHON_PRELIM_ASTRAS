"""Readiness must match the configured model exactly and the official success body."""
import app.main as main


def test_health_ready(client, monkeypatch):
    monkeypatch.setattr(main, "_ollama_status", lambda: True)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_unavailable(client, monkeypatch):
    monkeypatch.setattr(main, "_ollama_status", lambda: False)
    assert client.get("/health").status_code == 503
    assert client.get("/healthz").json() == {"status": "ok"}


def test_health_requires_exact_model(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(main, "OLLAMA_MODEL", "example:large")
    monkeypatch.setattr(main.requests, "get", lambda *a, **kw: Mock(ok=True, json=lambda: {
        "models": [{"name": "example:small"}]
    }))
    assert main._ollama_status() is False
