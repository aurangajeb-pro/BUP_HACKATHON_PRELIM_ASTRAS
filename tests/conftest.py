"""Offline tests are reproducible; live interpretation is an explicit separate gate."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--run-llm-tests", action="store_true", default=False,
                     help="Run live Ollama interpretation tests (requires configured model)")


def pytest_configure(config):
    config.addinivalue_line("markers", "live_llm: requires real configured Ollama inference")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-llm-tests"):
        for item in items:
            if "live_llm" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-llm-tests to test real model interpretation"))


@pytest.fixture(scope="session")
def ollama_available():
    from app.main import _ollama_status
    if not _ollama_status():
        pytest.fail("Configured Ollama model is unavailable; live interpretation was not verified")
    return True


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_directives(monkeypatch):
    import app.llm as llm
    def respond(messages):
        notes = json.loads(messages[1]["content"])["operator_notes"]
        return json.dumps({"interpretations": [
            {"note_index": n["note_index"], "applies": False, "directive_type": "no_op",
             "structured_adjustment": None, "explanation": "Test-only irrelevant note."}
            for n in notes
        ]})
    monkeypatch.setattr(llm, "_call_ollama", respond)
