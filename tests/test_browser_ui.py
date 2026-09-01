
import importlib.util
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "17_browser-ui"
    / "app.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("browser_ui_app", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClassifier:
    def __init__(self, probability):
        self.probability = probability
        self.received_texts = []

    def score_many(self, texts, *, batch_size=8):
        assert batch_size == 1
        self.received_texts.extend(texts)
        return [self.probability for _ in texts]


class ChunkClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.received_texts = []

    def score_many(self, texts, *, batch_size=8):
        assert batch_size == 8
        self.received_texts.extend(texts)
        return self.probabilities[: len(texts)]


def test_check_returns_ai_score_and_word_count(tmp_path):
    module = load_module()
    classifier = FakeClassifier(0.87341)
    app = module.create_app(
        classifier,
        model_name="qwen3-variable",
        frontend_dist=tmp_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/check",
            json={"text": "word " * module.MIN_WORDS},
        )
        health = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "score": 87.341,
        "label": "AI-generated",
        "word_count": module.MIN_WORDS,
    }
    assert health.json() == {
        "status": "ok",
        "model": "qwen3-variable",
        "model_label": "Qwen3-0.6B with a variable-position readout",
        "max_chunk_size": None,
    }
    assert classifier.received_texts == [("word " * module.MIN_WORDS).strip()]


def test_selected_model_loads_once_at_startup(monkeypatch, tmp_path):
    module = load_module()
    classifier = FakeClassifier(0.25)
    load_calls = []

    def fake_load_classifier(model_name, *, artifact_path, device):
        load_calls.append((model_name, artifact_path, device))
        return classifier

    monkeypatch.setattr(module, "load_classifier", fake_load_classifier)
    app = module.create_app(
        model_name="distilbert",
        device="cpu",
        frontend_dist=tmp_path,
    )

    with TestClient(app) as client:
        client.post("/api/check", json={"text": "word " * module.MIN_WORDS})
        client.post("/api/check", json={"text": "word " * module.MIN_WORDS})

    assert load_calls == [("distilbert", None, "cpu")]


def test_fetch_missing_model_runs_the_existing_downloader(tmp_path):
    module = load_module()
    calls = []

    class Result:
        returncode = 0

    def fake_run(command):
        calls.append(command)
        return Result()

    module.fetch_missing_model(
        "qwen3-variable",
        project_dir=tmp_path,
        run_command=fake_run,
    )

    assert calls == [[
        sys.executable,
        str(
            tmp_path
            / "scripts"
            / "15_classifier-api"
            / "download-models.py"
        ),
        "--fetch",
        "--model",
        "qwen3-variable",
    ]]


def test_fetch_missing_model_avoids_network_when_artifact_is_ready(
    tmp_path,
):
    module = load_module()
    artifact = tmp_path / "models" / "qwen3-variable"
    artifact.mkdir(parents=True)
    (artifact / "detector-config.json").write_text(
        "{}", encoding="utf-8"
    )
    (artifact / "model.safetensors").write_bytes(b"weights")

    def fail_if_called(command):
        pytest.fail(f"Unexpected download command: {command}")

    module.fetch_missing_model(
        "qwen3-variable",
        project_dir=tmp_path,
        run_command=fail_if_called,
    )


def test_chunk_mode_scores_non_overlapping_chunks(tmp_path):
    module = load_module()
    classifier = ChunkClassifier([0.9, 0.2, 0.1])
    app = module.create_app(
        classifier,
        model_name="logreg",
        frontend_dist=tmp_path,
    )
    text = " ".join(f"word{index}" for index in range(50))

    with TestClient(app) as client:
        response = client.post(
            "/api/check",
            json={"text": text, "chunk_size": 20},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == 46.0
    assert payload["label"] == "human-written"
    assert payload["word_count"] == 50
    assert payload["chunk_size"] == 20
    assert [chunk["score"] for chunk in payload["chunks"]] == [
        90.0,
        20.0,
        10.0,
    ]
    assert [chunk["token_count"] for chunk in payload["chunks"]] == [
        20,
        20,
        10,
    ]
    assert "".join(chunk["text"] for chunk in payload["chunks"]) == text
    assert classifier.received_texts == [
        chunk["text"] for chunk in payload["chunks"]
    ]


def test_chunk_size_cannot_exceed_model_limit(tmp_path):
    module = load_module()
    classifier = FakeClassifier(0.5)
    classifier.max_text_length = 64
    app = module.create_app(
        classifier,
        model_name="qwen3-variable",
        frontend_dist=tmp_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/check",
            json={
                "text": "word " * module.MIN_WORDS,
                "chunk_size": 100,
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "The selected model supports at most 64 tokens per chunk."
    )


def test_chunk_boundaries_use_the_model_tokenizer():
    module = load_module()

    class CharacterTokenizer:
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return {
                "offset_mapping": [
                    (index, index + 1)
                    for index, character in enumerate(text)
                    if not character.isspace()
                ]
            }

    classifier = FakeClassifier(0.5)
    classifier.tokenizer = CharacterTokenizer()
    chunks = module.split_text_into_token_chunks(
        classifier,
        "ab cd",
        chunk_size=3,
    )

    assert chunks == [("ab c", 3), ("d", 1)]


def test_check_rejects_text_shorter_than_minimum(tmp_path):
    module = load_module()
    app = module.create_app(FakeClassifier(0.5), frontend_dist=tmp_path)

    with TestClient(app) as client:
        response = client.post("/api/check", json={"text": "too short"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Please enter at least 50 words."


def test_missing_frontend_returns_actionable_message(tmp_path):
    module = load_module()
    app = module.create_app(FakeClassifier(0.5), frontend_dist=tmp_path)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 503
    assert "frontend has not been built" in response.json()["detail"]
