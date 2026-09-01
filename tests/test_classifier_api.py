
from pathlib import Path

import pytest

import ai_detector



def test_registry_contains_every_local_classifier_variant():
    registry = ai_detector.model_registry(Path("/project"))

    assert set(registry) == {
        "logreg",
        "distilbert",
        "distilbert-lora",
        "distilbert-mica",
        "modernbert",
        "gpt2-variable",
        "gpt2-fixed",
        "qwen3-variable",
        "qwen3-fixed",
    }
    assert registry["logreg"].kind == "sklearn"
    assert registry["distilbert-lora"].kind == "peft"
    assert registry["distilbert-mica"].kind == "peft"
    assert registry["gpt2-fixed"].kind == "causal"
    assert registry["logreg"].artifact_path == Path(
        "/project/models/logreg/logreg-ai-detector.joblib"
    )
    for model_spec in registry.values():
        assert Path("/project/models") in model_spec.artifact_path.parents
        assert "/scripts/" not in str(model_spec.artifact_path)


def test_missing_artifact_error_points_to_hub_fetch(tmp_path):
    spec = ai_detector.model_registry(tmp_path)["distilbert"]

    with pytest.raises(
        ai_detector.ArtifactNotReadyError,
        match=(
            r"uv run python scripts/15_classifier-api/"
            r"download-models\.py --fetch --model distilbert"
        ),
    ):
        ai_detector.ensure_artifact_ready(spec)


def test_variable_causal_batch_places_readout_after_each_text():
    input_rows, mask_rows = ai_detector.build_causal_batch(
        [[5, 6], [7]],
        pad_token_id=0,
        eos_token_id=2,
        context_length=6,
        readout_position="variable",
    )

    assert input_rows == [[5, 6, 2], [7, 2, 0]]
    assert mask_rows == [[1, 1, 1], [1, 1, 0]]


def test_fixed_causal_batch_places_readout_at_final_position():
    input_rows, mask_rows = ai_detector.build_causal_batch(
        [[5, 6], [7]],
        pad_token_id=0,
        eos_token_id=2,
        context_length=6,
        readout_position="fixed",
    )

    assert input_rows == [
        [5, 6, 0, 0, 0, 2],
        [7, 0, 0, 0, 0, 2],
    ]
    assert mask_rows == [
        [1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
    ]


def test_causal_batch_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="readout_position"):
        ai_detector.build_causal_batch(
            [[5]],
            pad_token_id=0,
            eos_token_id=2,
            context_length=6,
            readout_position="middle",
        )
    with pytest.raises(ValueError, match="context length"):
        ai_detector.build_causal_batch(
            [[5, 6, 7, 8, 9, 10]],
            pad_token_id=0,
            eos_token_id=2,
            context_length=6,
            readout_position="fixed",
        )


def test_score_payload_uses_zero_to_one_hundred_scale():
    assert ai_detector.score_payload(0.123456) == {"score": 12.3456}
    assert ai_detector.score_payload(1.0) == {"score": 100.0}
    with pytest.raises(ValueError, match="between 0 and 1"):
        ai_detector.score_payload(1.01)


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ai_detector.validate_text("  \n")
