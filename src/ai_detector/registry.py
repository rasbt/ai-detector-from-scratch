"""Model registry and artifact validation."""

from collections import namedtuple
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ModelSpec = namedtuple("ModelSpec", "name kind artifact_path description")


class ArtifactNotReadyError(RuntimeError):
    """Raised when a model export is missing or incomplete."""


def model_registry(project_dir=PROJECT_DIR):
    models_dir = project_dir / "models"
    specs = [
        ModelSpec(
            name="logreg",
            kind="sklearn",
            artifact_path=models_dir / "logreg" / "logreg-ai-detector.joblib",
            description="TF-IDF logistic regression with Platt scaling",
        ),
        ModelSpec(
            name="distilbert",
            kind="encoder",
            artifact_path=models_dir / "distilbert",
            description="Fully fine-tuned DistilBERT",
        ),
        ModelSpec(
            name="distilbert-lora",
            kind="peft",
            artifact_path=models_dir / "distilbert-lora",
            description="DistilBERT with a LoRA adapter",
        ),
        ModelSpec(
            name="distilbert-mica",
            kind="peft",
            artifact_path=models_dir / "distilbert-mica",
            description="DistilBERT with a MiCA adapter",
        ),
        ModelSpec(
            name="modernbert",
            kind="encoder",
            artifact_path=models_dir / "modernbert",
            description="Fully fine-tuned ModernBERT-base",
        ),
        ModelSpec(
            name="gpt2-variable",
            kind="causal",
            artifact_path=models_dir / "gpt2-variable",
            description="GPT-2 with a variable-position readout",
        ),
        ModelSpec(
            name="gpt2-fixed",
            kind="causal",
            artifact_path=models_dir / "gpt2-fixed",
            description="GPT-2 with a fixed-position readout",
        ),
        ModelSpec(
            name="qwen3-variable",
            kind="causal",
            artifact_path=models_dir / "qwen3-variable",
            description="Qwen3-0.6B with a variable-position readout",
        ),
        ModelSpec(
            name="qwen3-fixed",
            kind="causal",
            artifact_path=models_dir / "qwen3-fixed",
            description="Qwen3-0.6B with a fixed-position readout",
        ),
    ]
    return {spec.name: spec for spec in specs}


def _has_model_weights(spec):
    if spec.kind == "sklearn":
        return spec.artifact_path.is_file()

    if spec.kind == "peft":
        patterns = ("adapter_model*.safetensors", "adapter_model*.bin")
    else:
        patterns = (
            "model*.safetensors",
            "pytorch_model*.bin",
            "model*.index.json",
            "pytorch_model*.index.json",
        )
    return any(
        any(spec.artifact_path.glob(pattern)) for pattern in patterns
    )


def artifact_status(spec):
    if spec.kind == "sklearn":
        if spec.artifact_path.is_file():
            return True, "ready"
        return False, "model file is missing"

    if not spec.artifact_path.is_dir():
        return False, "artifact directory is missing"
    if not (spec.artifact_path / "detector-config.json").is_file():
        return False, "detector-config.json is missing"
    if not _has_model_weights(spec):
        return False, "model weights are missing"
    return True, "ready"


def ensure_artifact_ready(spec):
    ready, status = artifact_status(spec)
    if ready:
        return
    raise ArtifactNotReadyError(
        f"{spec.name} is not ready: {status}. Expected export at "
        f"{spec.artifact_path}. Run `uv run python "
        "scripts/15_classifier-api/download-models.py --fetch --model "
        f"{spec.name}`, or pass "
        "--artifact with a complete export."
    )
