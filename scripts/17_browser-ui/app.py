#!/usr/bin/env python3
"""Serve the local AI-detector API and its built React interface."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import re
import subprocess
import sys

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import Field, ValidationError, create_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
FRONTEND_DIST = SCRIPT_DIR / "frontend" / "dist"
MIN_WORDS = 50
MIN_CHUNK_SIZE = 10

from ai_detector import (
    artifact_status,
    load_classifier,
    model_registry,
    score_payload,
)


CheckRequest = create_model(
    "CheckRequest",
    text=(str, Field(min_length=1)),
    chunk_size=(
        int | None,
        Field(default=None, ge=MIN_CHUNK_SIZE),
    ),
)
ChunkResponse = create_model(
    "ChunkResponse",
    text=(str, ...),
    score=(float, ...),
    label=(str, ...),
    token_count=(int, ...),
)
CheckResponse = create_model(
    "CheckResponse",
    score=(float, ...),
    label=(str, ...),
    word_count=(int, ...),
    chunk_size=(int | None, None),
    chunks=(list[ChunkResponse] | None, None),
)


def count_words(text):
    return len(re.findall(r"\S+", text.strip()))


def fetch_missing_model(
    model_name,
    *,
    project_dir=PROJECT_DIR,
    run_command=None,
):
    spec = model_registry(project_dir)[model_name]
    ready, _ = artifact_status(spec)
    if ready:
        return

    if run_command is None:
        run_command = subprocess.run
    command = [
        sys.executable,
        str(
            project_dir
            / "scripts"
            / "15_classifier-api"
            / "download-models.py"
        ),
        "--fetch",
        "--model",
        model_name,
    ]
    result = run_command(command)
    if result.returncode:
        raise SystemExit(result.returncode)


def classifier_chunk_limit(classifier):
    max_text_length = getattr(classifier, "max_text_length", None)
    if max_text_length is not None:
        return int(max_text_length)

    max_length = getattr(classifier, "max_length", None)
    if max_length is None:
        return None

    tokenizer = getattr(classifier, "tokenizer", None)
    special_tokens = 0
    if tokenizer is not None:
        count_special_tokens = getattr(
            tokenizer,
            "num_special_tokens_to_add",
            None,
        )
        if count_special_tokens is not None:
            special_tokens = int(count_special_tokens(pair=False))
    return max(1, int(max_length) - special_tokens)


def token_offsets(
    classifier,
    text,
):
    tokenizer = getattr(classifier, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = [
                (int(start), int(end))
                for start, end in encoded["offset_mapping"]
                if int(end) > int(start)
            ]
            if offsets:
                return offsets
        except (KeyError, NotImplementedError, TypeError, ValueError):
            pass

    return [match.span() for match in re.finditer(r"\S+", text)]


def split_text_into_token_chunks(
    classifier,
    text,
    chunk_size,
):
    offsets = token_offsets(classifier, text)
    chunks = []
    for token_start in range(0, len(offsets), chunk_size):
        token_end = min(token_start + chunk_size, len(offsets))
        char_start = 0 if token_start == 0 else offsets[token_start][0]
        char_end = (
            len(text)
            if token_end == len(offsets)
            else offsets[token_end][0]
        )
        chunk_text = text[char_start:char_end]
        if chunk_text.strip():
            chunks.append((chunk_text, token_end - token_start))
    return chunks


def create_app(
    classifier = None,
    *,
    model_name = "logreg",
    artifact_path = None,
    device = "auto",
    frontend_dist = FRONTEND_DIST,
):
    registry = model_registry()
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    model_label = registry[model_name].description

    @asynccontextmanager
    async def lifespan(app):
        if classifier is None:
            app.state.classifier = load_classifier(
                model_name,
                artifact_path=artifact_path,
                device=device,
            )
        else:
            app.state.classifier = classifier
        yield

    application = FastAPI(
        title="Local AI Detector",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "model_label": model_label,
            "max_chunk_size": classifier_chunk_limit(
                application.state.classifier
            ),
        }

    @application.post(
        "/api/check",
        response_model=CheckResponse,
        response_model_exclude_none=True,
    )
    def check_text(payload=Body(...)):
        try:
            payload = CheckRequest(**payload)
        except (TypeError, ValidationError) as error:
            detail = error.errors() if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        text = payload.text.strip()
        word_count = count_words(text)
        if word_count < MIN_WORDS:
            raise HTTPException(
                status_code=422,
                detail=f"Please enter at least {MIN_WORDS} words.",
            )

        active_classifier = application.state.classifier
        if payload.chunk_size is None:
            probability = active_classifier.score_many(
                [text], batch_size=1
            )[0]
            score = score_payload(probability)["score"]
            chunks = None
        else:
            chunk_limit = classifier_chunk_limit(active_classifier)
            if chunk_limit is not None and payload.chunk_size > chunk_limit:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"The selected model supports at most {chunk_limit} "
                        "tokens per chunk."
                    ),
                )
            chunk_pairs = split_text_into_token_chunks(
                active_classifier,
                text,
                payload.chunk_size,
            )
            chunk_texts = [chunk_text for chunk_text, _ in chunk_pairs]
            batch_size = 1 if registry[model_name].kind == "causal" else 8
            probabilities = active_classifier.score_many(
                chunk_texts,
                batch_size=batch_size,
            )
            token_total = sum(token_count for _, token_count in chunk_pairs)
            probability = sum(
                chunk_probability * token_count
                for chunk_probability, (_, token_count) in zip(
                    probabilities,
                    chunk_pairs,
                    strict=True,
                )
            ) / token_total
            score = score_payload(probability)["score"]
            chunks = [
                ChunkResponse(
                    text=chunk_text,
                    score=score_payload(chunk_probability)["score"],
                    label=(
                        "AI-generated"
                        if chunk_probability >= 0.5
                        else "human-written"
                    ),
                    token_count=token_count,
                )
                for chunk_probability, (chunk_text, token_count) in zip(
                    probabilities,
                    chunk_pairs,
                    strict=True,
                )
            ]

        label = "AI-generated" if score >= 50.0 else "human-written"
        return CheckResponse(
            score=score,
            label=label,
            word_count=word_count,
            chunk_size=payload.chunk_size,
            chunks=chunks,
        )

    if (frontend_dist / "index.html").is_file():
        application.mount(
            "/",
            StaticFiles(directory=frontend_dist, html=True),
            name="frontend",
        )
    else:
        @application.get("/{requested_path:path}", include_in_schema=False)
        def frontend_not_built(requested_path):
            del requested_path
            raise HTTPException(
                status_code=503,
                detail=(
                    "The frontend has not been built. Run "
                    "`npm ci --ignore-scripts` and `npm run build` in "
                    "scripts/17_browser-ui/frontend."
                ),
            )

    return application


app = create_app()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Serve the browser UI with one classifier kept in memory.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(model_registry()),
        default="logreg",
        help="Classifier to load when the app starts.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Override the selected model's default artifact path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Download the selected model from the Hugging Face Hub if needed.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.fetch_missing and args.artifact is not None:
        parser.error("--fetch-missing cannot be combined with --artifact")
    return args


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.fetch_missing:
        fetch_missing_model(args.model)
    selected_app = create_app(
        model_name=args.model,
        artifact_path=args.artifact,
        device=args.device,
    )
    uvicorn.run(selected_app, host=args.host, port=args.port)
