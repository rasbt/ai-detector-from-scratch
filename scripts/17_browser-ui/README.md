&nbsp;
# Local browser UI

[← Stage 16: Substack case study](../16_case-study-substack/) · [Project index](../../README.md) · [Stage 18: Reinforcement learning →](../18_reinforcement-learning/)

This folder contains a React/Vite interface and a FastAPI backend for the local
AI-text classifiers. One model is selected when the app starts, loaded once,
and kept in memory for subsequent checks.

Run all commands from the project directory.

Install the backend dependencies once:

```bash
uv sync --group browser-ui
```

When using a neural detector, include the transformer inference group:

```bash
uv sync --group browser-ui --group transformer-inference
```

&nbsp;
## One-time frontend setup

Use the committed lockfile and disable dependency lifecycle scripts:

```bash
cd scripts/17_browser-ui/frontend
npm ci --ignore-scripts --audit=false --fund=false
npm run build
cd ../../..
```

If the build reports that esbuild was not installed correctly, explicitly run
its installation step and build again:

```bash
cd scripts/17_browser-ui/frontend
npm rebuild esbuild
npm run build
cd ../../..
```

&nbsp;
## Check the available models

```bash
uv run python scripts/15_classifier-api/classify.py --list-models
```

The selected model must be marked ready under the top-level `models/`
directory. Download a missing model with:

```bash
uv run python scripts/15_classifier-api/download-models.py \
  --fetch \
  --model MODEL_NAME
```

&nbsp;
## Start the app

Logistic regression is the default:

```bash
uv run python scripts/17_browser-ui/app.py
```

Select another model at launch with `--model`. For example:

```bash
uv run python scripts/15_classifier-api/download-models.py \
  --fetch \
  --model qwen3-variable
```

Then start the app:

```bash
uv run python scripts/17_browser-ui/app.py \
  --model qwen3-variable \
  --device mps
```

Alternatively, explicitly fetch the selected model when it is missing:

```bash
uv run python scripts/17_browser-ui/app.py \
  --model qwen3-variable \
  --device mps \
  --fetch-missing
```

On a CUDA machine:

```bash
uv run python scripts/17_browser-ui/app.py \
  --model qwen3-variable \
  --device cuda
```

The selected model remains in memory until the server stops. The UI displays
the active model reported by the backend. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000).

&nbsp;
## Chunk analysis

Leave the chunk-size field empty to analyze the submitted text once. Enter a
size such as 100 to divide the text into non-overlapping 100-token chunks. The
result view highlights AI-leaning chunks in orange and human-leaning chunks in
green.

Transformer models use their saved tokenizer to determine the boundaries.
Logistic regression uses whitespace-delimited tokens because it has no model
tokenizer. The combined score is the token-count-weighted average of the chunk
scores. The last chunk can be shorter than the selected size.

Each transformer model reports its maximum supported chunk size to the
interface. Logistic regression has no fixed model limit. These limits do not
affect the default whole-text option, where each classifier's normal
truncation behavior still applies.

&nbsp;
## Frontend development

Start the backend in one terminal. Add `--model` and `--device` when needed:

```bash
uv run python scripts/17_browser-ui/app.py --model logreg
```

Start Vite in a second terminal:

```bash
cd scripts/17_browser-ui/frontend
npm run dev
```

Vite proxies `/api` requests to FastAPI at `http://127.0.0.1:8000`.
