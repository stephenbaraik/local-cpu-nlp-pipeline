# Local CPU NLP Pipeline

Staged CPU/GPU NLP pipeline that turns cybersecurity/AI-security web-capture
PDFs into five keyphrases, a topic label, and a one-sentence summary. Built
for the "Technical Assessment: Local CPU NLP Pipeline" brief, and benchmarked
against Denzel's single-call LLM pipeline (`denzel code/`).

## Install

**One command (Windows):** `.\setup.ps1` -- creates the venv, installs, downloads
the Gemma model, and runs a 1-document smoke test. Safe to re-run; it skips
whatever's already done. It does not set up GPU support -- see below.

**Manual, or if you're not on Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Requires Python 3.11+. All dependencies and pinned versions are in
`pyproject.toml`.

**GPU (optional).** `pip install -e .` installs the plain PyPI build of
torch, which is **CPU-only** -- `--device cuda` will silently fall back to
CPU with no error (this cost real time in this project once already). If
you have an NVIDIA GPU and want to use it, reinstall torch for your CUDA
version after the step above, e.g. for CUDA 12.6:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

Then verify it actually took before trusting any `--device cuda` run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# must print True -- if it prints False, GPU runs will silently execute on CPU
```

Gemma runs through `llama-cpp-python` on the q4_0 QAT GGUF build. It's the
only model this project needs you to fetch manually into a specific
folder (~3.3GB, gitignored) -- everything else downloads itself on first
use. Pick the command for your OS, run it from the project root:

**macOS / Linux:**

```bash
mkdir -p models_gguf
curl -L -o models_gguf/gemma-4-E2B_q4_0-it.gguf \
  "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf"
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path models_gguf | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf" -OutFile "models_gguf\gemma-4-E2B_q4_0-it.gguf"
```

**Any OS, via Python** (resumable if it drops mid-download, unlike the raw
`curl`/`Invoke-WebRequest` above):

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('google/gemma-4-E2B-it-qat-q4_0-gguf', 'gemma-4-E2B_q4_0-it.gguf', local_dir='models_gguf')
"
```

SecureBERT and ModernBERT (both torch weights and ModernBERT's published
int8 ONNX build) download automatically from Hugging Face on first use,
cached under `~/.cache/huggingface` -- no folder to point at, no command
to run. `models_onnx/securebert.onnx` is the one exception, and it isn't
downloaded at all -- see the ONNX backend step below.

For the ONNX backend, export SecureBERT once (fp32 -- see
`src/pipeline/onnx_export.py` for why not int8):

```bash
python -m pipeline.onnx_export
```

## Verify your setup

Before the full run, confirm the install actually works with one document
(seconds, not minutes) -- same flags as the real run below, just one PDF:

```bash
python -m pipeline run --match-denzel --protocol A --doc 035f --force
```

If that prints a `run_id:` line with no errors, the install is good. If it
fails, it's almost always one of: the Gemma GGUF file missing from
`models_gguf/` (see Install above), or a Hugging Face download blocked by a
firewall/proxy (SecureBERT and ModernBERT download on first use).

## Run

Every command below runs the **Denzel-matched Protocol A pipeline** --
`--match-denzel` reproduces Denzel's page-1-only input, `--protocol A` is
their single-pass, no-warm-up timing method. This is the only
configuration this README asks you to run; it's what the results in
`reports/` are built from.

```bash
python -m pipeline run --match-denzel --protocol A --force
python -m pipeline report                        # writes runs/<run_id>/results.json
python -m pipeline metrics                        # writes the 3-layer metrics.json + documents.csv / stages.csv
```

`--force` recomputes everything instead of serving a cached artifact --
a benchmark that times a cache hit is a meaningless number.

For a GPU run under the same matched configuration (see the GPU install
step above first):

```bash
python -m pipeline run --match-denzel --protocol A --force --device cuda --dtype fp16
python -m pipeline report
python -m pipeline metrics
```

Put PDFs in `pdfs/` (already populated with the assessment's 15-document
corpus).

## Test

```bash
pytest tests -m "not slow"    # fast suite, no models, ~10s
pytest tests                  # full suite, real models, several minutes
```

## Results

`python -m pipeline report` (after the `run` above) writes the assessment's
required results file to `runs/<run_id>/results.json`: per document,
accepted/rejected status, rejection reason, top-5 keyphrases, predicted
label + confidence, generated summary, and the run's full config.

`python -m pipeline metrics` writes the three-layer `metrics.json` plus
`documents.csv` / `stages.csv` -- the same files behind the CSVs already in
`reports/cpu/` and `reports/gpu/`. Generated reports (CSVs, the benchmark
presentation) live under `reports/`.
