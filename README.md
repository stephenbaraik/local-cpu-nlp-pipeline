# Local CPU NLP Pipeline

Staged CPU/GPU NLP pipeline that turns cybersecurity/AI-security web-capture
PDFs into five keyphrases, a topic label, and a one-sentence summary. Built
for the "Technical Assessment: Local CPU NLP Pipeline" brief, and benchmarked
against Denzel's single-call LLM pipeline (`denzel code/`).

Docs:
- [`CLAUDE.md`](CLAUDE.md) -- enforceable project rules (hardware, benchmark
  protocol, non-negotiables).
- [`Build_guide.md`](Build_guide.md) -- the reasoning behind those rules.
- [`Reporting.md`](Reporting.md) -- how results are captured and written up.
- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) -- the write-up
  required by the brief (cleaning strategy, model integration, long-document
  handling, untrusted-content handling, optimizations, trade-offs).

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

Before a full run, confirm the install actually works with one document
(seconds, not minutes):

```bash
python -m pipeline run --doc 035f --force
```

If that prints a `run_id:` line with no errors, the install is good. If it
fails, it's almost always one of: the Gemma GGUF file missing from
`models_gguf/` (see Install above), or a Hugging Face download blocked by a
firewall/proxy (SecureBERT and ModernBERT download on first use).

## Run

```bash
python -m pipeline run                          # all stages, all PDFs in pdfs/, cached
python -m pipeline run --force                   # ignore cache, recompute everything
python -m pipeline run --only clean              # force-recompute clean + everything downstream
python -m pipeline run --through keywords        # stop after stage 4
python -m pipeline run --doc 035f --force        # one document, ignore cache

python -m pipeline run --device cuda --dtype fp16   # GPU encoders, fp16
python -m pipeline run --match-denzel --protocol A  # page-1 gate, matches Denzel's timing method

python -m pipeline report                        # artifacts/ -> runs/<run_id>/results.json
python -m pipeline bench                         # per-model timing, cache always off
python -m pipeline bench --grid                  # thread x worker peak-RSS grid
python -m pipeline metrics                       # writes the 3-layer run/documents/stages metrics.json + CSVs
python -m pipeline compare --modes full,page1
python -m pipeline compare --modes guard-on,guard-off
python -m pipeline compare --modes torch,onnx
```

Put PDFs in `pdfs/` (already populated with the assessment's 15-document
corpus). Config is defaults plus `NLP_*` environment variables -- see
`src/pipeline/config.py` for the full list (`NLP_MAX_PAGES`,
`NLP_INJECTION_GUARD`, `NLP_BACKEND`, `NLP_DEVICE`, `NLP_DTYPE`,
`NLP_CPU_THREADS`, `NLP_BATCH_SIZE`, ...). Hardware flags on `run` and `bench`
(`--device`, `--dtype`, `--threads`, `--batch-size`, `--protocol`,
`--taxonomy`, `--match-denzel`, `--injection-check`) set the matching env var
before the run starts -- see `src/pipeline/__main__.py`.

## Test

```bash
pytest tests -m "not slow"    # fast suite, no models, ~10s
pytest tests                  # full suite, real models, several minutes
```

## Results

`python -m pipeline run --force && python -m pipeline report` writes the
assessment's required results file to `runs/<run_id>/results.json`: per
document, accepted/rejected status, rejection reason, top-5 keyphrases,
predicted label + confidence, generated summary, and the run's full config.

`python -m pipeline bench` writes `runs/<run_id>/benchmark.json` (init vs.
inference timing, peak RSS, environment + library versions).
`python -m pipeline metrics` writes the three-layer `metrics.json` plus
`documents.csv` / `stages.csv` used for the CPU/GPU benchmark comparison.
Generated reports (CSVs, the benchmark presentation) live under `reports/`.
