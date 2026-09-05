# One-shot setup: venv, install, model download, smoke test.
# Run from the project root: .\setup.ps1
#
# What "activate, install, run" actually requires on this project:
#   1. venv + activate
#   2. pip install -e .            (does NOT fetch the Gemma model -- pip can't pull GGUF weights)
#   3. download the Gemma GGUF     (3.3GB, one-time, required -- summarize fails without it)
#   4. run a 1-document smoke test (proves the install actually works, ~seconds not minutes)
#
# GPU is intentionally NOT set up here -- see README.md "GPU (optional)".
# pip installs CPU-only torch; that's correct for step 2 above.

$ErrorActionPreference = "Stop"

Write-Host "== Local CPU NLP Pipeline setup ==" -ForegroundColor Cyan

# 1. venv
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/4] Creating .venv ..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "`n[1/4] .venv already exists, skipping" -ForegroundColor Yellow
}

$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "venv creation didn't produce $venvPython -- check the python -m venv output above"
}

# 2. install
Write-Host "`n[2/4] pip install -e . (this is CPU-only torch by design -- see README for GPU)" -ForegroundColor Yellow
& $venvPython -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed, exit code $LASTEXITCODE" }

# 3. Gemma GGUF model
$modelPath = "models_gguf\gemma-4-E2B_q4_0-it.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "`n[3/4] Downloading Gemma GGUF (~3.3GB, one-time) ..." -ForegroundColor Yellow
    & $venvPython -c @"
from huggingface_hub import hf_hub_download
hf_hub_download('google/gemma-4-E2B-it-qat-q4_0-gguf', 'gemma-4-E2B_q4_0-it.gguf', local_dir='models_gguf')
"@
    if ($LASTEXITCODE -ne 0) { throw "Gemma model download failed, exit code $LASTEXITCODE" }
} else {
    Write-Host "`n[3/4] Gemma GGUF already present, skipping download" -ForegroundColor Yellow
}

# 4. smoke test -- one document, proves the install works end to end
Write-Host "`n[4/4] Running a 1-document smoke test ..." -ForegroundColor Yellow
$firstPdf = Get-ChildItem -Path "pdfs" -Filter "*.pdf" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $firstPdf) {
    Write-Host "No PDFs found in pdfs/ -- put at least one .pdf there before running the pipeline." -ForegroundColor Red
} else {
    # --doc matches a doc_id prefix OR a filename substring -- the filename
    # itself always exists, unlike a guessed content-hash prefix.
    & $venvPython -m pipeline run --doc $firstPdf.Name --force 2>&1 | Select-Object -Last 12
}

Write-Host "`n== Setup complete ==" -ForegroundColor Cyan
Write-Host "Activate the venv in your own shell before running commands directly:"
Write-Host "  .venv\Scripts\activate"
Write-Host "Then run the full corpus:"
Write-Host "  python -m pipeline run --force"
Write-Host "  python -m pipeline report"
Write-Host "See README.md for benchmarking, GPU setup, and CLI flags."
