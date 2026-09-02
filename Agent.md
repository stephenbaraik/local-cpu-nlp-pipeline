# AGENTS.md

Instructions for coding agents working in this repository. Read this before
touching anything. If something here contradicts a comment in the code, this
file wins and the comment is a bug.

## What this project is

A local CPU NLP pipeline that turns PDFs of cybersecurity web captures into
five keyphrases, a topic label, and a one sentence summary. It exists to answer
an engineering question for the team: is a multi-encoder pipeline faster than a
single small LLM doing the same job, and can cheap GPUs run the encoders.

It is an evaluation harness, not a service. There is no API, no database, no
users. Do not add any.

The full design is in `docs/staged-pipeline-design.md`. The scope and the
questions being answered are in `docs/cpu-nlp-pipeline-prd.md`.

## Architecture in one paragraph

Seven stages run in order: extract, clean, validate, keywords, classify,
context, summarize. Each writes a JSON artifact to `artifacts/<doc_id>/` and
skips work whose inputs have not changed. The runner iterates stages on the
outside and documents on the inside, so one model is loaded at a time and
released before the next. Configuration comes from defaults plus `NLP_*`
environment variables and is written into every results file.

## Commands

```bash
python -m pipeline run                       # all stages, all docs, cached
python -m pipeline run --only clean,validate # named stages only
python -m pipeline run --through keywords    # stages 1 to 4
python -m pipeline run --doc a3f0 --force    # one document, ignore cache
python -m pipeline report                    # artifacts -> runs/<id>/results.json
python -m pipeline bench --stage classify    # timing, cache always disabled
python -m pipeline compare --modes full,page1

pytest tests -m "not slow"                   # fast suite, no models, seconds
pytest tests                                 # includes model-backed tests
```

## Hard rules

**Never modify the pristine document text.** `clean` produces the canonical
body and everything downstream reads it without altering it. Candidate
generation works on segments derived from that text; the summarizer receives the
original. Any transformation that feeds one stage must not leak into another.

**Never cache a benchmark.** `bench` runs with caching disabled and there is no
flag to change that. Timing a cache hit is the one way to publish a badly wrong
number here.

**Declare every config key a stage reads.** Each stage lists `config_keys`, and
that list is what the cache fingerprint is built from. Reading a config value
without declaring it means the stage serves stale results silently after that
value changes. If you add a config read, add the key in the same commit.

**Bump `stage_version` when stage logic changes in a shared helper.** The code
fingerprint hashes the stage module only and does not follow imports. Editing
`segmentation.py` will not invalidate `keywords` on its own.

**Record the run configuration in every results file.** Mode, backend, device,
provider, max_pages, injection_guard, threads, workers, gguf path, stage
versions. There are eight axes now and a results file without its config is
unattributable within a week.

**One document failing never kills a run.** Catch per document, write an
artifact with `status: "error"` and the traceback, continue. Downstream stages
skip that document.

**No GPU or cloud inference in the default path.** CPU is the baseline the whole
evaluation rests on. GPU is opt-in via `NLP_DEVICE=cuda` and applies to encoders
only. Gemma stays on CPU.

## Known traps

These are real defects that were found the hard way. Do not reintroduce them.

**KeyBERT silently falls back to MiniLM.** `KeyBERT(model=some_hf_model)` does
not raise. `select_backend` accepts only a `BaseEmbedder` instance and quietly
downloads a default MiniLM otherwise, so SecureBERT is never used and the output
looks fine. Both the torch and ONNX embedders must subclass
`keybert.backend.BaseEmbedder`. This trap was hit twice in this project.

**SecureBERT has no trained pooler.** It is a ModernBERT masked LM checkpoint.
Use attention-masked mean pooling over `last_hidden_state`, then L2 normalise.
Raw `pooler_output` or the CLS token is not a sentence embedding here.

**Gemma 4 E2B is multimodal.** Load it with `AutoProcessor` and the multimodal
model class, not `AutoTokenizer` plus `AutoModelForCausalLM`. In practice the
default path is the q4_0 QAT GGUF through llama-cpp-python, and the transformers
path exists only as a baseline.

**Decode only new tokens.** Decoding the whole output tensor returns the prompt
as part of the summary, which also means any injected instruction lands in the
results file.

**Word-boundary matching in the sentence filter.** A substring test makes `ai`
match inside `said` and keeps most of the document. Use `\b...\b`.

**CountVectorizer bridges punctuation.** It treats newlines as whitespace, so a
single string yields bigrams like `county alameda` spanning a sentence
boundary. Pass an iterable of segments so each is an isolated document. Note the
residual limit: stop-words are removed before n-gramming, so `jailbreaking
paper` can still form inside one segment. That needs MMR, not segmentation.

**Python needs a fixed-width lookbehind.** The abbreviation guard in
`segmentation.py` is split into several same-length lookbehinds for this reason.
Merging them into one alternation raises at import time.

**FlashAttention 2 needs Ampere or newer.** ModernBERT reaches for it. On a
GTX 1080 Ti it falls back, so `NLP_ATTN_IMPL` defaults to `sdpa`. Also, fp16 on
Pascal is much slower than fp32, so int8 is the only real GPU speed path on that
card.

## Code style

Plain, direct Python. Type hints on public functions. `from __future__ import
annotations` at the top of every module.

Comments explain why, not what. A comment saying what the next line does is
noise. A comment saying which trap a line avoids is worth keeping.

No new dependencies without a reason written in the commit message. The
dependency set is already awkward: `optimum-onnx` pins `transformers<4.58`
while Gemma 4 needs `transformers>=5`, which is why ONNX export runs in a
separate virtual environment and inference uses raw `onnxruntime` with no
`optimum` import. Do not try to unify those environments.

Logging: INFO to console, DEBUG to `runs/<run_id>/pipeline.log`. If you find
yourself wanting a number from the debug log while reading results, that number
should be a field in the results file instead.

## Testing

Add a test with any behaviour change to stages 1, 2, 3 or 6. They are pure and
model-free, so there is no excuse.

The cache invalidation test is the most important one in the suite. It asserts
that changing a stage's declared config key invalidates that stage and
everything downstream, and leaves upstream and unrelated stages untouched. If
you change the fingerprinting logic, that test must still pass unmodified.

Model-backed tests are marked `slow` and assert shape and non-emptiness, not
exact values. Quantization moves the values, and a test that pins them will fail
for the wrong reason.

Fixtures live in `tests/fixtures/`. There must always be a Cloudflare
interstitial, a 403 page long enough to clear the word-count rule, a short but
genuine article, and a document containing a prompt injection.

## What not to do

- Do not add a web API, a database, a queue, or a Docker orchestration layer.
- Do not move Gemma to GPU. It confounds the question being measured.
- Do not delete or edit files under `artifacts/` by hand. Use `--force`.
- Do not change published defaults. They reproduce the numbers already in
  circulation. Add a flag instead.
- Do not tune thresholds to make this corpus look better. Fifteen documents
  cannot support that, and the rejection rules must generalise to captures
  nobody has seen.
- Do not silence a warning without understanding it. Two of the traps above
  first appeared as warnings.
- Do not present a timing number without stating `max_pages`. The team's
  pipeline reads only the first page; this one reads whole documents. Comparing
  them directly compares different amounts of work.

## When you are unsure

Write down the assumption in the docstring or the commit message and continue.
An assumption someone can find and argue with is fine. A silent guess is not.