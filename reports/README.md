# Reports index

What's in this folder, and how to read it. Written for anyone who wasn't in
the room for the benchmark runs.

## Files

| File | What it is |
|---|---|
| `Technical_Assessment_Presentation.pptx` | The assessment deck: objectives, approach, results, and findings. Start here. |
| `cpu/cpu_matched-page1_protocolA_per-document_2026-09-05.csv` | One row per PDF. CPU, page-1 input, Protocol A (single pass, no warm-up -- matches Denzel's method). |
| `cpu/cpu_matched-page1_protocolA_per-stage_2026-09-05.csv` | Same run, broken down by pipeline stage (keywords / classify / summarize) instead of by document. |
| `gpu/gpu_matched-page1_protocolA_per-document_2026-09-05.csv` | Same run, on the GPU (GTX 1650, encoders in fp16). |
| `gpu/gpu_matched-page1_protocolA_per-stage_2026-09-05.csv` | GPU run, per-stage breakdown. |

**Naming pattern:** `<device>_<input-config>_<protocol>_<grain>_<date run was captured>.csv`.
`matched-page1` = only page 1 of each PDF was read, the setting that makes
timing comparable to Denzel's pipeline (which also only reads page 1).
`protocolA` = single pass, no warm-up -- the method this project uses for
head-to-head comparison against Denzel (Protocol B, warm-up + 3 timed runs
median, is for internal stage-latency analysis, not for this comparison; the
two must never be mixed in one table).

## These replace an earlier, wrong set of files

An earlier version of this folder had CSVs timestamped 2026-09-04. They are
gone, for two separate reasons, both real and both found by re-running the
pipeline rather than trusting the labels on disk:

1. **The old `gpu/` file wasn't a GPU run.** Its `total_time_sec` was
   byte-identical to the CPU file, and its own GPU telemetry showed the card
   idling at 300MHz throughout. It was captured before this machine's torch
   install had CUDA support; `--device cuda` was accepted but silently fell
   back to CPU.
2. **The "CPU, full document" (46.1s) / "GPU, full document" (11.8s)**
   numbers quoted in an earlier version of the deck came from a scratch file
   that was deleted before being re-verified, and the GPU number in it likely
   inherited the same silent-fallback bug -- it can't be trusted either.

The CSVs in this folder now are fresh, freshly re-run today with CUDA
actually working (confirmed in `runs/2026-09-05T06-42-32Z/pipeline.log`:
`loading model: securebert:torch:cuda:sdpa`), and both files' numbers are
independently traceable to a `runs/<run_id>/` directory with a real
`config.json` and `pipeline.log`.

**What changed in the deck:** the "full document" comparison rows are gone.
Only the page-1-matched, Protocol A numbers are shown, because that's what's
currently backed by a file:

| Configuration | Total time (s), n=5 shared docs | vs. Denzel |
|---|---|---|
| Denzel (single LLM) | 4.9 | 1.0x |
| Ours -- CPU | 41.7 | 8.6x |
| Ours -- GPU | 30.4 | 6.3x |

GPU is 1.4x faster than CPU here -- a real, verified number, but much
smaller than the 3.9x this folder claimed before the fix.

## Reading the per-document CSVs

- `document` -- the source URL, decoded and readable (added on top of the
  raw pipeline output for this handoff; `pdf_id` / `pdf_name` next to it are
  the original machine-readable columns, kept for traceability).
- `rejected`, `reject_reason` -- did the PDF clear the content-quality gate.
- `total_time_sec` -- wall time for that document, extraction through
  summary. This is the number quoted in the deck.
- `inference_time_sec` -- model compute only (keyword + classify + summarize),
  no extraction/cleaning/I-O. Comparable to Denzel's `single_turn_time_sec`.
- `peak_ram_mb`, `peak_vram_mb` -- see the caution below.
- `predicted_label`, `confidence`, `keywords`, `summary` -- the actual output
  for that document, so a speed number can be checked against what it
  produced.

**Caution on `peak_vram_mb`:** in earlier exports it showed non-zero values
even on CPU-only runs -- a metrics-collection artifact (a stale CUDA
context), not real usage. Not re-verified in this pass. Don't quote it
without checking the corresponding `config.json` device first.

## Reading the per-stage CSVs

Long format: one row per document per pipeline stage. Use this to see where
time goes within a document rather than just the total -- e.g. the
classification stage (ModernBERT-large zero-shot, one forward pass per
candidate label) is consistently the largest single cost, which is the
finding on the "Where the time goes" slide in the deck. Those per-stage
numbers are Protocol B -- a different, separately-labeled measurement from
the head-to-head table above, by design.

## Still missing

No full-document (`--match-denzel` off) CSV exists for either device. If
that comparison is needed, regenerate it explicitly rather than reusing old
numbers:

```bash
python -m pipeline run --force --device cpu --protocol A && python -m pipeline metrics
python -m pipeline run --force --device cuda --dtype fp16 --protocol A && python -m pipeline metrics
```

Then rename the output the same way as the files above
(`cpu_full-document_protocolA_per-document_<date>.csv`, etc.).

## Source of truth

`Reporting.md` at the project root has the rule this folder follows: every
number must trace to a file here, and every file must trace to a
reproducible command.
