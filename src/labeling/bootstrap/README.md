# Bootstrap sampling (one-off)

`sample_corpus.py` samples all Markdown files without filename-keyword or
recipe-structure filtering. A model response of `not_a_recipe` is retained as
labeling data, so the labeling manifest always contains exactly the requested
number of documents.

This is a **one-off bootstrap tool** for generating the initial manifests. It
is not part of the production auto-labeling runtime. Re-run it only when
intentionally creating a new sample, and keep the generated manifest and seed
with that labeling batch for reproducibility.

With the two corpus roots, the default source ratios are 40% RecipeTin Eats
and 60% Well Plated:

```bash
PYTHONPATH=src python -m labeling.bootstrap.sample_corpus \
  --source recipetineats=/path/to/recipetineats \
  --source wellplated=/path/to/wellplated
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m labeling.bootstrap.sample_corpus `
  --source "recipetineats=C:\path\to\recipetineats" `
  --source "wellplated=C:\path\to\wellplated"
```

Windows Command Prompt:

```bat
set PYTHONPATH=src
python -m labeling.bootstrap.sample_corpus ^
  --source "recipetineats=C:\path\to\recipetineats" ^
  --source "wellplated=C:\path\to\wellplated"
```

It writes `data/samples/manifest.jsonl` (500 documents), `holdout.jsonl` (50
documents), and `sampling_report.json`. It also registers every labeling
manifest fingerprint as `in_flight` in `data/dedup_store.db` before any model
call. After a structured extraction is durably appended to the training JSONL,
the labeling orchestrator must change that record to `accepted`; production
uses the same database to skip duplicates. Sampling is deterministic for a
given seed. Use `--ratio NAME=WEIGHT` to set different source proportions.
