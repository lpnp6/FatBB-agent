# Bootstrap sampling (one-off)

`run.py` is the only bootstrap entry point. It scans one source directory,
deduplicates and samples it, then labels the resulting manifest. It does not
use filename-keyword or recipe-structure filtering; `not_a_recipe` is retained
as a valid training label.

This is a **one-off bootstrap tool** for generating the initial manifests. It
is not part of the production auto-labeling runtime. Re-run it only when
intentionally creating a new sample, and keep the generated manifest and seed
with that labeling batch for reproducibility.

## Run the complete bootstrap workflow

Set the API key and run one command. On its first run it samples, persists
deduplication reservations, and labels; later runs reuse the manifest and
checkpoint to resume safely.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"  # optional
export OPENAI_MODEL="gpt-4o"                                  # optional
PYTHONPATH=src python -m labeling.bootstrap.run \
  --source-dir /path/to/markdown-corpus
```

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://your-compatible-endpoint/v1" # optional
$env:OPENAI_MODEL = "gpt-4o"                                 # optional
$env:PYTHONPATH = "src"
python -m labeling.bootstrap.run --source-dir C:\path\to\markdown-corpus
```

It writes validated recipe outputs and valid `not_a_recipe` classification
records to `data/bootstrap/training.jsonl`. Recipe records can be mapped to
the graph; non-recipe records remain training-only. Progress is atomically
recorded in `data/bootstrap/checkpoint.json`, allowing the same command to
resume without repeating completed API calls.
