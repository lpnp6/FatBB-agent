# QLoRA Fine-Tuning: Complete Technical Reference

> **Model**: Qwen2.5-3B-Instruct → FatBB food knowledge-graph extractor
> **Hardware**: RTX 4060 8 GB VRAM, 32 GB RAM, Windows 11
> **Result**: 114 MB LoRA adapter → 3.3 GB Q8_0 GGUF → Ollama (`fatbb-labeler`)

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment](#environment)
3. [Data Pipeline](#data-pipeline)
4. [Training](#training)
5. [Memory Management](#memory-management)
6. [Export to Ollama](#export-to-ollama)
7. [OllamaLabelingClient](#ollamalabelingclient)
8. [Troubleshooting](#troubleshooting)
9. [File Size Reference](#file-size-reference)
10. [One-Command Re-Export](#one-command-re-export)

---

## Architecture Overview

```
                         ┌──────────────────────────┐
                         │  data/bootstrap/          │
                         │  training.jsonl (499)     │
                         └──────────┬───────────────┘
                                    │ dataset_builder.py
                                    │ 85/15 stratified split
                                    ▼
                         ┌──────────────────────────┐
                         │  train.jsonl (424)        │
                         │  val.jsonl   (75)         │
                         └──────────┬───────────────┘
                                    │ train.py
                                    │ QLoRA: r=16, α=32, 4-bit NF4
                                    │ 3 epochs, lr=2e-4, cosine schedule
                                    │ max_seq=4096, SDPA attention
                                    ▼
                         ┌──────────────────────────┐
                         │  adapter_model.safetensors │
                         │  114 MB (LoRA weights)     │
                         └──────────┬───────────────┘
                                    │ export.py
                                    │ merge_and_unload()
                                    ▼
                         ┌──────────────────────────┐
                         │  merged model (bf16)       │
                         │  ~5.9 GB                   │
                         └──────────┬───────────────┘
                                    │ convert_hf_to_gguf.py
                                    │ --outtype q8_0
                                    ▼
                         ┌──────────────────────────┐
                         │  Q8_0 GGUF                 │
                         │  3.3 GB                    │
                         └──────────┬───────────────┘
                                    │ ollama create
                                    ▼
                         ┌──────────────────────────┐
                         │  fatbb-labeler:latest      │
                         │  Ollama model              │
                         └──────────────────────────┘
```

The same `LabelingClient` interface drives both bootstrap (OpenAI API) and production (local Ollama):

```
              LabelingClient (abstract)
              ├── label(markdown) → ExtractionResult
              └── repair(raw_output, error) → ExtractionResult
                     │
         ┌───────────┴───────────┐
         │                       │
  OpenAILabelingClient    OllamaLabelingClient
   (bootstrap phase)       (production phase)
```

---

## Environment

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4060, 8 GB VRAM |
| CUDA | 12.4 |
| RAM | 32 GB DDR4 |
| OS | Windows 11 Home China (build 26200) |
| Disk | NVMe SSD |

### Python Dependencies (exact versions)

```
torch==2.6.0+cu124
transformers==5.14.1
peft==0.20.0
datasets==5.0.1
accelerate==1.14.0
bitsandbytes==0.50.0
sentencepiece==0.2.2
ollama==0.6.2
```

These are the versions **actually used** in the successful training run. Version pinning matters: `transformers==5.14.1` moved `DataCollatorForCompletionOnlyLM` to `trl`, which is why `_CompletionOnlyCollator` was reimplemented locally.

### GPU Memory Budget (3B model, 4-bit QLoRA)

| Component | VRAM |
|-----------|------|
| Base model (4-bit NF4) | ~2.1 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| Optimizer state (paged_adamw_8bit) | ~0.8 GB |
| Activations (batch=1, seq=4096, SDPA) | ~3.5 GB |
| CUDA context + overhead | ~0.5 GB |
| **Total** | **~7.2 GB** |

Peak usage is highest during the backward pass (gradient computation). The model fits 8 GB with ~0.8 GB headroom.

---

## Data Pipeline

### Source

499 labeled records from the bootstrap phase (`data/bootstrap/training.jsonl`):
- 464 recipe labels + 35 `not_a_recipe` rejections
- Two source domains: RecipeTin Eats (183) and Well Plated (241)
- 100% core field coverage, zero parse errors

### dataset_builder.py

Converts the flat JSONL into Qwen-chat-template Alpaca format:

```
Input:  {"id": "...", "input": "<markdown>", "output": {<nested JSON>}}
Output: {"instruction": "...", "input": "<markdown>", "output": "<compact JSON>"}
```

**Key decisions**:
- **85/15 stratified split** by source domain: recipetineats (183/32), wellplated (241/43)
- `not_a_recipe` records included in both splits so the model learns rejection
- `output` serialized as compact JSON (`separators=(",", ":")`) — saves ~15% tokens vs pretty-printed
- Manual stratified split, no sklearn dependency
- Fixed seed 42

```
data/training/
├── train.jsonl          # 424 examples
├── val.jsonl            # 75 examples
└── dataset_stats.json   # Split statistics
```

### Chat Template (assembled at training time)

```
<|im_start|>system
{system.txt — full food KG extraction prompt with enum tables}<|im_end|>
<|im_start|>user
{instruction}\n\n{markdown}<|im_end|>
<|im_start|>assistant
{compact JSON output}<|im_end|>
```

Only assistant tokens contribute to loss (`_CompletionOnlyCollator` masks all preceding tokens with -100).

---

## Training

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 16 | Good expressivity/size tradeoff for 3B model |
| `alpha` | 32 | scaling = alpha/r = 2.0, strong adaptation signal |
| `dropout` | 0.05 | Light regularization to prevent adapter overfitting |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All 7 linear projections per transformer block. Embedding (`embed_tokens`) and output head (`lm_head`) left frozen — they already contain the base model's language understanding |
| Bias | none | No additional bias terms trained |

This targets **every `nn.Linear`** in each Qwen2.5 transformer block (4 attention + 3 SwiGLU MLP), covering the full forward pass through the adapter path. At r=16 with 36 layers and hidden_size=2048, this produces ~114 MB of trainable parameters (vs ~6 GB for full fine-tuning).

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `per_device_train_batch_size` | 1 | One 4096-token sample at a time — fits 8 GB VRAM |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 without materializing 4 sequences |
| `num_train_epochs` | 3 | Sufficient for 424-example dataset; early stopping at patience=5 prevents overfitting |
| `learning_rate` | 2e-4 | Standard QLoRA peak LR; cosine decay to near-zero |
| `warmup_ratio` | 0.03 | Linear warmup over ~11 steps |
| `lr_scheduler_type` | cosine | Smooth decay from peak → ~0, no plateau tuning needed |
| `optim` | paged_adamw_8bit | 8-bit AdamW with CPU offload for optimizer state pages |
| `weight_decay` | 0.0 | Standard for LoRA — adapter weights don't need L2 regularization |
| `max_grad_norm` | 0.3 | Gradient clipping to prevent loss spikes in early training |
| `max_seq_length` | 4096 | Balanced: fits 8 GB with SDPA, covers the full system prompt + longest recipe + JSON output |
| `seed` | 42 | Reproducible shuffles and dropout |

### Training Execution

```bash
# Set env var BEFORE launching
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full training
PYTHONPATH=src python -m labeling_sft.train

# Smoke test (verify pipeline, no OOM)
PYTHONPATH=src python -m labeling_sft.train --max_steps 5

# Override any hyperparameter
PYTHONPATH=src python -m labeling_sft.train --learning_rate 1e-4 --max_seq_length 8192
```

**Training duration**: ~375 steps × ~12s/step ≈ 75 minutes on RTX 4060.

### _CompletionOnlyCollator

Self-contained 30-line collator. Replaces `DataCollatorForCompletionOnlyLM` which was moved from `transformers` to `trl` in v5. Algorithm:

1. Tokenize `<|im_start|>assistant` to get the response template token IDs
2. For each sequence in the batch, find the position of `<|im_start|>assistant`
3. Mask all tokens **before** that position with `-100` in labels
4. Loss is computed only on assistant-generated content

### Checkpointing & Resume

Every 25 steps, Trainer writes:

```
checkpoint-25/
├── adapter_model.safetensors   # LoRA weights
├── optimizer.pt                # AdamW momentum + variance
├── scheduler.pt                # Cosine LR decay position
├── trainer_state.json          # step, epoch, best_loss, loss history
└── rng_state.pth               # Random state (shuffle, dropout)
```

**Auto-resume** — re-running the same command after Ctrl+C continues from the latest `checkpoint-*` directory. What's lost: ≤ 24 steps (~2 minutes). What's NOT lost: model weights, optimizer momentum, LR position, step count.

### GPU Memory Diagnostics

The training loop logs GPU memory state at key points:

```
[mem] startup (before anything)    | free=7.50 GB / total=8.00 GB
[mem] after dataset load (CPU)    | free=7.50 GB / total=8.00 GB
[mem] after 4-bit model load      | free=4.92 GB / total=8.00 GB
[mem] after LoRA apply            | free=4.60 GB / total=8.00 GB
[mem] before train() — post cache | free=4.58 GB / total=8.00 GB
```

The 4-bit base model consumes ~3.1 GB. LoRA adds ~0.3 GB. The remaining ~4.6 GB is available for activations during forward/backward — the critical window where OOMs occur.

---

## Memory Management

Three complementary strategies keep everything within 8 GB:

### 1. SDPA Attention (`attn_implementation="sdpa"`)

PyTorch's fused Scaled Dot-Product Attention computes attention in **tiled blocks** instead of materializing the full `[B, H, N, N]` attention matrix.

| Implementation | Memory (4096 tokens) | Works on Windows |
|---------------|---------------------|-------------------|
| eager (default) | O(n²) — ~2.1 GB peak | Yes |
| sdpa | O(n) — ~0.5 GB peak | Yes |
| flash_attn_2 | O(n) — ~0.3 GB peak | **No** (Linux only) |

At 4096 tokens, SDPA saves ~1.5 GB peak VRAM vs eager. This is the biggest single memory win.

### 2. `device_map="auto"`

Lets `accelerate` decide per-layer GPU vs CPU placement. Acts as a safety net: if VRAM runs low, less-critical layers spill to CPU RAM instead of crashing with OOM. In practice, all layers fit on GPU with 4-bit quantization, but this prevents edge-case failures.

### 3. `expandable_segments:True`

Tells PyTorch's CUDA allocator to merge adjacent free blocks into larger contiguous segments. Fixes the "enough total free VRAM but no single block big enough" failure.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### If OOM Still Occurs

1. Reduce sequence length: `--max_seq_length 2048`
2. Enable gradient checkpointing (already on): trades compute for memory, ~20% slower but saves ~2 GB
3. Use CPU offload for optimizer: `paged_adamw_8bit` already pages optimizer state to CPU
4. If training with batch>1, reduce to 1 and increase gradient accumulation

### Explaining the `torch_dtype` Warning

transformers v5+ emits: `[transformers] 'torch_dtype' is deprecated! Use 'dtype' instead!`

This is cosmetic — `torch_dtype` still works. The export script uses `torch_dtype` for backward compatibility with older transformers. To suppress:

```python
# Old (works everywhere, deprecated warning)
model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=compute_dtype)

# New (transformers v5+, no warning)
model = AutoModelForCausalLM.from_pretrained(..., dtype=compute_dtype)
```

---

## Export to Ollama

### Step 1: Merge LoRA → Full Model (GPU required)

```bash
PYTHONPATH=src python -m labeling_sft.export \
    --adapter_dir models/qwen2.5-3b-fatbb-v1 \
    --output_dir models/qwen2.5-3b-fatbb-v1-merged
```

**What happens**:
1. Load base model (`Qwen/Qwen2.5-3B-Instruct`) in bfloat16 — no quantization, needed for clean merge
2. Attach LoRA adapter via `PeftModel.from_pretrained()`
3. `model.merge_and_unload()` — folds adapter weights into base weights
4. Save merged model + tokenizer to output directory

**Output**: ~5.9 GB (bf16 safetensors)

### Step 2: Convert to GGUF

The conversion uses llama.cpp's `convert_hf_to_gguf.py`. On Windows without a C++ toolchain, the pure-Python approach works:

```bash
# Download llama.cpp conversion scripts (one-time)
# The scripts live in llama_cpp_convert/ (gitignored)

cd llama_cpp_convert && PYTHONPATH=. python convert_hf_to_gguf.py \
    --outtype q8_0 \
    --outfile ../models/qwen2.5-3b-fatbb-v1.Q8_0.gguf \
    ../models/qwen2.5-3b-fatbb-v1-merged
```

**Why Q8_0**:
- `f16` / `bf16`: ~6 GB, full quality but larger than needed
- `q8_0`: ~3.3 GB, near-lossless (PSNR > 40 dB), good default for local inference
- `q4_K_M`: would be ~1.8 GB but requires the `llama-quantize` C++ binary
- The convert script only supports: `f32, f16, bf16, q8_0, tq1_0, tq2_0, auto`

**Qwen2.5 model architecture** (detected automatically):
```
arch: Qwen2ForCausalLM
hidden_size: 2048
num_layers: 36
vocab_size: 151936
intermediate_size: 11008
num_attention_heads: 16
num_kv_heads: 2 (GQA)
rope_theta: 1000000.0
context_length: 32768
```

### Step 3: Create Ollama Model

**Modelfile** (`models/qwen2.5-3b-fatbb-v1.Modelfile`):

```dockerfile
FROM D:/FatBB-agent/models/qwen2.5-3b-fatbb-v1.Q8_0.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

```bash
ollama create fatbb-labeler -f models/qwen2.5-3b-fatbb-v1.Modelfile
```

**Important**: The GGUF path must be absolute on Windows. Relative paths cause `pull model manifest: file does not exist`.

**Windows-specific**: Ollama's default port may differ from 11434. Check with: `netstat -an | grep LISTENING | grep ollama` or `curl http://127.0.0.1:8000/api/tags`. In our environment Ollama uses port 8000.

### Testing the Ollama Model

```bash
# Interactive mode — paste markdown, Ctrl+Z then Enter
PYTHONPATH=src python -m labeling.test_label

# From file
PYTHONPATH=src python -m labeling.test_label recipe.md

# Specify host if non-default port
PYTHONPATH=src python -m labeling.test_label recipe.md --host http://127.0.0.1:8000
```

---

## OllamaLabelingClient

### Design

Implements the `LabelingClient` interface, identical contract to `OpenAILabelingClient`. The labeling orchestrator can swap backends by changing one import:

```python
# Bootstrap: cloud API
from labeling.clients import OpenAILabelingClient as Client

# Production: local Ollama
from labeling.clients import OllamaLabelingClient as Client

client = Client(
    model="fatbb-labeler:latest",
    label_prompt_builder=label_builder,
    repair_prompt_builder=repair_builder,
)
result = await client.label(markdown)
```

### Key Differences from OpenAI Client

| | OpenAILabelingClient | OllamaLabelingClient |
|---|---|---|
| Backend | OpenAI-compatible API | Local Ollama server |
| Auth | `api_key` | None (localhost) |
| Endpoint | `base_url` | `host` (default `http://127.0.0.1:11434`) |
| Max tokens | `max_tokens` | `num_predict` |
| Context window | Implicit | `num_ctx` (default 32768) |
| JSON mode | `response_format: json_object` | Prompt-based (system prompt instructs JSON-only output) |
| Usage | `prompt_tokens / completion_tokens` | `prompt_eval_count / eval_count` |
| Concurrency default | 5 | 1 (local model, sequential is safer) |
| Response type | `ChatCompletion` (dict-like) | `ChatResponse` (Pydantic model, attribute access) |

### Response Normalization

The `ollama` Python library returns Pydantic `ChatResponse` objects (attribute access: `.message.content`, `.prompt_eval_count`). Plain dicts would use key access (`["message"]["content"]`). The `_val()` helper handles both:

```python
def _val(obj, field, default=None):
    if hasattr(obj, field):
        return getattr(obj, field, default)
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default
```

### Context Window (`num_ctx`)

The system prompt with full JSON schema + enum tables consumes ~3,500 tokens. Ollama's default context is 4,096 — which overflows as soon as any markdown is added. The client defaults `num_ctx=32768` (matching the GGUF metadata `context_length`), passed as an Ollama option:

```python
options={
    "num_ctx": 32768,
    "num_predict": 8192,
    "temperature": 0.1,
}
```

Without this, Ollama returns: `"request (N tokens) exceeds the available context size (4096 tokens)"`.

---

## Troubleshooting

### Training

| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM during forward pass | Activation memory exceeds 8 GB | Reduce `max_seq_length` to 2048 |
| CUDA OOM with "reserved but unallocated" | Memory fragmentation | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `torch_dtype` deprecation warning | transformers v5 renamed parameter | Cosmetic, ignore, or change to `dtype=` |
| `DataCollatorForCompletionOnlyLM` not found | Moved to `trl` in transformers v5 | Uses local `_CompletionOnlyCollator` instead |
| Training loss not decreasing | Learning rate too low, or LoRA rank too small | Check `lr=2e-4`, `r=16`; verify adapter applied |

### GGUF Conversion

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'conversion'` | Missing llama.cpp Python modules | Download full `conversion/` + `gguf-py/` from llama.cpp repo |
| `ModuleNotFoundError: No module named 'sentencepiece'` | Qwen2 tokenizer uses SentencePiece | `pip install sentencepiece` |
| GitHub unreachable | Network restriction | Download scripts individually via `raw.githubusercontent.com` |
| `llama-cpp-python` install fails | No C++ compiler on Windows | Use pure-Python path (download conversion scripts, don't build C++ lib) |
| Long path errors during pip install | Windows MAX_PATH limit | Set `TMPDIR=C:\tmp` before `pip install` |

### Ollama

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ConnectionError` from Python ollama client | Server not running or wrong port | Start Ollama app; check port with `curl http://127.0.0.1:8000/api/tags` |
| Ollama on port 8000, not 11434 | Windows Ollama app uses different default | Pass `--host http://127.0.0.1:8000` or set `OLLAMA_HOST` |
| `pull model manifest: file does not exist` | Relative GGUF path in Modelfile | Use absolute Windows path: `FROM D:/FatBB-agent/models/...` |
| `exceeds the available context size (4096 tokens)` | System prompt + input > default ctx | Client passes `num_ctx=32768`; also set in Modelfile: `PARAMETER num_ctx 32768` |
| Empty response from chat API | `ollama` library returns Pydantic model, not dict | Use attribute access: `response.message.content`, not `response["message"]["content"]` |
| Response not valid JSON / wrapped in markdown | Model wasn't prompted strictly enough | System prompt explicitly requires "no Markdown fences"; temperature 0.1 reduces randomness |

---

## File Size Reference

| Artifact | Size | Format |
|----------|------|--------|
| Base model (Qwen2.5-3B-Instruct) | ~6 GB | FP16 on HuggingFace |
| Base model (4-bit NF4, in VRAM) | ~2.1 GB | 4-bit quantized |
| LoRA adapter | 114 MB | safetensors, r=16, 7 target modules |
| Training dataset (train.jsonl) | 6.3 MB | 424 records, compact JSON |
| Validation dataset (val.jsonl) | 1.2 MB | 75 records |
| Merged model (bf16) | 5.9 GB | safetensors, 1 shard |
| GGUF Q8_0 | 3.3 GB | Single file, 434 tensors |
| Ollama model (in ollama models dir) | 3.3 GB | Copied from GGUF |

**LoRA adapter is ~2% the size of the full model** — the whole point of parameter-efficient fine-tuning.

---

## One-Command Re-Export

After retraining the adapter, re-export everything:

```bash
# 1. Merge
PYTHONPATH=src python -m labeling_sft.export \
    --adapter_dir models/qwen2.5-3b-fatbb-v1 \
    --output_dir models/qwen2.5-3b-fatbb-v1-merged

# 2. GGUF
cd llama_cpp_convert && PYTHONPATH=. python convert_hf_to_gguf.py \
    --outtype q8_0 \
    --outfile ../models/qwen2.5-3b-fatbb-v1.Q8_0.gguf \
    ../models/qwen2.5-3b-fatbb-v1-merged && cd ..

# 3. Update Ollama
ollama create fatbb-labeler -f models/qwen2.5-3b-fatbb-v1.Modelfile
```

---

## Key Lessons

1. **SDPA is the silent hero** — switching from eager to SDPA attention is a one-line change (`attn_implementation="sdpa"`) that saves ~1.5 GB VRAM at 4096 tokens. It works on Windows where flash_attn doesn't.

2. **Always expandable_segments** — on Windows especially, CUDA memory fragmentation causes false OOMs. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` eliminates this class of failure.

3. **GGUF conversion on Windows requires the pure-Python path** — no C++ compiler, no CMake. Download the llama.cpp Python conversion scripts directly. Don't try to build `llama-cpp-python` from source.

4. **Ollama's Python client returns Pydantic models** — use attribute access (`.message.content`), not dict access. Helper functions that handle both make testing easier.

5. **The system prompt dominates context** — at ~3,500 tokens, it's 85% of Ollama's default 4,096-token window. Always set `num_ctx` explicitly (the GGUF metadata says 32,768).

6. **Absolute paths in Modelfile on Windows** — Ollama on Windows doesn't resolve relative paths in Modelfile `FROM` directives.

7. **Compact JSON for training data** — using `separators=(",", ":")` saves ~15% tokens vs pretty-printed JSON. At 499 records × ~2000 output tokens each, that's ~150K tokens saved.

8. **Stratified split preserves domain balance** — a random split can accidentally concentrate one domain in train or val. Manual stratification ensures both recipetineats and wellplated are proportionally represented in both sets.
