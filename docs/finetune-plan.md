# Fine-Tuning Framework

> **Status**: Implemented. All code under `src/labeling_sft/`.
> **Target hardware**: RTX 4060 8 GB (or any GPU with ≥8 GB VRAM).
> **Companion document**: [finetune-experience.md](finetune-experience.md) — complete technical reference covering the full training → GGUF → Ollama pipeline, memory management deep-dive, and troubleshooting.

## Frameworks

| Framework | Version | Role |
|-----------|---------|------|
| **PyTorch** | `>=2.4` | Tensor computation, autograd, GPU execution. The foundation everything else builds on. |
| **transformers** | `>=4.45` | Hugging Face model hub. Loads Qwen2.5-3B-Instruct weights, tokenizer, and provides `Trainer` (training loop, checkpointing, evaluation) and `BitsAndBytesConfig` (4-bit quantization). |
| **peft** | `>=0.12` | Parameter-Efficient Fine-Tuning. Applies LoRA adapters (`LoraConfig` → `get_peft_model`) and provides `prepare_model_for_kbit_training()` to make quantized weights trainable. Also does `merge_and_unload()` for export. |
| **bitsandbytes** | `>=0.43` | 4-bit NF4 quantization (`BitsAndBytesConfig`). Compresses the 3B base model from ~6 GB (FP16) to ~2 GB, leaving room for LoRA adapters + optimizer state in 8 GB VRAM. |
| **datasets** | `>=2.20` | Memory-mapped dataset loading (`Dataset.from_list()` → `.map(tokenize)`). Streams data from disk, avoids loading all 499 examples into RAM at once. |
| **accelerate** | `>=0.33` | Device placement (`device_map="auto"`), mixed-precision dispatch. Abstracts away multi-GPU / CPU offload — though we only use a single GPU. |

### Framework Interaction Flow

```
datasets                bitsandbytes
  │ JSONL → Dataset         │ 4-bit NF4 quantize
  ▼                         ▼
tokenizer (transformers)   base model (transformers)
  │ text → input_ids         │ Qwen2.5-3B (2 GB in VRAM)
  ▼                         ▼
train dataset             peft
  │                         │ apply LoRA (r=16, alpha=32)
  │                         │ 7 target modules per layer
  ▼                         ▼
Trainer (transformers) ←─── LoRA model
  │
  │  DataCollator (_CompletionOnlyCollator)
  │  Cosine LR scheduler
  │  paged_adamw_8bit optimizer
  │  EarlyStoppingCallback
  │  Auto-resume from checkpoint-*
  ▼
checkpoints/  (every 25 steps)
  ├── adapter_model.safetensors
  ├── optimizer.pt + scheduler.pt
  ├── trainer_state.json
  └── rng_state.pth
```

---

## Context

QLoRA fine-tunes Qwen2.5-3B-Instruct to extract structured food knowledge graph data from recipe Markdown. The schema and labeling approach are specified in `docs/schema-design.md` and `docs/labeling-pipeline-plan.md`.

Bootstrap seed data: **499 labeled records** in `data/bootstrap/training.jsonl` (464 recipes + 35 not_a_recipe, zero parse errors, 100% core field coverage).

## Directory Structure

```
src/labeling_sft/
├── __init__.py                      # Package marker
├── system.txt                       # System prompt (self-contained copy)
├── dataset_builder.py               # JSONL → Qwen-chat-template Alpaca format
├── training_config.py               # QLoRAConfig dataclass
├── train.py                         # QLoRA training entry point
├── evaluate.py                      # Holdout evaluation
└── export.py                        # Adapter merge + GGUF export

data/
├── bootstrap/
│   └── training.jsonl               # 499 labeled records (input)
└── training/
    ├── train.jsonl                  # ~424 Alpaca-format training examples
    ├── val.jsonl                    # ~75 Alpaca-format validation examples
    └── dataset_stats.json           # Split statistics

models/
└── qwen2.5-3b-fatbb-v1/             # LoRA adapter output
    ├── checkpoint-25/               #   auto-saved every 25 steps
    ├── checkpoint-50/
    ├── ...
    ├── adapter_model.safetensors    #   final adapter weights
    └── tokenizer files
```

## Module Details

### 1. `dataset_builder.py` — Training Data Converter

Converts bootstrap JSONL into Alpaca format wrapped in Qwen chat template.

**Input** (`data/bootstrap/training.jsonl`):
```json
{"id": "...", "input": "<markdown>", "output": {<nested JSON>}, "source_path": "...", ...}
```

**Output** (`data/training/train.jsonl`, one record per line):
```json
{
  "instruction": "Extract structured food knowledge from the following recipe. Output valid JSON matching the FatBB food knowledge graph schema.",
  "input": "<full markdown content>",
  "output": "<compact JSON string>"
}
```

**Chat template** (assembled in `train.py`, not pre-rendered in JSONL):
```
<|im_start|>system
{system.txt}<|im_end|>
<|im_start|>user
{instruction}\n\n{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

**Key decisions**:
- 85/15 train/val split, stratified by source domain (recipetineats 183/32, wellplated 241/43)
- `not_a_recipe` records (35) kept in both splits
- `output` serialized as compact JSON (`separators=(",", ":")`) saving ~15% tokens vs pretty-printed
- Manual stratified split (no sklearn dependency)
- Seed 42, fixed

**CLI**:
```bash
python -m labeling_sft.dataset_builder
```

### 2. `training_config.py` — Hyperparameters

All defaults from `docs/labeling-pipeline-plan.md` QLoRA spec. Every field overridable via CLI.

```python
@dataclass
class QLoRAConfig:
    # --- Model ---
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")

    # --- 4-bit Quantization ---
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # --- Training ---
    per_device_train_batch_size: int = 1     # 1 sample per GPU step
    gradient_accumulation_steps: int = 4      # effective batch = 1 × 4 = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # --- Checkpointing ---
    output_dir: str = "models/qwen2.5-3b-fatbb-v1"
    logging_steps: int = 10
    save_steps: int = 25                    # ~15 checkpoints over full run
    save_total_limit: int = 2

    # --- Data ---
    train_file: str = "data/training/train.jsonl"
    val_file: str = "data/training/val.jsonl"
    seed: int = 42
```

### 3. `train.py` — Training Entry Point

**Pipeline**:
```
1. Load system prompt from labeling_sft/system.txt
2. Load train/val JSONL → format as Qwen chat template strings
3. Load tokenizer (pad_token=eos_token, padding_side="right")
4. Tokenize: max_length=4096, truncation
5. Load 4-bit quantized base model (BitsAndBytesConfig, device_map="auto", attn_implementation="sdpa")
6. prepare_model_for_kbit_training() → apply LoRA (get_peft_model)
7. _CompletionOnlyCollator: masks all non-assistant tokens in labels
8. Auto-resume: scan output_dir for checkpoint-* directories
9. Trainer.train() → save adapter + tokenizer
```

**_CompletionOnlyCollator** — self-contained collator (30 lines). Replaces `DataCollatorForCompletionOnlyLM` which was moved to `trl` in transformers v5. Tokenizes `<|im_start|>assistant`, finds its position in each sequence, masks all preceding tokens with `-100` in labels. Loss computed only on assistant-generated content.

**Auto-resume** — before creating `TrainingArguments`, scans `output_dir/checkpoint-*` and picks the latest. If found, passes `resume_from_checkpoint` to Trainer. Re-running the same command after Ctrl+C continues from the last saved step, restoring model weights, optimizer momentum, LR scheduler position, and RNG state.

**GPU Memory Management** — three strategies work together to keep Qwen2.5-3B + LoRA within 8 GB VRAM:

| Layer | Setting | What it does | Why it matters |
|-------|---------|---------------|----------------|
| Attention computation | `attn_implementation="sdpa"` | Uses PyTorch's fused Scaled Dot-Product Attention — computes attention in tiled blocks instead of materializing the full `[B, H, N, N]` matrix. | Reduces attention memory from O(n²) → O(n). At 4096 tokens this saves ~1–2 GB peak VRAM vs the default eager implementation. Works on all platforms (Windows/Linux) without requiring the `flash-attn` package. |
| Device placement | `device_map="auto"` | Lets `accelerate` decide per-layer GPU vs CPU placement at load time, rather than forcing every layer onto GPU 0 (`device_map={"": 0}`). | Provides a safety valve: if VRAM runs low, less-critical layers spill to CPU RAM instead of crashing. Also balances weights across multiple GPUs if available. |
| Fragment prevention | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Tells PyTorch's CUDA allocator to merge adjacent free blocks into larger contiguous segments. | Fixes the "enough total free VRAM but no single block big enough" failure — the `reserved but unallocated` OOM that the error message itself flags. |

These are complementary: SDPA reduces **how much** memory the forward/backward pass consumes (the root cause), `device_map="auto"` provides a **safety net** if peak still exceeds 8 GB, and `expandable_segments` eliminates **fragmentation** false positives where VRAM looks full but isn't.

Set the environment variable before launching training:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m labeling_sft.train
```

If OOM still occurs, reduce sequence length at runtime:
```bash
python -m labeling_sft.train --max_seq_length 2048
```

**CLI**:
```bash
# Full training
python -m labeling_sft.train

# Smoke test (5 steps, verify pipeline loads without OOM)
python -m labeling_sft.train --max_steps 5 --num_train_epochs 1

# Override any config field
python -m labeling_sft.train --learning_rate 1e-4 --save_steps 50
```

### 4. `evaluate.py` — Evaluation

Runs greedy-decoding inference on val.jsonl against the fine-tuned model.

**Metrics**:
| Metric | Target |
|--------|--------|
| JSON validity | > 85% |
| Schema validity | > 80% |
| Enum accuracy | > 80% |
| Not-a-recipe accuracy | Tracked separately |
| Per-field coverage | vs gold labels |

**CLI**:
```bash
python -m labeling_sft.evaluate \
  --adapter_dir models/qwen2.5-3b-fatbb-v1 \
  --val_file data/training/val.jsonl
```

### 5. `export.py` — Model Export

Merges LoRA adapter into base model for deployment.

**Steps**:
1. Load base model in bfloat16 (no quantization — needed for clean merge)
2. Load PEFT adapter
3. `model.merge_and_unload()`
4. Save merged model + tokenizer to `models/qwen2.5-3b-fatbb-v1-merged/`
5. (Optional) Convert to GGUF (q8_0) via `convert_hf_to_gguf.py`

**CLI**:
```bash
python -m labeling_sft.export \
  --adapter_dir models/qwen2.5-3b-fatbb-v1 \
  --output_dir models/qwen2.5-3b-fatbb-v1-merged
```

### 6. `system.txt`

Identical copy of `src/labeling/prompts/system.txt`. Makes `labeling_sft` self-contained — no import dependency on `labeling.prompts` at training time.

---

## Checkpoint / Crash Recovery

Every 25 steps the Trainer writes a full training state snapshot:

```
checkpoint-25/
├── adapter_model.safetensors   # LoRA weights
├── optimizer.pt                # AdamW momentum + variance
├── scheduler.pt                # Cosine LR decay position
├── trainer_state.json          # step, epoch, best_loss, loss history
└── rng_state.pth               # Random state (shuffle, dropout)
```

**What Ctrl+C loses**: steps since the last checkpoint (≤ 24 steps, ~2 minutes on RTX 4060).

**What Ctrl+C does NOT lose**: model weights, optimizer momentum, LR schedule position, step count — all restored on re-run.

**How to resume**: run the exact same command. `train.py` auto-detects the latest `checkpoint-*` directory.

---

## Installation

```bash
pip install -e ".[finetune]"
```

Dependencies are optional — the base `fatbb-agent` CLI and labeling pipeline do not require them.

---

## Verification Checklist

1. `dataset_builder.py` → train.jsonl (424) + val.jsonl (75), stratified split confirmed
2. `train.py --max_steps 5` → full pipeline loads, no OOM, one training step completes
3. `train.py` (full) → 3 epochs, ~375 steps, `adapter_model.safetensors` produced
4. Ctrl+C → re-run → auto-resumes from latest checkpoint
5. `evaluate.py` → JSON validity > 85%, enum accuracy > 80%
6. `export.py` → merged model loads and produces valid JSON on single inference
