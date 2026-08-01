"""QLoRA fine-tuning entry point for Qwen2.5-3B-Instruct on recipe labeling data.

Wraps each Alpaca-format example in the Qwen chat template with the system
prompt loaded from ``labeling_sft/system.txt``.  Only assistant tokens
contribute to the loss (via ``DataCollatorForCompletionOnlyLM``).

Usage::

    PYTHONPATH=src python -m labeling_sft.train
    PYTHONPATH=src python -m labeling_sft.train --max_steps 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

from .training_config import QLoRAConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU memory diagnostics
# ---------------------------------------------------------------------------

def _gpu_snapshot(tag: str) -> None:
    """Log current GPU memory state with a human-readable *tag*."""
    import torch

    if not torch.cuda.is_available():
        logger.info("[mem] %s | CUDA unavailable", tag)
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)                      # ①
    allocated = torch.cuda.memory_allocated(0)                                 # ②
    reserved = torch.cuda.memory_reserved(0)                                   # ③
    peak_alloc = torch.cuda.max_memory_allocated(0)                            # ④

    def _gb(b: int) -> str:
        return f"{b / (1024 ** 3):.2f} GB"

    logger.info(
        "[mem] %s | free=%s / total=%s | alloc=%s | reserved=%s | peak_alloc=%s",
        tag,
        _gb(free_bytes),
        _gb(total_bytes),
        _gb(allocated),
        _gb(reserved),
        _gb(peak_alloc),
    )


def _make_memory_watchdog(
    max_train_snapshots: int = 3,
    max_eval_snapshots: int = 3,
):
    """Return a ``TrainerCallback`` subclass instance that snapshots GPU memory
    around training steps and eval cycles.

    Only the first *max_train_snapshots* steps and *max_eval_snapshots*
    evaluations are logged to avoid spam.  Inheriting from the HF base class
    ensures all other callback methods (``on_init_end``, ``on_epoch_begin``,
    etc.) have the default no-op implementation.
    """
    from transformers import TrainerCallback

    class _Watchdog(TrainerCallback):
        def __init__(self, max_train: int, max_eval: int):
            self._train_count = 0
            self._eval_count = 0
            self._max_train = max_train
            self._max_eval = max_eval

        def on_step_begin(self, args, state, control, **kwargs):
            if self._train_count < self._max_train:
                _gpu_snapshot(f"step {state.global_step + 1}  BEGIN")

        def on_step_end(self, args, state, control, **kwargs):
            if self._train_count < self._max_train:
                _gpu_snapshot(f"step {state.global_step + 1}  END")
                self._train_count += 1

        def on_evaluate(self, args, state, control, **kwargs):
            if self._eval_count < self._max_eval:
                _gpu_snapshot(f"eval  #{self._eval_count + 1}  (after)")
                self._eval_count += 1

    return _Watchdog(max_train_snapshots, max_eval_snapshots)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def load_system_prompt() -> str:
    """Return the system prompt shipped with the package."""
    return (
        files("labeling_sft")
        .joinpath("system.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def format_example(record: dict[str, str], system_prompt: str) -> str:
    """Wrap one Alpaca record in the Qwen chat template.

    Returns a single string suitable for tokenisation by the Qwen tokenizer.
    """
    user_content = f"{record['instruction']}\n\n{record['input']}"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{record['output']}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(jsonl_path: str, system_prompt: str) -> list[dict[str, str]]:
    """Load an Alpaca-format JSONL file, returning formatted text strings.

    Each returned dict has keys ``"text"`` (the full chat-template string)
    and ``"output"`` (the assistant-only portion, kept for reference).
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append({
                "text": format_example(record, system_prompt),
                "output": record["output"],
            })

    logger.info("Loaded %d examples from %s", len(examples), jsonl_path)
    return examples


# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------

def _build_bnb_config(config: QLoRAConfig) -> Any:
    """Build a ``BitsAndBytesConfig`` from the training config."""
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_model_and_tokenizer(config: QLoRAConfig) -> tuple[Any, Any]:
    """Load the 4-bit quantised base model and its tokenizer.

    Returns ``(model, tokenizer)``.  The model is **not** yet wrapped
    with LoRA — call ``apply_lora()`` afterwards.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer for %s", config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading 4-bit quantised model: %s", config.model_id)
    bnb_config = _build_bnb_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",             # auto-distribute layers; offloads to CPU when needed
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch SDPA — O(n) memory, works everywhere
    )
    logger.info(
        "Model device map: %s",
        {name: str(device) for name, device in model.hf_device_map.items()}
        if hasattr(model, "hf_device_map") else "single-device",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1

    return model, tokenizer


def apply_lora(model: Any, config: QLoRAConfig) -> Any:
    """Wrap *model* with a LoRA adapter and enable k-bit training."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(
    examples: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    *,
    cache_dir: str | None = None,
) -> Any:
    """Tokenize a list of ``{"text": ...}`` dicts into a ``Dataset``.

    If *cache_dir* is given and exists on disk, the pre-tokenized dataset is
    loaded directly (skipping CPU tokenisation).  Otherwise the dataset is
    tokenised and persisted to *cache_dir* for subsequent runs.
    """
    from datasets import Dataset

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            logger.info("Loading cached tokenized dataset from %s", cache_dir)
            return Dataset.load_from_disk(cache_dir)

    dataset = Dataset.from_list(examples)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # collator handles padding
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text", "output"])
    logger.info("Tokenized %d examples (max_length=%d)", len(tokenized), max_length)

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(cache_dir)
        logger.info("Cached tokenized dataset to %s", cache_dir)

    return tokenized


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class _CompletionOnlyCollator:
    """Pad a batch and mask labels so loss is computed only on assistant tokens.

    Replaces ``DataCollatorForCompletionOnlyLM`` (moved to ``trl`` in
    transformers v5) with a self-contained 30-line implementation.
    """

    def __init__(self, tokenizer: Any, response_template: str = "<|im_start|>assistant"):
        self._tokenizer = tokenizer
        self._response_ids: list[int] = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def _find_response_start(self, seq: list[int]) -> int | None:
        rt = self._response_ids
        for i in range(len(seq) - len(rt) + 1):
            if seq[i:i + len(rt)] == rt:
                return i + len(rt)
        return None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self._tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()

        for i in range(len(examples)):
            ids = batch["input_ids"][i].tolist()
            start = self._find_response_start(ids)
            labels[i, :start if start is not None else len(ids)] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(config: QLoRAConfig) -> None:
    """Execute the full QLoRA training pipeline."""
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for QLoRA training. "
            "No GPU detected — check that PyTorch was installed with CUDA support "
            "and your GPU drivers are working."
        )
    _gpu_snapshot("startup (before anything)")

    # -- Data ---------------------------------------------------------------
    train_file = Path(config.output_dir) / "train.jsonl"
    val_file = Path(config.output_dir) / "val.jsonl"
    for name, path in [("train", train_file), ("val", val_file)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found: {path}\n"
                f"  Run: python -m labeling_sft.dataset_builder --output_dir {config.output_dir}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"{name} file is empty: {path}")

    system_prompt = load_system_prompt()

    train_examples = load_jsonl_dataset(str(train_file), system_prompt)
    val_examples = load_jsonl_dataset(str(val_file), system_prompt)
    _gpu_snapshot("after dataset load (CPU)")

    # -- Model & tokenizer --------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(config)
    _gpu_snapshot("after 4-bit model load")
    model = apply_lora(model, config)
    _gpu_snapshot("after LoRA apply")

    train_cache = (
        str(Path(config.output_dir) / "cache" / "train.tok")
        if config.dataset_cache else None
    )
    val_cache = (
        str(Path(config.output_dir) / "cache" / "val.tok")
        if config.dataset_cache else None
    )
    tokenized_train = tokenize_dataset(
        train_examples, tokenizer, config.max_seq_length, cache_dir=train_cache,
    )
    tokenized_val = tokenize_dataset(
        val_examples, tokenizer, config.max_seq_length, cache_dir=val_cache,
    )
    _gpu_snapshot("after tokenization (CPU)")

    # -- Collator -----------------------------------------------------------
    collator = _CompletionOnlyCollator(tokenizer)

    # -- Auto-resume: pick up the latest checkpoint --------------------------
    resume_from = config.resume_from_checkpoint
    if resume_from is None:
        output = Path(config.output_dir)
        checkpoints = sorted(output.glob("checkpoint-*")) if output.is_dir() else []
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logger.info("Auto-resuming from latest checkpoint: %s", resume_from)

    # -- Training arguments -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=1,          # single sample per eval forward pass
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=8,             # split eval across 8 micro-batches;
                                                # each forward pass frees activations
                                                # before the next one starts — critical
                                                # because gradient checkpointing does
                                                # NOT run during eval
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,              # avoid forked-process CUDA context copies
        dataloader_pin_memory=False,           # no pinned-host memory; saves VRAM
        report_to=[],                          # no wandb / tensorboard
        seed=config.seed,
        resume_from_checkpoint=resume_from,
    )

    # -- Trainer ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            _make_memory_watchdog(max_train_snapshots=3, max_eval_snapshots=3),
        ],
    )

    logger.info("Starting training — %d train / %d val examples",
                len(tokenized_train), len(tokenized_val))
    _gpu_snapshot("before train() — pre empty_cache")
    torch.cuda.empty_cache()                    # free any lingering allocs from model load
    _gpu_snapshot("before train() — post empty_cache")
    trainer.train()

    # -- Save ---------------------------------------------------------------
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Model and tokenizer saved to %s", config.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-3B-Instruct on recipe labeling data",
    )
    # Allow overriding any QLoRAConfig field
    defaults = QLoRAConfig()
    for field_name in _config_fields():
        default = getattr(defaults, field_name)
        if field_name in ("lora_target_modules",):
            continue
        if isinstance(default, bool) and default is True:
            parser.add_argument(f"--{field_name}", action="store_true", default=None)
            parser.add_argument(f"--no-{field_name}", action="store_false", dest=field_name, default=None)
        elif isinstance(default, bool):
            parser.add_argument(f"--{field_name}", action="store_true", default=None)
        else:
            parser.add_argument(f"--{field_name}", type=type(default), default=None)
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Limit training to N steps (for smoke testing)",
    )
    return parser


def _config_fields() -> list[str]:
    return [f.name for f in QLoRAConfig.__dataclass_fields__.values()]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()
    config = QLoRAConfig.from_args(args)

    run_training(config)
