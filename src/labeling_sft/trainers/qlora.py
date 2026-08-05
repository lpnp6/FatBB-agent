"""QLoRA fine-tuning trainer for Qwen2.5-3B-Instruct on recipe labeling data.

Wraps each Alpaca-format example in the Qwen chat template with the system
prompt loaded from ``labeling_sft/system.txt``.  Only assistant tokens
contribute to the loss (via ``_CompletionOnlyCollator``).

Usage::

    PYTHONPATH=src python -m labeling_sft.trainers.qlora
    PYTHONPATH=src python -m labeling_sft.trainers.qlora --max_steps 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

from labeling_sft.configs.qlora import QLoRAConfig, _qlora_config_fields
from labeling_sft.contracts import TrainingResult
from labeling_sft.interfaces.trainer import BaseTrainer

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

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    peak_alloc = torch.cuda.max_memory_allocated(0)

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
    """Return a ``TrainerCallback`` subclass that snapshots GPU memory."""
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
# Collator
# ---------------------------------------------------------------------------

class _CompletionOnlyCollator:
    """Pad a batch and mask labels so loss is computed only on assistant tokens.

    Replaces ``DataCollatorForCompletionOnlyLM`` (moved to ``trl`` in
    transformers v5) with a self-contained implementation.
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
# Trainer
# ---------------------------------------------------------------------------

class QLoRATrainer(BaseTrainer):
    """QLoRA fine-tuning trainer for Qwen2.5-3B-Instruct.

    Pipeline: load data → load 4-bit model + apply LoRA → tokenize → train.
    """

    config: QLoRAConfig

    def __init__(self, config: QLoRAConfig) -> None:
        super().__init__(config)

    # ── BaseTrainer implementation ──────────────────────────────────────

    def load_data(
        self,
        train_path: str,
        val_path: str,
    ) -> tuple[Any, Any]:
        """Load and format training / validation JSONL files."""
        train_file = Path(train_path)
        val_file = Path(val_path)
        for name, path in [("train", train_file), ("val", val_file)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} file not found: {path}\n"
                    f"  Run: python -m labeling_sft.dataset_builder --output_dir {self.config.output_dir}"
                )
            if path.stat().st_size == 0:
                raise ValueError(f"{name} file is empty: {path}")

        system_prompt = load_system_prompt()

        train_examples = self._load_jsonl(str(train_file), system_prompt)
        val_examples = self._load_jsonl(str(val_file), system_prompt)
        _gpu_snapshot("after dataset load (CPU)")

        return train_examples, val_examples

    def load_model(self) -> tuple[Any, Any]:
        """Load 4-bit base model, tokenizer, and apply LoRA."""
        model, tokenizer = self._load_base_model_and_tokenizer()
        _gpu_snapshot("after 4-bit model load")
        model = self._apply_lora(model)
        _gpu_snapshot("after LoRA apply")
        return model, tokenizer

    def train(
        self,
        train_path: str,
        val_path: str,
    ) -> TrainingResult:
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

        # -- Data ---------------------------------------------------------
        train_examples, val_examples = self.load_data(train_path, val_path)

        # -- Model & tokenizer --------------------------------------------
        model, tokenizer = self.load_model()

        # -- Tokenize -----------------------------------------------------
        train_cache = (
            str(Path(self.config.output_dir) / "cache" / "train.tok")
            if self.config.dataset_cache else None
        )
        val_cache = (
            str(Path(self.config.output_dir) / "cache" / "val.tok")
            if self.config.dataset_cache else None
        )
        tokenized_train = self._tokenize_dataset(
            train_examples, tokenizer, self.config.max_seq_length, cache_dir=train_cache,
        )
        tokenized_val = self._tokenize_dataset(
            val_examples, tokenizer, self.config.max_seq_length, cache_dir=val_cache,
        )
        _gpu_snapshot("after tokenization (CPU)")

        # -- Collator -----------------------------------------------------
        collator = _CompletionOnlyCollator(tokenizer)

        # -- Auto-resume: pick up the latest checkpoint -------------------
        resume_from = self.config.resume_from_checkpoint
        if resume_from is None:
            output = Path(self.config.output_dir)
            checkpoints = sorted(output.glob("checkpoint-*")) if output.is_dir() else []
            if checkpoints:
                resume_from = str(checkpoints[-1])
                logger.info("Auto-resuming from latest checkpoint: %s", resume_from)

        # -- Training arguments -------------------------------------------
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_accumulation_steps=8,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            eval_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
            seed=self.config.seed,
            resume_from_checkpoint=resume_from,
        )

        # -- Trainer ------------------------------------------------------
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
        torch.cuda.empty_cache()
        _gpu_snapshot("before train() — post empty_cache")

        train_result = trainer.train()

        # -- Save ---------------------------------------------------------
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info("Model and tokenizer saved to %s", self.config.output_dir)

        # -- Build result ------------------------------------------------
        final_eval_loss: float | None = None
        if hasattr(train_result, 'metrics') and 'eval_loss' in train_result.metrics:
            final_eval_loss = float(train_result.metrics['eval_loss'])

        return TrainingResult(
            output_dir=self.config.output_dir,
            adapter_path=self.config.output_dir,
            base_model_id=self.config.model_id,
            final_eval_loss=final_eval_loss,
            total_steps=getattr(train_result, 'global_step', 0),
            best_checkpoint=getattr(trainer.state, 'best_model_checkpoint', None),
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(jsonl_path: str, system_prompt: str) -> list[dict[str, str]]:
        """Load an Alpaca-format JSONL file, returning formatted text strings."""
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

    def _load_base_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load the 4-bit quantised base model and its tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        logger.info("Loading tokenizer for %s", self.config.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logger.info("Loading 4-bit quantised model: %s", self.config.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        logger.info(
            "Model device map: %s",
            {name: str(device) for name, device in model.hf_device_map.items()}
            if hasattr(model, "hf_device_map") else "single-device",
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model, tokenizer

    def _apply_lora(self, model: Any) -> Any:
        """Wrap *model* with a LoRA adapter and enable k-bit training."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _tokenize_dataset(
        examples: list[dict[str, str]],
        tokenizer: Any,
        max_length: int,
        *,
        cache_dir: str | None = None,
    ) -> Any:
        """Tokenize a list of ``{"text": ...}`` dicts into a ``Dataset``."""
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
                padding=False,
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
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-3B-Instruct on recipe labeling data",
    )
    defaults = QLoRAConfig()
    for field_name in _qlora_config_fields():
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()
    config = QLoRAConfig.from_cli_args(args)

    trainer = QLoRATrainer(config)
    trainer.train(
        train_path=str(Path(config.output_dir) / "train.jsonl"),
        val_path=str(Path(config.output_dir) / "val.jsonl"),
    )
