"""QLoRA fine-tuning trainer for Alpaca-format datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labeling_sft.configs.qlora import QLoRAConfig
from labeling_sft.contracts import ArtifactLocation, DatasetRecord, DatasetSplit, TrainingResult
from labeling_sft.artifact_store import artifact_store
from labeling_sft.dataset_loaders import LocalJsonlDatasetLoader
from labeling_sft.interfaces.dataset_loader import BaseDatasetLoader
from labeling_sft.interfaces.trainer import BaseTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU memory diagnostics
# ---------------------------------------------------------------------------

def _gpu_snapshot(tag: str) -> None:
    """Log current GPU memory state with a human-readable *tag*."""
    import torch # pyright: ignore[reportMissingImports]

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
    from transformers import TrainerCallback # pyright: ignore[reportMissingImports]

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

def load_system_prompt(path: str | Path) -> str:
    """Read the system prompt from *path*."""
    return Path(path).read_text(encoding="utf-8").strip()


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
        import torch # pyright: ignore[reportMissingImports]

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
    """QLoRA fine-tuning trainer for causal language models.

    Pipeline: load data → load 4-bit model + apply LoRA → tokenize → train.
    The configured model must be compatible with the selected LoRA modules.
    """

    config: QLoRAConfig

    def __init__(
        self,
        config: QLoRAConfig,
        dataset_loader: BaseDatasetLoader | None = None,
    ) -> None:
        super().__init__(config)
        if (
            not config.project_name
            or config.project_name in {".", ".."}
            or Path(config.project_name).name != config.project_name
            or not config.system_prompt_path
        ):
            raise ValueError("project_name and system_prompt_path are required")
        if not dataset_loader:
            raise ValueError("dataset_loader is required")
        self._dataset_loader = dataset_loader
        self._configure_logging(config.project_name)

    # ── BaseTrainer implementation ──────────────────────────────────────

    def load_data(
        self,
        split: DatasetSplit,
    ) -> tuple[Any, Any]:
        """Load standardized records from the dataset split and format them."""
        system_prompt = load_system_prompt(self.config.system_prompt_path)
        train_examples = self._format_records(self._dataset_loader.load(split.train), system_prompt)
        val_examples = self._format_records(self._dataset_loader.load(split.val), system_prompt)
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
        split: DatasetSplit,
        artifact_target: ArtifactLocation,
    ) -> TrainingResult:
        """Execute the full QLoRA training pipeline."""
        import torch # pyright: ignore[reportMissingImports]
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():
            logger.error("CUDA GPU is required for QLoRA training, but no GPU was detected")
            raise RuntimeError(
                "CUDA GPU is required for QLoRA training. "
                "No GPU detected — check that PyTorch was installed with CUDA support "
                "and your GPU drivers are working."
            )
        _gpu_snapshot("startup (before anything)")

        # -- Data ---------------------------------------------------------
        train_examples, val_examples = self.load_data(split)

        # -- Model & tokenizer --------------------------------------------
        model, tokenizer = self.load_model()

        # -- Tokenize -----------------------------------------------------
        train_cache = str(self.config.project_dir / "cache" / "train.tok") if self.config.dataset_cache else None
        val_cache = str(self.config.project_dir / "cache" / "val.tok") if self.config.dataset_cache else None
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
            output = self.config.project_dir
            checkpoints = sorted(output.glob("checkpoint-*")) if output.is_dir() else []
            if checkpoints:
                resume_from = str(checkpoints[-1])
                logger.info("Auto-resuming from latest checkpoint: %s", resume_from)

        # -- Training arguments -------------------------------------------
        training_args = TrainingArguments(
            output_dir=str(self.config.project_dir),
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
        trainer.save_model(self.config.project_dir)
        tokenizer.save_pretrained(self.config.project_dir)
        logger.info("Model and tokenizer saved to %s", self.config.project_dir)

        # -- Build result ------------------------------------------------
        final_eval_loss: float | None = None
        if hasattr(train_result, 'metrics') and 'eval_loss' in train_result.metrics:
            final_eval_loss = float(train_result.metrics['eval_loss'])

        artifact = artifact_store(artifact_target).publish(
            self.config.project_dir, artifact_target
        )
        return TrainingResult(
            model=artifact,
            adapter=artifact,
            base_model_id=self.config.model_id,
            final_eval_loss=final_eval_loss,
            total_steps=getattr(train_result, 'global_step', 0),
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _configure_logging(project_name: str) -> None:
        log_path = Path.home() / ".fatbb" / project_name / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if any(getattr(handler, "_fatbb_log_path", None) == str(log_path) for handler in logger.handlers):
            return

        for handler in list(logger.handlers):
            if getattr(handler, "_fatbb_log_path", None):
                logger.removeHandler(handler)
                handler.close()

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
            handler.setFormatter(formatter)
            handler._fatbb_log_path = str(log_path)  # type: ignore[attr-defined]
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    @staticmethod
    def _format_records(
        records: list[DatasetRecord], system_prompt: str
    ) -> list[dict[str, str]]:
        """Apply the chat template to standardized dataset records."""
        examples = [
            {
                "text": format_example(
                    {
                        "instruction": record.instruction,
                        "input": record.input,
                        "output": record.output,
                    },
                    system_prompt,
                ),
                "output": record.output,
            }
            for record in records
        ]
        logger.info("Loaded %d examples", len(examples))
        return examples

    def _load_base_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load the 4-bit quantised base model and its tokenizer."""
        import torch # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # pyright: ignore[reportMissingImports]

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
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # pyright: ignore[reportMissingImports]

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias=self.config.lora_bias,
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
        from datasets import Dataset # pyright: ignore[reportMissingImports]

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
