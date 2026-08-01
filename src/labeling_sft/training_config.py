"""QLoRA hyperparameter configuration for Qwen2.5-3B-Instruct fine-tuning.

All defaults match the values specified in ``docs/labeling-pipeline-plan.md``
and ``docs/finetune-plan.md``: r=16, alpha=32, 4-bit, batch=1, grad_accum=4,
lr=2e-4, epochs=3, max_seq=8192.  Fits a single RTX 4060 8 GB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QLoRAConfig:
    """QLoRA hyperparameters for fine-tuning Qwen2.5-3B-Instruct.

    Every field can be overridden via the corresponding CLI flag in
    ``train.py``.  The defaults are tuned for a single RTX 4060 8 GB.
    """

    # ── Model ────────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    """Hugging Face model id or local path."""

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_r: int = 16
    """LoRA rank (docs: r=16)."""

    lora_alpha: int = 32
    """LoRA scaling factor (docs: alpha=32)."""

    lora_dropout: float = 0.05
    """Dropout applied to LoRA adapter weights."""

    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    """Linear projection layers to apply LoRA to.

    Covers all ``nn.Linear`` layers in each Qwen2.5 transformer block:
    attention (4 projections) + SwiGLU MLP (3 projections).  The embedding
    and output projection (``lm_head``) are left frozen.
    """

    # ── 4-bit quantisation ───────────────────────────────────────────────
    load_in_4bit: bool = True
    """Load base model in 4-bit (NF4) to fit in 8 GB VRAM."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype for 4-bit layers (bfloat16 on Ampere+ GPUs)."""

    bnb_4bit_quant_type: str = "nf4"
    """Quantisation data type — NF4 recommended for normally-distributed weights."""

    bnb_4bit_use_double_quant: bool = True
    """Apply a second quantisation round to the quantisation constants."""

    # ── Training ─────────────────────────────────────────────────────────
    per_device_train_batch_size: int = 1
    """Micro-batch size per GPU (1 = one 8192-token sample at a time)."""

    gradient_accumulation_steps: int = 4
    """Accumulate gradients over this many micro-batches before updating.
    Effective batch size = 1 × 4 = 4."""

    learning_rate: float = 2e-4
    """Peak learning rate (docs: lr=2e-4)."""

    num_train_epochs: int = 3
    """Number of full passes over the training set (docs: epochs=3)."""

    max_seq_length: int = 4096
    """Maximum tokenised sequence length.  4096 fits 8 GB with Flash-Attn-2;
    override with ``--max_seq_length 8192`` if you have headroom."""

    warmup_ratio: float = 0.03
    """Fraction of training steps used for linear warmup."""

    lr_scheduler_type: str = "cosine"
    """Learning-rate schedule: cosine decay from peak to near-zero."""

    optim: str = "paged_adamw_8bit"
    """8-bit AdamW optimiser with CPU off-load for memory efficiency."""

    weight_decay: float = 0.0
    """Weight decay — 0.0 is standard for LoRA fine-tuning."""

    max_grad_norm: float = 0.3
    """Gradient clipping threshold."""

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "models/qwen2.5-3b-fatbb-v1"
    """Directory to save the LoRA adapter weights and tokenizer."""

    logging_steps: int = 10
    """Log training metrics every N steps."""

    save_steps: int = 25
    """Save a checkpoint every N steps (~15 checkpoints over 375 steps)."""

    save_total_limit: int = 2
    """Keep at most this many checkpoints (oldest deleted first)."""

    seed: int = 42
    """Random seed for reproducibility."""

    dataset_cache: bool = True
    """Persist tokenized datasets to ``{output_dir}/cache/``.
    Re-loading skips CPU tokenisation on restart."""

    resume_from_checkpoint: str | None = None
    """Path to a checkpoint directory to resume training from."""

    # ── Helpers ──────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args: Any) -> QLoRAConfig:
        """Build a config from an argparse namespace, only overriding
        fields that were explicitly set on the command line."""
        defaults = cls()
        overrides: dict[str, Any] = {}
        for field_name in _config_fields():
            value = getattr(args, field_name, None)
            if value is not None and value != getattr(defaults, field_name):
                overrides[field_name] = value
        return cls(**overrides)


def _config_fields() -> list[str]:
    """Return the names of the fields defined on ``QLoRAConfig``."""
    return [f.name for f in QLoRAConfig.__dataclass_fields__.values()]
