# labeling_sft Refactor Design — Interface Abstraction & Implementation Isolation

> **Status**: Design phase. Not yet implemented.
> **Target**: Make `src/labeling_sft/` extensible — swap model architectures, data sources, export formats, and eval strategies without touching shared infrastructure.

---

## 1. Motivation

The current `labeling_sft/` is a flat module of standalone functions and one dataclass. Adding a second training strategy (full fine-tune, DPO), a second data source (synthetic, multi-corpus), or a second export target (vLLM, Ollama Modelfile) means copy-pasting `train.py` and diverging from the original — no shared contracts, no reusable infrastructure.

This refactor introduces **5 abstract base classes** and **explicit data contracts** so that:

| Goal | How |
|------|-----|
| Swap training strategies | New `BaseTrainer` subclass in `trainers/` |
| Add a data source | New `BaseDatasetBuilder` subclass in `dataset_builders/` |
| Add an export format | New `BaseExporter` subclass in `exporters/` |
| Change evaluation metric | New `BaseEvaluator` subclass in `evaluators/` |
| All combinations work | They only depend on interfaces + contracts, never on each other's concrete types |

---

## 2. Target Directory Structure

```
src/labeling_sft/
├── interfaces/                       # 抽象接口层 —— 只定义，不实现
│   ├── __init__.py                   #   统一 re-export: BaseConfig, BaseDatasetBuilder,
│   │                                 #     BaseTrainer, BaseExporter, BaseEvaluator,
│   │                                 #     所有 contracts dataclass
│   ├── contracts.py                  #   跨模块数据契约（dataclass）
│   ├── config.py                     #   BaseConfig ABC
│   ├── dataset_builder.py            #   BaseDatasetBuilder ABC
│   ├── trainer.py                    #   BaseTrainer ABC
│   ├── exporter.py                   #   BaseExporter ABC
│   └── evaluator.py                  #   BaseEvaluator ABC
│
├── configs/                          # Config 实现族
│   ├── __init__.py                   #   re-export QLoRAConfig
│   └── qlora.py                      #   QLoRAConfig(BaseConfig)
│
├── dataset_builders/                 # DatasetBuilder 实现族
│   ├── __init__.py                   #   re-export BootstrapDatasetBuilder
│   └── bootstrap.py                  #   BootstrapDatasetBuilder(BaseDatasetBuilder)
│
├── trainers/                         # Trainer 实现族
│   ├── __init__.py                   #   re-export QLoRATrainer
│   └── qlora.py                      #   QLoRATrainer(BaseTrainer)
│
├── exporters/                        # Exporter 实现族
│   ├── __init__.py                   #   re-export GGUFExporter
│   └── gguf.py                       #   GGUFExporter(BaseExporter)
├── artifact_store/                   # ArtifactStore 实现族
│   └── local.py                      #   LocalArtifactStore(BaseArtifactStore)
│
├── evaluators/                       # Evaluator 实现族
│   ├── __init__.py                   #   re-export QwenEvaluator
│   └── qwen.py                       #   QwenEvaluator(BaseEvaluator)
│
├── system.txt                        # System prompt（不变）
└── __init__.py                       # 顶层 re-export，保持向后兼容
```

### 2.1 命名约定

| 层级 | 目录/文件 | 类名 | 示例 |
|------|----------|------|------|
| 抽象 | `interfaces/{name}.py` | `Base{Name}` | `interfaces/trainer.py` → `BaseTrainer` |
| 数据 | `interfaces/contracts.py` | `{Noun}` | `DatasetSplit`, `EvalReport` |
| 实现 | `{name}s/{variant}.py` | `{Variant}{Name}` | `trainers/qlora.py` → `QLoRATrainer` |

- 接口文件名用单数（`trainer.py`），实现目录用复数（`trainers/`）—— 避免 `import` 路径歧义。
- 实现文件名用变体名（`qlora.py`、`bootstrap.py`），一个文件一个类 —— 方便定位和 diff。

---

## 3. Data Contracts — `interfaces/contracts.py`

所有模块间传递的数据结构。**跨模块通信必须使用这些 dataclass，禁止裸 `dict` 跨越模块边界。**

### 3.1 DatasetBuilder → Trainer / Evaluator

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetRecord:
    """单条 Alpaca 格式样本。"""
    instruction: str
    input: str           # markdown 原文
    output: str          # compact JSON string（gold label）


@dataclass
class DatasetStats:
    """数据集划分统计。"""
    total_valid_records: int
    skipped_records: int
    recipe_count: int
    not_a_recipe_count: int
    train_count: int
    val_count: int
    val_split: float
    seed: int
    domain_distribution: dict[str, dict[str, int]]
    # 例: {"recipetineats": {"train": 183, "val": 32}, "wellplated": {"train": 241, "val": 43}}


@dataclass
class DatasetSplit:
    """DatasetBuilder.build() 的返回值。

    Trainer 和 Evaluator 通过此契约定位数据文件，不感知数据来源。
    """
    train_path: str      # train.jsonl 绝对路径
    val_path: str        # val.jsonl 绝对路径
    stats: DatasetStats
```

### 3.2 Trainer → Exporter / Evaluator

```python
@dataclass
class TrainingResult:
    """Trainer.train() 的返回值。

    Exporter 和 Evaluator 通过此契约定位模型产物，不感知训练细节。
    """
    output_dir: str              # 模型/适配器保存目录
    adapter_path: str            # LoRA adapter 路径
    base_model_id: str           # 基座模型 HuggingFace ID
    final_eval_loss: float | None
    total_steps: int
    best_checkpoint: str | None  # 最佳 checkpoint 路径
```

### 3.3 Evaluator → 外部

```python
@dataclass
class EvalReport:
    """单模型评估结果。"""
    model_label: str             # "Fine-tuned" | "Base" | ...
    total_examples: int
    json_valid: int
    json_validity_pct: float
    validator_pass: int
    validator_pass_pct: float
    enum_valid_fields: int
    enum_total_fields: int
    enum_accuracy_pct: float
    not_a_recipe_correct: int
    not_a_recipe_total: int
    not_a_recipe_accuracy_pct: float | None
    field_coverage: dict[str, dict[str, int]]    # {field: {present, total, pct}}
    validator_errors: dict[str, int]             # {error_type: count}
    raw_metrics: dict[str, Any]                  # 完整原始指标（调试用）


@dataclass
class ComparisonReport:
    """双模型对比评估结果（Evaluator.compare() 的返回值）。"""
    base_model_id: str
    adapter_dir: str
    base: EvalReport
    fine_tuned: EvalReport
    divergent_examples: list[dict[str, Any]]
    # 每条: {"index": int, "base_summary": str, "ft_summary": str}
```

### 3.4 Exporter → 外部

```python
@dataclass
class ExportResult:
    """Exporter.export() 的返回值。"""
    artifact: ArtifactLocation   # 导出产物位置
    format: str                  # "gguf" | "vllm" | ...
    size_mb: float               # 产物总大小
    base_model_id: str           # 使用的基座模型 ID
```

### 3.5 契约数据流总览

```
                        BaseConfig
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    DatasetBuilder        Trainer       Evaluator
            │               │               │
            ▼               ▼               ▼
       DatasetSplit   TrainingResult    EvalReport
            │               │          ComparisonReport
            │               │               │
            └───────┬───────┘               │
                    │                       │
                    ▼                       │
              train.jsonl                   │
              val.jsonl ────────────────────┘
                    │
                    ▼
               Exporter
                    │
                    ▼
              ExportResult
```

---

## 4. Abstract Interfaces

### 4.1 `BaseConfig` — `interfaces/config.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Self


class BaseConfig(ABC):
    """训练/评估配置的抽象基类。

    子类可以是 dataclass、pydantic BaseModel、或普通 class ——
    只要满足以下 property 和方法契约即可。
    """

    # ── 必须属性 ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Hugging Face model ID 或本地路径。"""
        ...

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """模型/适配器输出目录。"""
        ...

    @property
    @abstractmethod
    def seed(self) -> int:
        """随机种子。"""
        ...

    # ── 序列化 ────────────────────────────────────────────────────

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容 dict，用于保存/恢复配置。"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """从 dict 反序列化。"""
        ...

    @classmethod
    @abstractmethod
    def from_cli_args(cls, args: Any) -> Self:
        """从 argparse.Namespace 构建，仅覆盖显式传入的字段。

        典型实现：取 defaults，遍历所有 field，用 args 上非 None 的值覆盖。
        """
        ...

    # ── 校验 ──────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """校验配置合法性。返回问题描述列表，空列表 = 通过。

        子类可覆盖以实现自定义校验（如 lora_r > 0, max_seq_length >= 512 等）。
        """
        return []
```

**实现要点**：
- `QLoRAConfig` 可直接用 `@dataclass` 实现，`to_dict()` = `dataclasses.asdict()`，`from_dict()` = `cls(**data)`，`from_cli_args()` 已有现成逻辑。
- 未来 `FullFinetuneConfig` 只需保证同样的 3 个 property + 3 个方法即可被所有下游消费。

### 4.2 `BaseDatasetBuilder` — `interfaces/dataset_builder.py`

```python
from abc import ABC, abstractmethod

from .contracts import DatasetSplit, DatasetRecord


class BaseDatasetBuilder(ABC):
    """数据集构建器抽象。

    职责：将任意格式的原始标注数据转换为 Alpaca 格式的 train/val 划分。
    Trainer 和 Evaluator 只消费其产出的 JSONL 文件，不感知数据来源。
    """

    @abstractmethod
    def build(
        self,
        input_path: str,
        train_path: str,
        val_path: str,
        stats_path: str,
        val_split: float = 0.15,
        seed: int = 42,
    ) -> DatasetSplit:
        """执行数据集构建。

        Args:
            input_path:  原始标注数据路径（格式由子类定义）。
            train_path:  训练集输出路径。
            val_path:    验证集输出路径。
            stats_path:  统计信息输出路径。
            val_split:   验证集比例 (0.0–1.0)。
            seed:        随机种子。

        Returns:
            DatasetSplit: 划分后的文件路径 + 统计信息。

        Raises:
            FileNotFoundError: input_path 不存在。
            ValueError:        无有效记录。
        """
        ...

    @abstractmethod
    def load_split(
        self,
        train_path: str,
        val_path: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """加载已构建的数据集划分为内存中的 record 列表。

        供 Trainer.load_data() 和 Evaluator 读取数据使用。
        每条 record 为 {"instruction": ..., "input": ..., "output": ...}。

        Returns:
            (train_records, val_records)
        """
        ...
```

**现有代码映射**：`dataset_builder.py` 的 `build_dataset()` → `BootstrapDatasetBuilder.build()`。

### 4.3 `BaseTrainer` — `interfaces/trainer.py`

```python
from abc import ABC, abstractmethod
from typing import Any

from .config import BaseConfig
from .contracts import TrainingResult


class BaseTrainer(ABC):
    """模型训练器抽象。

    职责：接收 Config + 数据文件路径，执行训练，产出模型权重。

    拆分为三个独立可覆盖的阶段：
      load_data()  → 数据预处理
      load_model() → 模型加载 + 适配器配置
      train()      → 训练循环

    子类可以只替换一个阶段（如换模型架构），其余阶段复用。
    """

    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    @abstractmethod
    def load_data(
        self,
        train_path: str,
        val_path: str,
    ) -> tuple[Any, Any]:
        """加载并预处理训练/验证数据。

        Returns:
            (train_dataset, val_dataset)：已 tokenize 的 HuggingFace Dataset。
        """
        ...

    @abstractmethod
    def load_model(self) -> tuple[Any, Any]:
        """加载并配置模型和分词器。

        Returns:
            (model, tokenizer)：已加载到设备、应用 LoRA（如适用）的模型。
        """
        ...

    @abstractmethod
    def train(
        self,
        train_path: str,
        val_path: str,
    ) -> TrainingResult:
        """执行完整训练流程。

        默认实现：load_data() → load_model() → 训练循环 → 保存。

        Args:
            train_path: train.jsonl 路径。
            val_path:   val.jsonl 路径。

        Returns:
            TrainingResult。
        """
        ...
```

**现有代码映射**：`train.py` 的 `run_training()` 拆分为三个阶段方法 → `QLoRATrainer`。

### 4.4 `BaseExporter` — `interfaces/exporter.py`

```python
from abc import ABC, abstractmethod

from .contracts import ExportResult


class BaseExporter(ABC):
    """模型导出器抽象。

    职责：将训练产出的 adapter/checkpoint 转换为可部署格式。

    每种导出格式是一个独立的 BaseExporter 子类，互不依赖。
    """

    @abstractmethod
    def export(
        self,
        adapter_dir: str,
        output_dir: str,
        base_model_id: str | None = None,
        **kwargs,
    ) -> ExportResult:
        """执行模型导出。

        Args:
            adapter_dir:   训练产出的 adapter/checkpoint 目录。
            output_dir:    导出目标目录。
            base_model_id: 基座模型 ID（None 时自动推断）。

        Returns:
            ExportResult。

        Raises:
            FileNotFoundError: adapter_dir 不存在。
        """
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """该 Exporter 产出的格式标识符。

        如 "gguf_q8_0", "vllm_bf16" 等。
        供 CLI 和注册表自动发现。
        """
        ...
```

`GGUFExporter` 是唯一的导出路径：它接收 `TrainingResult` 与目标
`ArtifactLocation`，并返回含产物位置的 `ExportResult`。

### 4.5 `BaseEvaluator` — `interfaces/evaluator.py`

```python
from abc import ABC, abstractmethod
from typing import Any

from .contracts import EvalReport, ComparisonReport


class BaseEvaluator(ABC):
    """模型评估器抽象。

    职责：加载模型 + 验证集，执行推理，计算指标。

    支持两种评估模式：
      evaluate()  → 单模型评估
      compare()   → 基座模型 vs fine-tuned 模型对比
    """

    @abstractmethod
    def load_model(
        self,
        adapter_dir: str | None,
        base_model_id: str,
        **kwargs,
    ) -> tuple[Any, Any]:
        """加载模型和分词器。

        Args:
            adapter_dir:  LoRA adapter 路径。None 表示仅加载基座模型（用于对比）。
            base_model_id: HuggingFace 模型 ID。

        Returns:
            (model, tokenizer)
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        adapter_dir: str | None,
        val_path: str,
        base_model_id: str,
        output_report: str | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> EvalReport:
        """单模型评估。

        Args:
            adapter_dir:   adapter 路径。None = 仅基座模型。
            val_path:      val.jsonl 路径。
            base_model_id: 基座模型 ID。
            output_report: 报告 JSON 输出路径（None = 不写文件）。
            max_samples:   限制评估样本数（None = 全部）。

        Returns:
            EvalReport。
        """
        ...

    @abstractmethod
    def compare(
        self,
        adapter_dir: str,
        val_path: str,
        base_model_id: str,
        output_report: str | None = None,
        diff_examples: int = 5,
        max_samples: int | None = None,
        **kwargs,
    ) -> ComparisonReport:
        """双模型对比评估。

        先跑基座模型 → 释放显存 → 再跑 fine-tuned 模型 → 产出对比报告。

        Returns:
            ComparisonReport。
        """
        ...
```

**现有代码映射**：`evaluate.py` 的 `evaluate()` + `evaluate_with_comparison()` → `QwenEvaluator`。

---

## 5. Implementation Directories

### 5.1 `configs/` — Config 实现

```
configs/
├── __init__.py    # from .qlora import QLoRAConfig
└── qlora.py       # QLoRAConfig(BaseConfig) — 现有 training_config.py 迁移
```

**`QLoRAConfig`** 保持现有 23 个字段不变，只需：
1. 继承 `BaseConfig`
2. 添加 `to_dict()` → `dataclasses.asdict(self)`
3. 添加 `from_dict(data)` → `cls(**data)`
4. 重构 `from_cli_args()` → 走 `BaseConfig` 接口
5. 添加 `validate()` → 检查 `lora_r > 0`, `max_seq_length >= 512`, 等

**未来扩展**：
- `configs/full_ft.py` → `FullFinetuneConfig(BaseConfig)`：全参微调，无 LoRA/量化字段
- `configs/dpo.py` → `DPOConfig(BaseConfig)`：DPO 训练，额外 `beta`, `reference_model` 字段

### 5.2 `dataset_builders/` — DatasetBuilder 实现

```
dataset_builders/
├── __init__.py         # from .bootstrap import BootstrapDatasetBuilder
└── bootstrap.py        # BootstrapDatasetBuilder(BaseDatasetBuilder)
```

**`BootstrapDatasetBuilder`** 将现有 `dataset_builder.py` 的函数逻辑封装为类方法：
- `_extract_domain()`, `_validate_output()` 保持为私有静态方法
- `build_dataset()` → `build()`（返回 `DatasetSplit` 而非裸 dict）
- 新增 `load_split()` —— 简单的 JSONL 加载器

**未来扩展**：
- `dataset_builders/synthetic.py` → `SyntheticDatasetBuilder`：从 LLM 生成的合成数据构建
- `dataset_builders/multi_source.py` → `MultiSourceDatasetBuilder`：多数据源混合、加权采样

### 5.3 `trainers/` — Trainer 实现

```
trainers/
├── __init__.py    # from .qlora import QLoRATrainer
└── qlora.py       # QLoRATrainer(BaseTrainer)
```

**`QLoRATrainer`** 将现有 `train.py` 重构为三阶段：

| 阶段 | 方法 | 内容 |
|------|------|------|
| 数据 | `load_data()` | 从 JSONL 加载 → `format_example()` → 聊天模板 → `tokenize_dataset()` |
| 模型 | `load_model()` | `_build_bnb_config()` → `AutoModelForCausalLM` → `apply_lora()` |
| 训练 | `train()` | `_CompletionOnlyCollator` → `TrainingArguments` → `Trainer.train()` |

内部保留：
- `_gpu_snapshot()`, `_make_memory_watchdog()` → 静态/私有方法
- `_CompletionOnlyCollator` → 保持为私有内部类
- `format_example()` → 静态方法（Evaluator 也依赖它）
- `load_system_prompt()` → 静态方法

**未来扩展**：
- `trainers/full_ft.py` → `FullFinetuneTrainer`：覆盖 `load_model()` 不加载 4-bit 量化
- `trainers/dpo.py` → `DPOTrainer`：覆盖 `load_data()` 加载 preference pairs，覆盖 `train()` 使用 `DPOTrainer`

### 5.4 `exporters/` — Exporter 实现

```
exporters/
├── __init__.py       # from .gguf import GGUFExporter
└── gguf.py           # GGUFExporter(BaseExporter) — format_name="gguf_q8_0"
```

`BaseArtifactStore` 位于 `interfaces/artifact_store.py`；本地实现位于
`artifact_store/local.py`。

**`GGUFExporter`**：
- `format_name` → `"gguf_q8_0"`
- 接收 `TrainingResult`，在临时目录中加载基座模型与 adapter，执行 `merge_and_unload()`，再调用 `convert_hf_to_gguf.py`
- 通过 `ArtifactStore` 物化输入并发布最终 GGUF；当前实现本地存储，其他后端按需添加

**未来扩展**：
- `exporters/vllm.py` → `VLLMExporter`：直接输出 vLLM 可加载的格式
- `exporters/ollama.py` → `OllamaExporter`：生成 Modelfile + 打包

### 5.5 `evaluators/` — Evaluator 实现

```
evaluators/
├── __init__.py    # from .qwen import QwenEvaluator
└── qwen.py        # QwenEvaluator(BaseEvaluator)
```

**`QwenEvaluator`** 将现有 `evaluate.py` 重构为类：

| 方法 | 内容 |
|------|------|
| `load_model()` | `_load_base_model()` / `load_eval_model()` 逻辑 |
| `evaluate()` | 现有 `evaluate()` 函数逻辑，返回 `EvalReport` |
| `compare()` | 现有 `evaluate_with_comparison()` 逻辑，返回 `ComparisonReport` |

内部保留：
- `_extract_json()`, `_check_enum_values()`, `_per_field_coverages()` → 私有方法
- `_run_eval_pass()` → 核心评估循环（单模型）
- `_print_metrics()`, `_print_comparison()`, `_print_example_diffs()` → 打印辅助方法

**未来扩展**：
- `evaluators/generic.py` → `GenericEvaluator`：不依赖 Qwen 聊天模板的通用评估
- `evaluators/batch.py` → `BatchEvaluator`：批量多模型对比（3+ 模型）

---

## 6. `__init__.py` 层级与向后兼容

### 6.1 顶层 `labeling_sft/__init__.py`

保持对外 API 不变：

```python
"""QLoRA fine-tuning framework for Qwen2.5-3B-Instruct food knowledge extraction."""

from labeling_sft.interfaces import (
    # Contracts
    DatasetRecord, DatasetStats, DatasetSplit,
    TrainingResult, EvalReport, ComparisonReport, ExportResult,
    # Interfaces
    BaseConfig, BaseDatasetBuilder, BaseTrainer, BaseExporter, BaseEvaluator,
)

from labeling_sft.configs import QLoRAConfig
from labeling_sft.dataset_builders import BootstrapDatasetBuilder
from labeling_sft.trainers import QLoRATrainer
from labeling_sft.exporters import GGUFExporter
from labeling_sft.evaluators import QwenEvaluator
```

### 6.2 各实现目录的 `__init__.py`

每个只做 re-export，不包含逻辑：

```python
# configs/__init__.py
from .qlora import QLoRAConfig
```

### 6.3 CLI 入口

现有 `if __name__ == "__main__":` 代码块迁移到各自的实现类中，保持 `python -m labeling_sft.dataset_builder` 等命令可用。过渡期保留原文件作为 thin wrapper：

```python
# dataset_builder.py（过渡期，指向新位置）
from labeling_sft.dataset_builders.bootstrap import BootstrapDatasetBuilder

if __name__ == "__main__":
    BootstrapDatasetBuilder.cli()  # 或保留原 argparse 逻辑
```

---

## 7. 扩展指南：如何新增一个实现

以「新增 DPO 训练器」为例：

### Step 1: 新增 Config（如需要）

```python
# configs/dpo.py
from dataclasses import dataclass
from labeling_sft.interfaces import BaseConfig

@dataclass
class DPOConfig(BaseConfig):
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    beta: float = 0.1
    learning_rate: float = 5e-5
    # ... 其余字段
```

### Step 2: 新增 Trainer

```python
# trainers/dpo.py
from labeling_sft.interfaces import BaseTrainer

class DPOTrainer(BaseTrainer):
    def load_data(self, train_path, val_path):
        # 加载 preference pairs，而非单条样本
        ...

    def load_model(self):
        # 可复用 QLoRATrainer 的模型加载逻辑
        ...

    def train(self, train_path, val_path):
        # 使用 trl.DPOTrainer 而非 transformers.Trainer
        ...
```

### Step 3: 注册到 `__init__.py`

```python
# trainers/__init__.py
from .qlora import QLoRATrainer
from .dpo import DPOTrainer
```

### Step 4: 即可使用

```python
config = DPOConfig(beta=0.1)
trainer = DPOTrainer(config)
result = trainer.train("data/train.jsonl", "data/val.jsonl")
# result 是 TrainingResult —— 与 QLoRATrainer 的返回值类型完全相同
# Exporter 和 Evaluator 可以直接消费它
```

---

## 8. 迁移步骤

| 阶段 | 内容 | 风险 |
|------|------|------|
| **Phase 1** | 创建 `interfaces/` 目录，写入所有 ABC + contracts。**不修改任何现有代码。** | 零风险 |
| **Phase 2** | 创建 `configs/qlora.py`，让 `QLoRAConfig` 继承 `BaseConfig`。原 `training_config.py` 改为 thin re-export wrapper。 | 低风险 |
| **Phase 3** | 创建 `dataset_builders/bootstrap.py`，将 `build_dataset()` 封装为 `BootstrapDatasetBuilder`。 | 低风险 |
| **Phase 4** | 创建 `trainers/qlora.py`，将 `run_training()` 拆分为 `QLoRATrainer`。 | 中风险 — 训练逻辑核心 |
| **Phase 5** | 创建 `exporters/gguf.py`，由其内部完成 adapter 合并与转换。 | 低风险 |
| **Phase 6** | 创建 `evaluators/qwen.py`，将 `evaluate()` + `evaluate_with_comparison()` 封装为 `QwenEvaluator`。 | 低风险 |
| **Phase 7** | 更新顶层 `__init__.py`，统一 re-export。删除旧 flat 文件（或保留为 thin wrapper 一个版本）。 | 一次性切换 |

每个 Phase 完成后跑现有 CLI 命令验证：
```bash
PYTHONPATH=src python -m labeling_sft.dataset_builder
PYTHONPATH=src python -m labeling_sft.train --max_steps 5
PYTHONPATH=src python -m labeling_sft.evaluate --adapter_dir ... --max_samples 10
PYTHONPATH=src python -m labeling_sft.export --adapter_dir ... --output_dir ...
```

---

## 9. 设计决策记录

| 决策 | 理由 |
|------|------|
| ABC 而非 Protocol | ABC 提供 `__init_subclass__` 钩子，方便未来做注册表自动发现；且 `@abstractmethod` 的错误信息比 Protocol 的 `TypeError` 更明确 |
| 实现目录用复数名 | `trainers/` vs `interfaces/trainer.py` 在 import 路径上不冲突，IDE 自动补全也容易区分 |
| 一个实现 = 一个文件 | 方便 code review diff，方便 git blame 定位，避免单个大文件膨胀 |
| `compare()` 在 Evaluator 上而非独立函数 | 对比评估需要管理 GPU 显存（先卸载 base 再加载 adapter），这个生命周期管理是 Evaluator 的职责 |
| `format_name` 用 property 而非类属性 | 允许 Exporter 在 `__init__` 中根据参数动态决定格式名（如 `gguf_q4_0` vs `gguf_q8_0`） |
| `load_split()` 在 DatasetBuilder 上 | Trainer 和 Evaluator 都需要加载数据，把这个能力放在数据的「所有者」上，避免重复实现 JSONL 读取 |
| Contracts 用 frozen dataclass | 不可变，可哈希，适合作为缓存 key；类型提示完整，IDE 支持好 |
