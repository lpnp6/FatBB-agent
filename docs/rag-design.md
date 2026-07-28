# RAG 模块设计

## 1. 目标与边界

本模块为上层业务提供可替换、可追溯的知识检索能力，并将检索到的证据整理为可供 LLM 使用的上下文。它支持以下检索形态：

- 文本检索：关键词（BM25）或向量相似度检索文档分块；
- 图检索：从实体、关系和限定条件出发检索节点、边或关系路径；
- 混合检索：合并多个检索器的结果，并可选地重排。

第一阶段不负责模型调用、对话记忆、采集原始资料或构建图谱。这些职责分别属于生成服务、会话服务、数据接入层和索引构建器。RAG 模块的输入是已索引的知识，输出是带来源和分数的证据。

## 2. 设计原则

1. 以证据而不是文档为检索返回值。文本块、图节点、边和路径都可以成为回答依据。
2. 以接口隔离基础设施。业务代码不依赖 Chroma、pgvector、Neo4j 或任意 Embedding 提供商。
3. 保留可追溯性。每条证据必须可以关联到原始文档、URI 或图谱来源。
4. 显式建模图谱。图节点、边和路径不只存入通用 `metadata`，以保留可查询和可解释的关系结构。
5. 写入与查询分离。`Retriever` 是只读接口；切分、Embedding、建索引、删除由独立的 `Indexer` 或具体存储实现负责。
6. 结果与上下文分离。检索结果保留完整结构；上下文构建器依据 token 预算和去重策略生成 Prompt 文本。

## 3. 模块结构

```text
src/rag/
├── models/
│   ├── common.py          # SourceRef、Metadata 等通用值对象
│   ├── document.py        # Document、TextChunk
│   ├── graph.py           # GraphNode、GraphEdge、GraphPath
│   ├── query.py           # RetrievalQuery、Filter
│   ├── evidence.py        # Evidence、EvidenceKind
│   └── context.py         # RAGContext
├── interfaces/
│   ├── retriever.py       # 统一只读检索接口
│   ├── stores.py          # 文本、向量、图存储端口
│   ├── embedding.py       # 向量化端口
│   ├── indexer.py         # 可选：知识写入/删除/重建接口
│   └── reranker.py        # 可选：重排接口
├── stores/
│   ├── memory_store.py
│   ├── chroma_store.py
│   ├── pgvector_store.py
│   └── neo4j_store.py
├── retrievers/
│   ├── memory_retriever.py
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   ├── graph_retriever.py
│   └── hybrid_retriever.py
├── context_builder.py
└── service.py             # 可选的 RAG 编排服务
```

第一阶段实现 `models/`、存储端口、`Retriever`、`BM25Retriever` 与 `PostgresTextChunkStore`。PostgreSQL 的 `pg_search` 在数据库内完成 BM25 评分、排序和 `top_k` 截断；`BM25Retriever` 只负责将结果转换为统一 `Evidence`。外部向量库和图数据库适配器保持为后续迭代。

其中 `retrievers/` 不能直接 import 某个数据库 SDK。存储实现集中在 `stores/`，由应用装配层注入对应的接口实现。

## 4. 领域模型

### 4.1 来源与原始资料

```python
@dataclass(frozen=True)
class SourceRef:
    document_id: str | None = None
    uri: str | None = None
    title: str | None = None
    locator: str | None = None  # 页码、章节、段落或文件偏移


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    source: SourceRef
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    content: str
    index: int
    source: SourceRef
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

`Document` 是原始资料，通常不直接参与召回；`TextChunk` 是文本检索的最小索引单位。`id` 应由接入层稳定生成，重建索引不能改变其含义。

### 4.2 图谱模型

```python
@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    properties: dict[str, object] = field(default_factory=dict)
    source: SourceRef | None = None


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source_node_id: str
    target_node_id: str
    relation: str
    properties: dict[str, object] = field(default_factory=dict)
    source: SourceRef | None = None


@dataclass(frozen=True)
class GraphPath:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
```

图检索必须能返回 `GraphPath`。单一实体只能回答“有什么”，路径才能回答“为什么有关”：例如 `Recipe -[CONTAINS]-> Ingredient -[BELONGS_TO]-> Category`。

### 4.3 查询模型

```python
RetrievalMode = Literal["keyword", "vector", "graph", "hybrid"]


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 5
    mode: RetrievalMode = "hybrid"
    filters: dict[str, object] = field(default_factory=dict)

    # 图检索可选约束；非图检索器可安全忽略
    entity_ids: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    max_hops: int = 2
```

`filters` 只承载平面、可序列化的业务筛选条件，例如 `source`、`tenant_id`、`category`、`language` 和权限范围。复杂布尔表达式或图遍历约束以后可扩展为专门的 Filter AST，第一阶段无需预先设计。

### 4.4 统一证据模型

```python
EvidenceKind = Literal["text_chunk", "graph_node", "graph_edge", "graph_path"]


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: EvidenceKind
    content: str                 # 可直接送入 LLM 上下文的文本形式
    score: float
    source: SourceRef | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    chunk: TextChunk | None = None
    node: GraphNode | None = None
    edge: GraphEdge | None = None
    path: GraphPath | None = None
```

约束：`kind == "text_chunk"` 时 `chunk` 必须存在；图类型同理。构造校验可放在 `__post_init__`，防止出现证据类型和载荷不一致的情况。

不同检索系统的分数不可直接比较。因此 `metadata` 应保存 `retriever`、`raw_score`、`rank` 和可选的 `score_type`；混合检索器生成统一的融合分数并将其放入 `score`。

### 4.5 最终上下文

```python
@dataclass(frozen=True)
class RAGContext:
    query: str
    evidences: tuple[Evidence, ...]
    text: str
    truncated: bool = False
```

`RAGContext` 是唯一直接交给 Prompt 层的模型。模型回答的引用应使用 `Evidence.id`，从而能够回溯到原始文档或图关系。

## 5. 接口设计

### 5.1 Retriever

```python
class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
        """返回按 score 降序排列、且数量不超过 query.top_k 的证据。"""
```

所有实现必须遵守：

- 空查询或无匹配返回空列表；
- 返回结果按最终分数降序；
- 应用其支持的筛选条件，不能因筛选不支持而返回越权数据；
- 每个结果写入实现名称到 `metadata["retriever"]`；
- 不在接口内修改索引或调用 LLM。

### 5.2 存储抽象

`Retriever` 不拥有数据库连接，也不执行 SDK 查询。它依赖下列存储端口，将“如何查询数据库”与“如何把结果解释为回答证据”分离：

```python
@dataclass(frozen=True)
class VectorMatch:
    chunk: TextChunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[VectorMatch]:
        """返回已完成 metadata 权限过滤的相似文本块。"""


class GraphStore(ABC):
    @abstractmethod
    def find_nodes(
        self,
        *,
        entity_ids: Sequence[str],
        text: str,
        filters: Mapping[str, object],
        top_k: int,
    ) -> list[GraphNode]: ...

    @abstractmethod
    def traverse(
        self,
        *,
        start_node_ids: Sequence[str],
        relation_types: Sequence[str],
        max_hops: int,
        filters: Mapping[str, object],
    ) -> list[GraphPath]: ...
```

另设 `EmbeddingProvider`，防止向量检索器同时耦合存储厂商和 Embedding 厂商：

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> Sequence[float]: ...
```

实现关系如下：

```text
VectorRetriever --uses--> EmbeddingProvider --adapter--> OpenAI / 本地模型
       |
       +--uses--> VectorStore --adapter--> Chroma / pgvector / Milvus

GraphRetriever ---uses--> GraphStore ---adapter--> Neo4j / Kùzu / 内存图
```

`VectorStore` 和 `GraphStore` 共享“存储端口”的架构位置，但不应强行继承一个只有 `search()` 的大而全接口：向量相似搜索和图遍历具有不同的输入、输出与约束。若未来出现公共生命周期操作，可只抽取窄接口：

```python
class HealthCheckable(Protocol):
    def ping(self) -> None: ...
```

这样既能替换底层数据库，也不会损失图查询的表达力。

### 5.3 可选 Indexer

```python
class Indexer(ABC):
    @abstractmethod
    def upsert_documents(self, documents: Sequence[Document]) -> None: ...

    @abstractmethod
    def delete_documents(self, document_ids: Sequence[str]) -> None: ...
```

文本实现内部可执行分块和向量化，再调用 `VectorStore` 写入；图实现内部可抽取实体关系，再调用 `GraphStore` 写入。写接口可依照实际存储能力定义为 `upsert_chunks`、`upsert_nodes`、`upsert_edges`，不必为了统一而放进 `Retriever`。

### 5.4 可选 Reranker

```python
class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[Evidence], *, top_k: int
    ) -> list[Evidence]: ...
```

Reranker 是可选依赖。它仅改变候选证据的排序和分数，不负责从底层索引扩大召回范围。

## 6. 检索实现与演进

### 6.1 MemoryRetriever

用于单元测试和本地开发。实现简单的 token 重合度和 metadata 筛选。它是接口的参考实现，不追求召回质量。

### 6.2 VectorRetriever

职责：通过 `EmbeddingProvider` 生成查询向量，调用 `VectorStore.search()`，并将 `TextChunk` 转换为 `Evidence`。它不认识向量库 SDK；向量库客户端和 Embedding 客户端都由构造函数注入的适配器封装。

### 6.2.1 第一阶段 PostgreSQL + BM25

第一版使用 PostgreSQL 保存 `rag_text_chunks`，并为 `document_id` 与 JSONB `metadata` 建立索引。`PostgresTextChunkStore` 同时实现 `BM25SearchStore`：它使用 `pg_search` 的 BM25 索引执行匹配，使用 `paradedb.score()` 取得数据库计算的分数，并在数据库侧完成 `metadata @> filter`、排序和 `LIMIT top_k`。

数据库对象由版本化 SQL migration [migrations/postgres/0001_create_rag_text_chunks.sql](../migrations/postgres/0001_create_rag_text_chunks.sql) 在部署阶段一次性执行，而不是由应用启动创建：

```bash
psql "$DATABASE_URL" -f migrations/postgres/0001_create_rag_text_chunks.sql
```

应用运行时不得执行 DDL；`PostgresTextChunkStore` 仅执行 chunk 的读写和 BM25 查询。

`BM25Retriever` 不再包含分词、TF/DF 统计或 BM25 公式；它只校验非空查询、调用 `search_bm25()`，再将 `ScoredTextChunk` 包装为 `Evidence`。若未来替换搜索引擎，只需实现相同的 `BM25SearchStore` 端口，不影响 `Retriever`、`Evidence` 或业务调用代码。

### 6.3 GraphRetriever

职责：从查询中识别实体或接收显式 `entity_ids`，通过 `GraphStore.find_nodes()` 和 `GraphStore.traverse()` 按 `relation_types`、`max_hops` 与权限筛选遍历子图。它可返回节点、边和路径；路径默认转为简明的关系三元组文本，避免把整个子图塞入上下文。

### 6.4 HybridRetriever

并行执行文本检索和图检索，完成以下处理：

1. 对每个来源独立召回候选集；
2. 根据 `Evidence.id` 或相同来源消除重复；
3. 使用 Reciprocal Rank Fusion（RRF）作为第一版融合策略；
4. 可选调用 `Reranker`；
5. 返回最终 `top_k` 条证据。

RRF 不要求不同检索器的原始分数具有相同尺度，适合第一版混合检索：

```text
score(evidence) = Σ 1 / (k + rank_i)
```

其中 `k` 可先取 60。后续有离线标注数据后，再替换为学习排序或加权融合。

## 7. 上下文构建

`ContextBuilder` 接收 `RetrievalQuery` 和 `list[Evidence]`，按如下顺序生成 `RAGContext`：

1. 按分数去重；同一文档的相邻文本块可合并；
2. 为每条证据生成带引用 ID 的片段，例如 `[E:text:123] ...`；
3. 估算 token 数，保留高分且来源多样的证据；
4. 达到预算时截断，并标记 `truncated=True`；
5. 不修改原始 `Evidence`。

默认不在 Retriever 中做 Prompt 拼接，否则会使检索器难以单测、重用和观察。

## 8. 调用流程

```text
用户问题
  -> RetrievalQuery
  -> Retriever（vector / graph / hybrid）
  -> list[Evidence]
  -> 可选 Reranker
  -> ContextBuilder
  -> RAGContext
  -> 上层 Prompt / LLM 服务
  -> 带 Evidence.id 引用的回答
```

索引流程独立运行：

```text
原始资料 -> Document -> Indexer
                    -> TextChunk -> 关键词 / 向量索引
                    -> 实体关系 -> 图索引
```

## 9. 测试策略

模型与接口测试：

- `Evidence.kind` 与结构化载荷一致性；
- `SourceRef` 在分块、图路径和证据转换中不丢失；
- `RetrievalQuery.top_k`、空查询和筛选条件边界。

Retriever 契约测试：

- 结果不超过 `top_k`，按分数降序；
- 无匹配时返回空列表；
- 筛选条件不会泄露不符合条件的数据；
- 每条结果包含来源、实现标识和稳定 ID。

实现测试：

- `MemoryRetriever`：关键词匹配、筛选、排序；
- `HybridRetriever`：RRF 融合、跨来源去重、部分检索器无结果；
- `GraphRetriever`：最大跳数、关系类型限制、路径序列化；
- `ContextBuilder`：token 预算、引用编号、相邻分块合并和截断标识。

## 10. 观测与安全

每次请求至少记录：查询模式、各检索器耗时、候选数量、最终证据 ID、分数、截断状态和失败原因。日志不得记录超出允许范围的原始敏感内容。

权限和租户过滤必须在底层检索阶段执行，`ContextBuilder` 的过滤只能作为防御性补充，不能成为唯一边界。

## 11. 实施顺序

1. 创建模型、`Retriever`、`TextChunkStore`、`BM25Retriever` 与 PostgreSQL 适配器，完成契约测试；
2. 实现 `ContextBuilder`，让上层服务可将 BM25 证据注入提示词；
3. 接入一个向量库适配器和 Embedding 适配器，完成 `VectorRetriever`；
4. 引入 `Indexer`，形成稳定的文本索引流水线；
5. 接入图数据库适配器和 `GraphRetriever`；
6. 实现 `HybridRetriever` + RRF，再按质量需求加入重排器。

该顺序先验证稳定的领域边界，再引入外部服务，能避免后续从向量 RAG 扩展到图 RAG 时重写业务调用代码。
