#!/usr/bin/env python3
"""One-off bootstrap script for creating labeling and holdout manifests.

Run this once (or deliberately again with a new seed) before bootstrap
labeling. It is not a production runtime component and must not be used by the
online/auto-labeling pipeline.

The script deliberately does not inspect file names or recipe structure to
classify inputs. Every Markdown document is eligible. Non-recipes are useful
labels and remain in the labeling manifest for the model to return as
``not_a_recipe``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


from labeling.dedup.simhash_store import SimHashDedupStore
from labeling.interfaces.dedup_store import HashStatus


MARKDOWN_SUFFIXES = {".md", ".markdown"}


@dataclass(frozen=True)
class Document:
    source: str
    path: Path
    relative_path: Path
    fingerprint: str


@dataclass
class Cluster:
    id: int
    fingerprints: set[str] = field(default_factory=set)
    documents: list[Document] = field(default_factory=list)

    def document_for_source(self, source: str) -> Document | None:
        """Use a stable representative when a cluster has multiple variants."""
        matches = [document for document in self.documents if document.source == source]
        return min(matches, key=lambda document: str(document.relative_path)) if matches else None


def discover_documents(source_roots: dict[str, Path]) -> list[tuple[str, Path, Path]]:
    """Collect Markdown files only; intentionally make no eligibility decision."""
    documents: list[tuple[str, Path, Path]] = []
    for source, root in source_roots.items():
        if not root.is_dir():
            raise ValueError(f"source root does not exist or is not a directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES:
                documents.append((source, path.resolve(), path.relative_to(root)))
    if not documents:
        raise ValueError("no Markdown files found in the supplied source roots")
    return documents


def cluster_documents(documents: Iterable[tuple[str, Path, Path]], threshold: int) -> list[Cluster]:
    """Cluster exact/near duplicates with the production SimHash fingerprint.

    The block index mirrors ``SimHashDedupStore`` but stays in memory so sample
    construction does not register documents as already labeled.
    """
    # Reuse the production fingerprint algorithm, but keep this script's
    # index in memory. Sampling must not populate the labeling lifecycle DB.
    fingerprint_store = SimHashDedupStore(Path(":memory:"), threshold=threshold)
    clusters: list[Cluster] = []
    block_index: dict[tuple[int, int], set[int]] = defaultdict(set)

    try:
        for source, path, relative_path in documents:
            markdown = path.read_text(encoding="utf-8", errors="replace")
            fingerprint = fingerprint_store.recipe_card_hash(markdown)
            # Matching one of the four SimHash blocks is only a cheap
            # candidate lookup; the Hamming-distance check below decides if
            # two files are truly in the same near-duplicate cluster.
            candidates: set[int] = set()
            for block_id, block_value in enumerate(fingerprint_store._hash_blocks(fingerprint)):
                candidates.update(block_index[(block_id, block_value)])

            matching_cluster_id = next(
                (
                    cluster_id
                    for cluster_id in sorted(candidates)
                    if any(
                        fingerprint_store._hamming(fingerprint, existing) <= threshold
                        for existing in clusters[cluster_id].fingerprints
                    )
                ),
                None,
            )
            if matching_cluster_id is None:
                matching_cluster_id = len(clusters)
                clusters.append(Cluster(id=matching_cluster_id))

            cluster = clusters[matching_cluster_id]
            cluster.fingerprints.add(fingerprint)
            cluster.documents.append(Document(source, path, relative_path, fingerprint))
            # Index every variant too: a later document may match this
            # variant even where it does not match the cluster's first file.
            for block_id, block_value in enumerate(fingerprint_store._hash_blocks(fingerprint)):
                block_index[(block_id, block_value)].add(matching_cluster_id)
    finally:
        fingerprint_store._db.close()

    return clusters


def allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate counts with largest remainder, preserving the requested total."""
    raw = {source: total * ratio for source, ratio in ratios.items()}
    allocated = {source: int(value) for source, value in raw.items()}
    remaining = total - sum(allocated.values())
    for source in sorted(ratios, key=lambda name: (raw[name] - allocated[name], name), reverse=True)[:remaining]:
        allocated[source] += 1
    return allocated


def select_clusters(
    clusters: list[Cluster],
    quotas: dict[str, int],
    used_cluster_ids: set[int],
    rng: random.Random,
) -> list[Document]:
    """Sample one representative per cluster while enforcing source quotas."""
    # A cluster can contain files from both sources. It can contribute to only
    # one split and one source quota, preventing train/holdout leakage.
    by_source: dict[str, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        if cluster.id in used_cluster_ids:
            continue
        for source in quotas:
            if cluster.document_for_source(source) is not None:
                by_source[source].append(cluster)

    selected: list[Document] = []
    for source in sorted(quotas):
        candidates = by_source[source]
        rng.shuffle(candidates)
        chosen = 0
        for cluster in candidates:
            if cluster.id in used_cluster_ids:
                continue
            document = cluster.document_for_source(source)
            if document is None:
                continue
            selected.append(document)
            used_cluster_ids.add(cluster.id)
            chosen += 1
            if chosen == quotas[source]:
                break
        if chosen != quotas[source]:
            raise ValueError(
                f"not enough unique documents for source {source!r}: "
                f"need {quotas[source]}, selected {chosen}"
            )
    return selected


def manifest_record(document: Document, split: str, cluster_id: int) -> dict[str, str | int]:
    return {
        "id": f"{split}:{document.source}:{document.relative_path.as_posix()}",
        "source": document.source,
        "path": str(document.path),
        "relative_path": document.relative_path.as_posix(),
        "recipe_card_hash": document.fingerprint,
        "dedup_cluster": cluster_id,
        "split": split,
    }


def write_jsonl(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def assert_not_registered(records: list[dict[str, str | int]], db_path: Path, threshold: int) -> None:
    """Refuse to overwrite an existing labeling lifecycle record on rerun."""
    store = SimHashDedupStore(db_path, threshold=threshold)
    try:
        for record in records:
            fingerprint = str(record["recipe_card_hash"])
            if store.lookup(fingerprint) is not None:
                raise ValueError(
                    "selected document is already present in the persistent dedup store: "
                    f"{record['path']}"
                )
    finally:
        store._db.close()


def persist_labeling_manifest(
    records: list[dict[str, str | int]], db_path: Path, threshold: int
) -> None:
    """Register sampled labeling documents before they reach the model.

    ``IN_FLIGHT`` reserves each selected fingerprint across restarts and later
    production runs. The labeling orchestrator must transition it to
    ``ACCEPTED`` only after its structured JSONL record is durably saved.
    Holdout records are intentionally not registered here: they are not part
    of the bootstrap labeling run.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SimHashDedupStore(db_path, threshold=threshold)
    try:
        for record in records:
            store.register(
                str(record["recipe_card_hash"]),
                str(record["path"]),
                HashStatus.IN_FLIGHT,
            )
    finally:
        store._db.close()


def build_manifests(
    source_roots: dict[str, Path],
    ratios: dict[str, float],
    target: int,
    holdout: int,
    seed: int,
    threshold: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]], dict[str, object]]:
    documents = discover_documents(source_roots)
    clusters = cluster_documents(documents, threshold)
    cluster_by_document = {
        (document.source, document.path): cluster.id
        for cluster in clusters
        for document in cluster.documents
    }
    rng = random.Random(seed)
    used_cluster_ids: set[int] = set()

    # Reserve holdout first so its duplicate cluster cannot leak into prompt
    # development or training. `not_a_recipe` documents stay valid examples.
    holdout_documents = select_clusters(clusters, allocate(holdout, ratios), used_cluster_ids, rng)
    labeling_documents = select_clusters(clusters, allocate(target, ratios), used_cluster_ids, rng)
    holdout_records = [
        manifest_record(document, "holdout", cluster_by_document[(document.source, document.path)])
        for document in holdout_documents
    ]
    labeling_records = [
        manifest_record(document, "labeling", cluster_by_document[(document.source, document.path)])
        for document in labeling_documents
    ]
    report: dict[str, object] = {
        "seed": seed,
        "simhash_threshold": threshold,
        "total_documents": len(documents),
        "unique_clusters": len(clusters),
        "duplicates_removed": len(documents) - len(clusters),
        "ratios": ratios,
        "labeling_count": len(labeling_records),
        "holdout_count": len(holdout_records),
        "labeling_by_source": dict(Counter(record["source"] for record in labeling_records)),
        "holdout_by_source": dict(Counter(record["source"] for record in holdout_records)),
        "note": "No filename or structure filtering was applied; not_a_recipe is a retained label.",
    }
    return labeling_records, holdout_records, report
