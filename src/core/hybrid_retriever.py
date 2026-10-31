"""Qdrant-Style Universal Hybrid Retriever (Dense Two-Tower Cosine + Sparse BM25/SPLADE + RRF).

Stage 1 & Stage 2 of the NexusRec AI Recommendation Funnel:
1. Hard Metadata Payload Pre-Filtering (`in_stock`, `max_price`, `category`, `domain`,
   `required_compatibility`).
2. Parallel Dense + Sparse Retrieval:
   - Dense Vector Channel: 64-dim Two-Tower cosine similarity + 32-dim OpenCV visual query
     similarity when a visual reference is provided.
   - Sparse Lexical Channel: BM25 / SPLADE-style weighted term-importance scoring over SKU,
     title, technical attributes, tags, and compatibility graph.
3. Reciprocal Rank Fusion (RRF):
   `RRF_score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))`
   with `k = 60` (industry-standard Qdrant / Elasticsearch RRF constant).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.catalog import CatalogItem, MultimodalCatalogStore, UserPersona
from src.core.two_tower import TwoTowerRecommender

RRF_K = 60


@dataclass(frozen=True)
class PayloadFilter:
    """Qdrant-style structured metadata filter applied prior to vector + sparse scoring."""

    domain: str | None = None
    category: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    in_stock_only: bool = True
    required_tags: list[str] | None = None
    compatible_with_item_id: str | None = None

    def to_qdrant_filter_json(self) -> dict[str, Any]:
        """Serialize filter into Qdrant-compatible `must` payload condition structure."""
        must_clauses: list[dict[str, Any]] = []
        if self.in_stock_only:
            must_clauses.append({"key": "in_stock", "match": {"value": True}})
        if self.domain:
            must_clauses.append({"key": "domain", "match": {"value": self.domain}})
        if self.category:
            must_clauses.append({"key": "category", "match": {"value": self.category}})
        if self.max_price is not None or self.min_price is not None:
            range_cond: dict[str, float] = {}
            if self.max_price is not None:
                range_cond["lte"] = float(self.max_price)
            if self.min_price is not None:
                range_cond["gte"] = float(self.min_price)
            must_clauses.append({"key": "unit_price", "range": range_cond})
        if self.required_tags:
            must_clauses.append({"key": "tags", "match": {"any": self.required_tags}})
        if self.compatible_with_item_id:
            must_clauses.append(
                {"key": "compatible_with", "match": {"value": self.compatible_with_item_id}}
            )
        return {"must": must_clauses}


@dataclass(frozen=True)
class HybridCandidate:
    """Candidate item scored by Dense Two-Tower, Visual Descriptor, Sparse BM25, and RRF."""

    item: CatalogItem
    dense_two_tower_score: float
    visual_similarity_score: float
    combined_dense_score: float
    sparse_bm25_score: float
    dense_rank: int
    sparse_rank: int
    rrf_fusion_score: float
    matched_terms: list[str]


def _tokenize(text: str) -> list[str]:
    """Tokenize technical retail & industrial queries while preserving hyphenated specs."""
    cleaned = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", cleaned)
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        if "-" in tok:
            expanded.extend(part for part in tok.split("-") if len(part) > 1)
    return expanded


class UniversalHybridRetriever:
    """Executes Qdrant-style pre-filtering, BM25/SPLADE sparse scoring, and RRF fusion."""

    def __init__(
        self,
        catalog_store: MultimodalCatalogStore,
        two_tower: TwoTowerRecommender,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.catalog_store = catalog_store
        self.two_tower = two_tower
        self.k1 = k1
        self.b = b

        # Build inverted BM25 document index over all catalog items
        self._doc_tokens: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        items = self.catalog_store.items
        for item in items:
            attr_text = " ".join(f"{k} {v}" for k, v in item.attributes.items())
            tags_text = " ".join(item.tags * 2)  # Boost explicit taxonomy tags (SPLADE-style)
            title_text = (item.title + " ") * 2
            corpus_str = (
                f"{title_text} {item.sku} {item.category} {item.subcategory} "
                f"{item.brand} {item.description} {attr_text} {tags_text}"
            )
            tokens = _tokenize(corpus_str)
            counts = Counter(tokens)
            self._doc_tokens[item.item_id] = counts
            self._doc_lengths[item.item_id] = len(tokens)
            for term in counts:
                self._df[term] += 1

        total_len = sum(self._doc_lengths.values())
        self._avg_dl = total_len / max(len(items), 1)
        self._num_docs = len(items)

    def apply_payload_filter(self, payload_filter: PayloadFilter) -> list[CatalogItem]:
        """Filter catalog items using hard metadata constraints."""
        filtered: list[CatalogItem] = []
        for item in self.catalog_store.items:
            if payload_filter.in_stock_only and not item.in_stock:
                continue
            if payload_filter.domain and item.domain != payload_filter.domain:
                continue
            if payload_filter.category and item.category.lower() != payload_filter.category.lower():
                continue
            if payload_filter.max_price is not None and item.unit_price > payload_filter.max_price:
                continue
            if payload_filter.min_price is not None and item.unit_price < payload_filter.min_price:
                continue
            if payload_filter.compatible_with_item_id:
                if payload_filter.compatible_with_item_id not in item.compatible_with:
                    continue
            if payload_filter.required_tags:
                item_tag_set = {t.lower() for t in item.tags}
                if not any(rt.lower() in item_tag_set for rt in payload_filter.required_tags):
                    continue
            filtered.append(item)
        return filtered

    def score_bm25(
        self,
        query_text: str,
        candidates: list[CatalogItem],
        extra_boost_terms: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Compute BM25 + SPLADE term expansion lexical scores for candidate items."""
        primary_tokens = _tokenize(query_text)
        boost_tokens = _tokenize(" ".join(extra_boost_terms)) if extra_boost_terms else []

        stop_words = {
            "the", "and", "for", "with", "that", "under", "below", "less", "than",
            "from", "into", "need", "find", "show", "looking", "pairs", "pair", "usd",
        }
        q_primary = [t for t in primary_tokens if t not in stop_words and len(t) > 1]
        q_boost = [t for t in boost_tokens if t not in stop_words and len(t) > 1 and t not in q_primary]

        scores: dict[str, float] = {}
        matched_map: dict[str, list[str]] = {}

        for item in candidates:
            doc_counts = self._doc_tokens.get(item.item_id, Counter())
            dl = self._doc_lengths.get(item.item_id, 1)
            score = 0.0
            matched: list[str] = []

            for term in q_primary:
                tf = doc_counts.get(term, 0)
                if tf == 0:
                    continue
                if term not in matched:
                    matched.append(term)
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_dl))
                score += 1.5 * idf * ((tf * (self.k1 + 1.0)) / denom)

            for term in q_boost:
                tf = doc_counts.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_dl))
                score += 0.25 * idf * ((tf * (self.k1 + 1.0)) / denom)

            scores[item.item_id] = round(score, 5)
            matched_map[item.item_id] = matched

        return scores, matched_map

    def retrieve(
        self,
        persona: UserPersona,
        query_text: str,
        payload_filter: PayloadFilter,
        query_visual_vec: np.ndarray | None = None,
        query_semantic_vec: np.ndarray | None = None,
        top_k: int = 20,
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
    ) -> dict[str, Any]:
        """Run Stage 1 (Filter + Two-Tower Candidate Generation) and Stage 2 (Hybrid RRF)."""
        filtered_items = self.apply_payload_filter(payload_filter)
        # Fallback if a very strict category filter yielded 0 items: relax category while keeping domain & max_price
        relaxed_filter_used = False
        if not filtered_items and payload_filter.category is not None:
            relaxed_filter = PayloadFilter(
                domain=payload_filter.domain,
                category=None,
                max_price=payload_filter.max_price,
                min_price=payload_filter.min_price,
                in_stock_only=payload_filter.in_stock_only,
            )
            filtered_items = self.apply_payload_filter(relaxed_filter)
            relaxed_filter_used = True

        user_emb, two_tower_scores = self.two_tower.score_candidates(
            persona=persona,
            query_visual_vec=query_visual_vec,
            query_semantic_vec=query_semantic_vec,
            candidate_items=filtered_items,
        )

        # Compute direct visual cosine similarity when visual query vector is supplied
        visual_scores: dict[str, float] = {}
        combined_dense: dict[str, float] = {}
        for item in filtered_items:
            tt_score = two_tower_scores[item.item_id]
            if query_visual_vec is not None and len(query_visual_vec) == 32:
                q_vis = np.asarray(query_visual_vec, dtype=np.float32)
                q_vis_norm = q_vis / max(float(np.linalg.norm(q_vis)), 1e-8)
                item_vis = np.asarray(item.visual_embedding, dtype=np.float32)
                vis_sim = float(np.dot(q_vis_norm, item_vis))
                visual_scores[item.item_id] = round(vis_sim, 4)
                combined_dense[item.item_id] = round(0.60 * tt_score + 0.40 * vis_sim, 5)
            else:
                # Default visual baseline is similarity to user's recent session visual profile
                item_vis = np.asarray(item.visual_embedding, dtype=np.float32)
                u_vis = user_emb[:32] / max(float(np.linalg.norm(user_emb[:32])), 1e-8)
                vis_sim = float(np.dot(u_vis, item_vis))
                visual_scores[item.item_id] = round(vis_sim, 4)
                combined_dense[item.item_id] = round(tt_score, 5)

        bm25_scores, matched_terms_map = self.score_bm25(
            query_text=query_text,
            candidates=filtered_items,
            extra_boost_terms=persona.intent_keywords,
        )

        # Rank by Dense and by Sparse independently
        dense_sorted = sorted(
            filtered_items, key=lambda x: combined_dense[x.item_id], reverse=True
        )
        sparse_sorted = sorted(
            filtered_items,
            key=lambda x: (bm25_scores[x.item_id], combined_dense[x.item_id]),
            reverse=True,
        )

        dense_ranks = {item.item_id: rank + 1 for rank, item in enumerate(dense_sorted)}
        sparse_ranks = {item.item_id: rank + 1 for rank, item in enumerate(sparse_sorted)}

        # Compute Reciprocal Rank Fusion (RRF)
        hybrid_candidates: list[HybridCandidate] = []
        for item in filtered_items:
            d_rank = dense_ranks[item.item_id]
            s_rank = sparse_ranks[item.item_id]
            rrf_raw = (dense_weight / (RRF_K + d_rank)) + (sparse_weight / (RRF_K + s_rank))
            # Normalize RRF to an intuitive [0, 1] scale relative to Rank-1 ideal score
            ideal_rrf = (dense_weight / (RRF_K + 1)) + (sparse_weight / (RRF_K + 1))
            rrf_normalized = rrf_raw / max(ideal_rrf, 1e-8)

            hybrid_candidates.append(
                HybridCandidate(
                    item=item,
                    dense_two_tower_score=round(two_tower_scores[item.item_id], 4),
                    visual_similarity_score=visual_scores[item.item_id],
                    combined_dense_score=round(combined_dense[item.item_id], 4),
                    sparse_bm25_score=round(bm25_scores[item.item_id], 4),
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    rrf_fusion_score=round(rrf_normalized, 4),
                    matched_terms=matched_terms_map.get(item.item_id, []),
                )
            )

        hybrid_candidates.sort(
            key=lambda c: (c.rrf_fusion_score, c.combined_dense_score), reverse=True
        )
        top_candidates = hybrid_candidates[:top_k]

        return {
            "user_embedding_64d": [round(float(x), 4) for x in user_emb.tolist()],
            "qdrant_payload_filter": payload_filter.to_qdrant_filter_json(),
            "relaxed_filter_used": relaxed_filter_used,
            "total_catalog_size": len(self.catalog_store.items),
            "filtered_candidate_count": len(filtered_items),
            "retrieved_count": len(top_candidates),
            "candidates": top_candidates,
        }
