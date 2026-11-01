"""Deterministic RecSys Offline & Online Benchmark Evaluation Suite.

Computes industry-standard Information Retrieval & Recommendation Systems metrics across
4 benchmark buyer cohorts (`alpine_mountaineer`, `plant_reliability_lead`,
`cnc_automation_architect`, `ultralight_trail_buyer`):
- `NDCG@10` (Normalized Discounted Cumulative Gain at K=10)
- `Recall@10`
- `Precision@10`
- `MRR` (Mean Reciprocal Rank)
- `Catalog Coverage (%)` across the 44-item enterprise catalog
- `Intra-List Diversity (ILD)` (mean pairwise feature/category dissimilarity)
- `Expected Gross Margin Lift ($)`

Compares 3 distinct recommender architectures:
1. `Baseline Popularity`: Ranks in-stock items purely by historical `popularity_score`.
2. `Pure Collaborative Filtering (Matrix Co-Occurrence)`: Ranks items by session item-to-item
   compatibility & co-occurrence cosine without visual/lexical fusion or MMR diversification.
3. `NexusRec Two-Tower + Hybrid RRF + Cross-Encoder + MMR`: Full 4-stage pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.catalog import CatalogItem, MultimodalCatalogStore
from src.core.nlp_assistant import ConversationalCommerceAssistant
from src.core.reranker import CrossEncoderAndMMRReranker


@dataclass(frozen=True)
class BenchmarkScenario:
    """Ground-truth benchmark query scenario for a buyer persona."""

    persona_id: str
    query: str
    relevant_item_grades: dict[str, int]  # Graded relevance: 3 = exact target, 2 = strong match, 1 = complementary


BENCHMARK_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        persona_id="alpine_mountaineer",
        query="waterproof alpine shell under $220 that pairs with crampon boots",
        relevant_item_grades={
            "RET-001": 3,
            "RET-002": 3,
            "RET-021": 3,
            "RET-006": 3,
            "RET-017": 3,
            "RET-013": 3,
            "RET-009": 2,
            "RET-010": 2,
            "RET-020": 2,
            "RET-004": 2,
            "RET-014": 2,
            "RET-022": 2,
            "RET-019": 2,
            "RET-015": 1,
            "RET-007": 1,
        },
    ),
    BenchmarkScenario(
        persona_id="plant_reliability_lead",
        query="high-pressure hydraulic seal replacement under $150",
        relevant_item_grades={
            "IND-001": 3,
            "IND-002": 3,
            "IND-006": 3,
            "IND-017": 3,
            "IND-003": 3,
            "IND-008": 3,
            "IND-022": 3,
            "IND-018": 2,
            "IND-010": 2,
            "IND-015": 2,
            "IND-011": 2,
            "IND-009": 2,
            "IND-021": 1,
        },
    ),
    BenchmarkScenario(
        persona_id="cnc_automation_architect",
        query="precision ceramic CNC spindle bearing and vibration telemetry under $250",
        relevant_item_grades={
            "IND-009": 3,
            "IND-018": 3,
            "IND-011": 3,
            "IND-012": 3,
            "IND-021": 3,
            "IND-020": 3,
            "IND-010": 2,
            "IND-017": 2,
            "IND-003": 2,
            "IND-001": 2,
            "IND-006": 2,
            "IND-008": 2,
            "IND-015": 2,
            "IND-002": 1,
            "IND-019": 1,
            "IND-022": 1,
        },
    ),
    BenchmarkScenario(
        persona_id="ultralight_trail_buyer",
        query="ultralight packable waterproof shell and breathable merino midlayer under $200",
        relevant_item_grades={
            "RET-004": 3,
            "RET-019": 3,
            "RET-011": 3,
            "RET-001": 3,
            "RET-018": 3,
            "RET-009": 3,
            "RET-010": 3,
            "RET-012": 2,
            "RET-015": 2,
            "RET-020": 2,
            "RET-007": 2,
            "RET-013": 2,
            "RET-022": 2,
            "RET-014": 1,
            "RET-016": 1,
        },
    ),
]


def compute_ndcg_at_k(ranked_item_ids: list[str], graded_relevance: dict[str, int], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (`NDCG@K`) with exponential gain `2^rel - 1`."""
    top_k_ids = ranked_item_ids[:k]
    dcg = 0.0
    for idx, item_id in enumerate(top_k_ids):
        rel = graded_relevance.get(item_id, 0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(idx + 2.0)

    ideal_rels = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = sum((2.0**r - 1.0) / math.log2(idx + 2.0) for idx, r in enumerate(ideal_rels) if r > 0)
    if idcg <= 1e-9:
        return 0.0
    return round(dcg / idcg, 4)


def compute_recall_at_k(ranked_item_ids: list[str], graded_relevance: dict[str, int], k: int = 10) -> float:
    """Compute fraction of relevant items (`rel > 0`) recovered in top-K."""
    relevant_set = {iid for iid, rel in graded_relevance.items() if rel > 0}
    if not relevant_set:
        return 0.0
    hits = sum(1 for iid in ranked_item_ids[:k] if iid in relevant_set)
    return round(hits / len(relevant_set), 4)


def compute_precision_at_k(ranked_item_ids: list[str], graded_relevance: dict[str, int], k: int = 10) -> float:
    """Compute fraction of top-K recommended items that are relevant (`rel > 0`)."""
    top_k_ids = ranked_item_ids[:k]
    if not top_k_ids:
        return 0.0
    relevant_set = {iid for iid, rel in graded_relevance.items() if rel > 0}
    hits = sum(1 for iid in top_k_ids if iid in relevant_set)
    return round(hits / len(top_k_ids), 4)


def compute_mrr(ranked_item_ids: list[str], graded_relevance: dict[str, int]) -> float:
    """Compute Reciprocal Rank of the first highly relevant item (`rel >= 2`)."""
    for idx, iid in enumerate(ranked_item_ids):
        if graded_relevance.get(iid, 0) >= 2:
            return round(1.0 / (idx + 1.0), 4)
    return 0.0


def compute_position_propensity(rank_k: int) -> float:
    """Examination propensity `p(rank_k) = 1 / log2(rank_k + 2)` (1-indexed rank)."""
    return float(1.0 / math.log2(max(1, rank_k) + 2.0))


def compute_ips_and_snips(
    ranked_item_ids: list[str],
    graded_relevance: dict[str, int],
    logging_item_ids: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute Counterfactual Inverse Propensity Scoring (`IPS`) & Self-Normalized IPS (`SNIPS`).

    Corrects for rank-position exposure bias in historical logging policy `pi_0`:
    - `p_logging(i)` is determined by item `i`'s rank under the historical logging policy
      `p(rank) = 1 / log2(rank + 2)`.
    - `p_target(i)` is the target policy exposure weight at the new rank position.
    - Importance weight `w_i = min(5.0, p_target(i) / p_logging(i))` (clipped for variance control).
    - `IPS = (1 / K) * sum_{i in top_K} w_i * r_i`
    - `SNIPS = (sum_{i in top_K} w_i * r_i) / (sum_{i in top_K} w_i)`
    """
    top_k = ranked_item_ids[:k]
    if not top_k:
        return {"ips_score": 0.0, "snips_score": 0.0, "effective_sample_size": 0.0}

    logging_ranks = {iid: idx + 1 for idx, iid in enumerate(logging_item_ids)}
    default_unseen_rank = len(logging_item_ids) + 8

    weighted_rewards: list[float] = []
    weights: list[float] = []

    for idx, iid in enumerate(top_k):
        target_rank = idx + 1
        log_rank = logging_ranks.get(iid, default_unseen_rank)

        p_target = compute_position_propensity(target_rank)
        p_log = compute_position_propensity(log_rank)

        # Clipped importance weight w_i = min(5.0, p_target / p_log)
        w_i = min(5.0, p_target / max(p_log, 1e-6))
        # Normalized graded reward in [0.0, 1.0]
        r_i = min(1.0, graded_relevance.get(iid, 0) / 3.0)

        weights.append(w_i)
        weighted_rewards.append(w_i * r_i)

    sum_w = sum(weights)
    sum_wr = sum(weighted_rewards)
    ips = sum_wr / len(top_k)
    snips = sum_wr / max(sum_w, 1e-8)
    # Kish's Effective Sample Size (ESS) = (sum w)^2 / sum(w^2)
    sum_w_sq = sum(w * w for w in weights)
    ess = (sum_w * sum_w) / max(sum_w_sq, 1e-8)

    return {
        "ips_score": round(float(ips), 4),
        "snips_score": round(float(snips), 4),
        "effective_sample_size": round(float(ess), 2),
    }


class RecSysBenchmarkEvaluator:
    """Runs head-to-head offline benchmark evaluation across Popularity, Pure CF, and NexusRec."""

    def __init__(
        self,
        catalog_store: MultimodalCatalogStore,
        assistant: ConversationalCommerceAssistant,
    ) -> None:
        self.catalog_store = catalog_store
        self.assistant = assistant

    def _rank_baseline_popularity(self, k: int = 10) -> list[CatalogItem]:
        in_stock = [it for it in self.catalog_store.items if it.in_stock]
        in_stock.sort(key=lambda x: (x.popularity_score, x.gross_margin_pct), reverse=True)
        return in_stock[:k]

    def _rank_pure_collaborative_filtering(self, persona_id: str, k: int = 10) -> list[CatalogItem]:
        """Rank items using session item-to-item co-occurrence without price or lexical filtering."""
        persona = self.catalog_store.get_persona(persona_id)
        recent_items = [
            self.catalog_store.get_item(iid)
            for iid in persona.recent_item_ids
            if self.catalog_store.get_item(iid) is not None
        ]
        recent_cats = {it.category for it in recent_items if it is not None}
        recent_ids = set(persona.recent_item_ids)

        scored: list[tuple[float, CatalogItem]] = []
        for item in self.catalog_store.items:
            if not item.in_stock:
                continue
            # Pure CF relies on historical item-item co-occurrence and category affinity,
            # without natural-language price ceiling parsing or BM25 lexical matching
            compat_hits = sum(1 for cid in item.compatible_with if cid in recent_ids)
            same_cat_bonus = 0.55 if item.category in recent_cats else 0.0
            cf_score = 0.25 * compat_hits + same_cat_bonus + 0.45 * item.popularity_score
            scored.append((cf_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair[1] for pair in scored[:k]]

    def _rank_nexusrec_pipeline(self, scenario: BenchmarkScenario, k: int = 10) -> list[CatalogItem]:
        out = self.assistant.execute_funnel(
            query=scenario.query,
            persona_id=scenario.persona_id,
            top_k=k,
            lambda_diversity=0.92,
            bandit_exploration_weight=0.04,
        )
        items: list[CatalogItem] = []
        for rec in out["recommendations"]:
            item_obj = self.catalog_store.get_item(rec["item"]["item_id"])
            if item_obj is not None:
                items.append(item_obj)
        return items

    def evaluate_all_models(self, k: int = 10) -> dict[str, Any]:
        """Evaluate Baseline Popularity, Pure CF, and NexusRec Two-Tower + MMR across all scenarios."""
        total_in_stock = sum(1 for it in self.catalog_store.items if it.in_stock)
        models_Spec = [
            ("baseline_popularity", "Baseline Popularity Ranker"),
            ("pure_collaborative_filtering", "Pure Collaborative Filtering (Matrix Co-Occurrence)"),
            ("nexusrec_two_tower_mmr", "NexusRec Two-Tower + Hybrid RRF + Cross-Encoder + MMR"),
        ]

        model_summaries: list[dict[str, Any]] = []
        per_cohort_breakdown: dict[str, dict[str, Any]] = {}
        logging_baseline_ids = [it.item_id for it in self._rank_baseline_popularity(k=25)]

        for model_key, model_label in models_Spec:
            ndcgs: list[float] = []
            recalls: list[float] = []
            precisions: list[float] = []
            mrrs: list[float] = []
            ilds: list[float] = []
            margins: list[float] = []
            ips_list: list[float] = []
            snips_list: list[float] = []
            unique_surfaced_ids: set[str] = set()
            cohort_details: list[dict[str, Any]] = []

            for scenario in BENCHMARK_SCENARIOS:
                if model_key == "baseline_popularity":
                    ranked_items = self._rank_baseline_popularity(k=k)
                elif model_key == "pure_collaborative_filtering":
                    ranked_items = self._rank_pure_collaborative_filtering(scenario.persona_id, k=k)
                else:
                    ranked_items = self._rank_nexusrec_pipeline(scenario, k=k)

                ranked_ids = [it.item_id for it in ranked_items]
                unique_surfaced_ids.update(ranked_ids)

                ndcg = compute_ndcg_at_k(ranked_ids, scenario.relevant_item_grades, k=k)
                rec = compute_recall_at_k(ranked_ids, scenario.relevant_item_grades, k=k)
                prec = compute_precision_at_k(ranked_ids, scenario.relevant_item_grades, k=k)
                mrr = compute_mrr(ranked_ids, scenario.relevant_item_grades)
                ild = CrossEncoderAndMMRReranker.compute_intra_list_diversity(ranked_items)
                cf_eval = compute_ips_and_snips(
                    ranked_item_ids=ranked_ids,
                    graded_relevance=scenario.relevant_item_grades,
                    logging_item_ids=logging_baseline_ids,
                    k=k,
                )
                avg_margin = float(
                    np.mean([it.unit_price * (it.gross_margin_pct / 100.0) for it in ranked_items])
                ) if ranked_items else 0.0

                ndcgs.append(ndcg)
                recalls.append(rec)
                precisions.append(prec)
                mrrs.append(mrr)
                ilds.append(ild)
                margins.append(avg_margin)
                ips_list.append(cf_eval["ips_score"])
                snips_list.append(cf_eval["snips_score"])

                cohort_details.append(
                    {
                        "persona_id": scenario.persona_id,
                        "query": scenario.query,
                        "ndcg_at_10": ndcg,
                        "recall_at_10": rec,
                        "precision_at_10": prec,
                        "mrr": mrr,
                        "ips_score": cf_eval["ips_score"],
                        "snips_score": cf_eval["snips_score"],
                        "intra_list_diversity": ild,
                        "top_3_skus": [it.sku for it in ranked_items[:3]],
                    }
                )

            coverage_pct = round((len(unique_surfaced_ids) / max(total_in_stock, 1)) * 100.0, 2)
            summary = {
                "model_id": model_key,
                "model_name": model_label,
                "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
                "recall_at_10": round(float(np.mean(recalls)), 4),
                "precision_at_10": round(float(np.mean(precisions)), 4),
                "mrr": round(float(np.mean(mrrs)), 4),
                "ips_score": round(float(np.mean(ips_list)), 4),
                "snips_score": round(float(np.mean(snips_list)), 4),
                "catalog_coverage_pct": coverage_pct,
                "intra_list_diversity": round(float(np.mean(ilds)), 4),
                "avg_unit_margin_usd": round(float(np.mean(margins)), 2),
                "cohorts": cohort_details,
            }
            model_summaries.append(summary)
            per_cohort_breakdown[model_key] = summary

        pop_ndcg = per_cohort_breakdown["baseline_popularity"]["ndcg_at_10"]
        cf_ndcg = per_cohort_breakdown["pure_collaborative_filtering"]["ndcg_at_10"]
        nexus_ndcg = per_cohort_breakdown["nexusrec_two_tower_mmr"]["ndcg_at_10"]

        pop_snips = max(per_cohort_breakdown["baseline_popularity"]["snips_score"], 1e-4)
        nexus_snips = per_cohort_breakdown["nexusrec_two_tower_mmr"]["snips_score"]
        snips_lift_multiplier = round(nexus_snips / pop_snips, 2)

        lift_vs_pop_pct = round(((nexus_ndcg - pop_ndcg) / max(pop_ndcg, 1e-6)) * 100.0, 1)
        lift_vs_cf_pct = round(((nexus_ndcg - cf_ndcg) / max(cf_ndcg, 1e-6)) * 100.0, 1)

        return {
            "evaluation_k": k,
            "benchmark_cohorts_count": len(BENCHMARK_SCENARIOS),
            "total_in_stock_catalog": total_in_stock,
            "headline_lifts": {
                "nexusrec_ndcg_at_10": nexus_ndcg,
                "ndcg_lift_vs_popularity_pct": lift_vs_pop_pct,
                "ndcg_lift_vs_pure_cf_pct": lift_vs_cf_pct,
                "nexusrec_mrr": per_cohort_breakdown["nexusrec_two_tower_mmr"]["mrr"],
                "nexusrec_ips_score": per_cohort_breakdown["nexusrec_two_tower_mmr"]["ips_score"],
                "nexusrec_snips_score": nexus_snips,
                "snips_lift_multiplier": snips_lift_multiplier,
                "nexusrec_catalog_coverage_pct": per_cohort_breakdown["nexusrec_two_tower_mmr"][
                    "catalog_coverage_pct"
                ],
                "nexusrec_intra_list_diversity": per_cohort_breakdown["nexusrec_two_tower_mmr"][
                    "intra_list_diversity"
                ],
            },
            "models": model_summaries,
        }
