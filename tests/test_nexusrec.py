"""Comprehensive Unit & Integration Test Suite for NexusRec AI.

Covers:
1. Multimodal Catalog (`src/core/catalog.py`): 44 items, Retail & Industrial domains,
   32-D deterministic OpenCV/NumPy visual descriptors, 32-D semantic descriptors,
   and 64-D L2-normalized multimodal vectors.
2. Two-Tower Neural Network (`src/core/two_tower.py`): UserTower + ItemTower 64-D L2
   normalization, in-batch negative InfoNCE contrastive loss calculation, and training
   step verification.
3. Universal Hybrid Retriever (`src/core/hybrid_retriever.py`): Qdrant-style hard metadata
   filtering (`in_stock`, `max_price`, `domain`), BM25 lexical scoring, and Reciprocal
   Rank Fusion (RRF).
4. Cross-Encoder, Thompson Sampling Bandit & MMR Reranker (`src/core/reranker.py`):
   Pairwise feature crosses, Beta(alpha, beta) posterior updates, and Maximal Marginal
   Relevance diversity filtering.
5. Conversational Intent Parser (`src/core/nlp_assistant.py`): Natural language price,
   domain, category, and compatibility extraction across Retail & Industrial queries.
6. RecSys Benchmark Evaluator (`src/core/evaluator.py`): NDCG@10, Recall@10, Precision@10,
   MRR, Catalog Coverage, and Intra-List Diversity (ILD) verification.
7. FastAPI Application & Strict Pydantic v2 Contracts (`src/api/main.py`): All REST routes
   and `extra="forbid"` validation via `TestClient`.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.catalog import (
    MultimodalCatalogStore,
    extract_query_visual_signature,
    extract_visual_descriptor,
)
from src.core.evaluator import (
    RecSysBenchmarkEvaluator,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from src.core.hybrid_retriever import PayloadFilter, UniversalHybridRetriever
from src.core.nlp_assistant import ConversationalCommerceAssistant
from src.core.reranker import CrossEncoderAndMMRReranker
from src.core.two_tower import EMBEDDING_DIM, TwoTowerRecommender


@pytest.fixture(scope="module")
def catalog_store() -> MultimodalCatalogStore:
    return MultimodalCatalogStore()


@pytest.fixture(scope="module")
def two_tower(catalog_store: MultimodalCatalogStore) -> TwoTowerRecommender:
    return TwoTowerRecommender(catalog_store=catalog_store, seed=42)


@pytest.fixture(scope="module")
def retriever(
    catalog_store: MultimodalCatalogStore,
    two_tower: TwoTowerRecommender,
) -> UniversalHybridRetriever:
    return UniversalHybridRetriever(catalog_store=catalog_store, two_tower=two_tower)


@pytest.fixture(scope="module")
def reranker(catalog_store: MultimodalCatalogStore) -> CrossEncoderAndMMRReranker:
    return CrossEncoderAndMMRReranker(items=catalog_store.items, seed=42)


@pytest.fixture(scope="module")
def assistant(
    catalog_store: MultimodalCatalogStore,
    retriever: UniversalHybridRetriever,
    reranker: CrossEncoderAndMMRReranker,
) -> ConversationalCommerceAssistant:
    return ConversationalCommerceAssistant(
        catalog_store=catalog_store,
        retriever=retriever,
        reranker=reranker,
    )


def test_multimodal_catalog_completeness_and_descriptors(
    catalog_store: MultimodalCatalogStore,
) -> None:
    """Verify catalog has >= 40 items across Retail & Industrial with normalized 32D/64D vectors."""
    stats = catalog_store.summary_stats()
    assert stats["total_items"] >= 40
    assert stats["retail_items"] >= 20
    assert stats["industrial_items"] >= 20
    assert stats["visual_dim"] == 32
    assert stats["semantic_dim"] == 32
    assert stats["multimodal_dim"] == 64

    sample = catalog_store.get_item("RET-001")
    assert sample is not None
    assert len(sample.visual_embedding) == 32
    assert len(sample.semantic_embedding) == 32
    assert len(sample.multimodal_features) == 64
    assert np.isclose(np.linalg.norm(sample.visual_embedding), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(sample.multimodal_features), 1.0, atol=1e-4)

    # Deterministic visual extraction check
    v1 = extract_visual_descriptor("SKU-1", "retail", "Alpine Shells", "#1E6B9B", "gore-tex-3l-ripstop")
    v2 = extract_visual_descriptor("SKU-1", "retail", "Alpine Shells", "#1E6B9B", "gore-tex-3l-ripstop")
    assert np.allclose(v1, v2)

    vis_preset = extract_query_visual_signature("hydraulic_viton_seal_kit")
    assert vis_preset["detected_domain"] == "industrial"
    assert len(vis_preset["visual_embedding"]) == 32


def test_two_tower_embeddings_and_infonce_loss(
    catalog_store: MultimodalCatalogStore,
    two_tower: TwoTowerRecommender,
) -> None:
    """Verify UserTower + ItemTower produce L2-normalized 64-D vectors and InfoNCE loss converges."""
    metrics = two_tower.training_metrics
    assert metrics.embedding_dim == EMBEDDING_DIM
    assert metrics.final_infonce_loss < metrics.initial_infonce_loss
    assert metrics.loss_reduction_pct > 25.0
    assert metrics.positive_pair_mean_cosine > metrics.negative_pair_mean_cosine
    assert metrics.separation_margin > 0.35

    persona = catalog_store.get_persona("alpine_mountaineer")
    u_emb, scores = two_tower.score_candidates(persona=persona)
    assert u_emb.shape == (EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(u_emb), 1.0, atol=1e-4)
    assert len(scores) == len(catalog_store.items)
    # Alpine mountaineer should score Retail Alpine Shell RET-001 higher than Industrial Seal IND-004
    assert scores["RET-001"] > scores["IND-004"]


def test_hybrid_retriever_payload_filtering_and_rrf(
    catalog_store: MultimodalCatalogStore,
    retriever: UniversalHybridRetriever,
) -> None:
    """Verify Qdrant hard metadata filtering and Dense + BM25 Reciprocal Rank Fusion."""
    persona = catalog_store.get_persona("plant_reliability_lead")
    p_filter = PayloadFilter(
        domain="industrial",
        max_price=150.0,
        in_stock_only=True,
    )
    out = retriever.retrieve(
        persona=persona,
        query_text="high-pressure hydraulic seal replacement under $150",
        payload_filter=p_filter,
        top_k=10,
    )

    assert out["filtered_candidate_count"] > 0
    for cand in out["candidates"]:
        assert cand.item.domain == "industrial"
        assert cand.item.unit_price <= 150.0
        assert cand.item.in_stock is True
        assert 0.0 < cand.rrf_fusion_score <= 1.0

    # Top candidate for hydraulic seal query should be a Hydraulic Seal (IND-001 or IND-002)
    top_item = out["candidates"][0].item
    assert top_item.item_id in {"IND-001", "IND-002", "IND-006"}


def test_cross_encoder_bandit_and_mmr_diversification(
    catalog_store: MultimodalCatalogStore,
    retriever: UniversalHybridRetriever,
    reranker: CrossEncoderAndMMRReranker,
) -> None:
    """Verify MMR diversity increases category spread when lambda_diversity is lowered."""
    persona = catalog_store.get_persona("alpine_mountaineer")
    p_filter = PayloadFilter(domain="retail", max_price=220.0, in_stock_only=True)
    ret_out = retriever.retrieve(
        persona=persona,
        query_text="waterproof alpine shell under $220 that pairs with crampon boots",
        payload_filter=p_filter,
        top_k=16,
    )

    # Pure relevance (lambda = 1.0) vs High diversity (lambda = 0.45)
    pure_rel = reranker.rerank_with_mmr(
        candidates=ret_out["candidates"],
        persona=persona,
        top_k=6,
        lambda_diversity=1.0,
    )
    high_div = reranker.rerank_with_mmr(
        candidates=ret_out["candidates"],
        persona=persona,
        top_k=6,
        lambda_diversity=0.45,
    )

    assert len(high_div["recommendations"]) == 6
    assert high_div["intra_list_diversity"] >= pure_rel["intra_list_diversity"]

    # Verify Thompson Sampling bandit posterior update
    arm_before = reranker.bandit.arms["RET-004"]
    alpha_prev = arm_before.alpha
    updated = reranker.bandit.update_arm("RET-004", reward=1.0)
    assert np.isclose(updated.alpha, alpha_prev + 1.0)


def test_conversational_nlp_intent_parser_and_funnel(
    assistant: ConversationalCommerceAssistant,
) -> None:
    """Verify natural language extraction of price ceilings, domains, and compatibility."""
    q1 = "waterproof alpine shell under $220 that pairs with crampon boots"
    parsed1 = assistant.parse_intent(q1)
    assert parsed1.detected_domain == "retail"
    assert parsed1.detected_category == "Alpine Shells"
    assert parsed1.max_price == 220.0
    assert "RET-005" in parsed1.compatibility_anchor_ids
    assert parsed1.intent_mode == "compatibility_cross_sell"

    q2 = "high-pressure hydraulic seal replacement under $150"
    parsed2 = assistant.parse_intent(q2)
    assert parsed2.detected_domain == "industrial"
    assert parsed2.detected_category == "Hydraulic Seals"
    assert parsed2.max_price == 150.0
    assert parsed2.intent_mode == "mro_replacement"

    funnel_res = assistant.execute_funnel(
        query=q1,
        persona_id="alpine_mountaineer",
        visual_preset="alpine_hardshell_jacket",
        top_k=6,
    )
    assert len(funnel_res["recommendations"]) == 6
    for rec in funnel_res["recommendations"]:
        assert rec["item"]["unit_price"] <= 220.0
        assert rec["item"]["in_stock"] is True


def test_recsys_evaluator_metrics_and_nexusrec_lift(
    catalog_store: MultimodalCatalogStore,
    assistant: ConversationalCommerceAssistant,
) -> None:
    """Verify deterministic NDCG@10, Recall@10, MRR, and NexusRec lift over baselines."""
    grades = {"A": 3, "B": 2, "C": 1}
    assert compute_ndcg_at_k(["A", "B", "C"], grades, k=3) == 1.0
    assert compute_recall_at_k(["A", "X", "B"], grades, k=3) == round(2 / 3, 4)
    assert compute_precision_at_k(["A", "X"], grades, k=2) == 0.5
    assert compute_mrr(["X", "A", "B"], grades) == 0.5

    evaluator = RecSysBenchmarkEvaluator(catalog_store=catalog_store, assistant=assistant)
    report = evaluator.evaluate_all_models(k=10)

    lifts = report["headline_lifts"]
    assert lifts["nexusrec_ndcg_at_10"] > 0.80
    assert lifts["ndcg_lift_vs_popularity_pct"] > 50.0
    assert lifts["ndcg_lift_vs_pure_cf_pct"] > 10.0
    assert lifts["nexusrec_mrr"] >= 0.90


def test_fastapi_endpoints_and_strict_pydantic_validation() -> None:
    """Verify all FastAPI endpoints and strict Pydantic v2 ConfigDict(extra='forbid')."""
    with TestClient(app) as client:
        # 1. Root UI HTML
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "NexusRec AI" in res_root.text
        assert "<details" in res_root.text

        # 2. Health check
        res_health = client.get("/api/v1/health")
        assert res_health.status_code == 200
        h_data = res_health.json()
        assert h_data["status"] == "healthy"
        assert h_data["embedding_dim"] == 64

        # 3. Catalog listing
        res_cat = client.get("/api/v1/catalog?domain=industrial&in_stock_only=true")
        assert res_cat.status_code == 200
        c_data = res_cat.json()
        assert len(c_data["items"]) >= 20
        assert all(i["domain"] == "industrial" and i["in_stock"] for i in c_data["items"])

        # 4. Discovery Funnel POST
        res_disc = client.post(
            "/api/v1/discover",
            json={
                "query": "high-pressure hydraulic seal replacement under $150",
                "persona_id": "plant_reliability_lead",
                "visual_preset": "hydraulic_viton_seal_kit",
                "top_k": 5,
                "lambda_diversity": 0.72,
            },
        )
        assert res_disc.status_code == 200
        d_data = res_disc.json()
        assert len(d_data["recommendations"]) == 5
        assert d_data["recommendations"][0]["item"]["unit_price"] <= 150.0

        # 5. Strict Pydantic v2 extra="forbid" must reject unknown fields with 422
        res_bad = client.post(
            "/api/v1/discover",
            json={
                "query": "alpine shell",
                "unexpected_hack_field": "should_fail",
            },
        )
        assert res_bad.status_code == 422

        # 6. Evaluation endpoint
        res_eval = client.get("/api/v1/evaluate")
        assert res_eval.status_code == 200
        e_data = res_eval.json()
        assert len(e_data["models"]) == 3
        assert "two_tower_training_verification" in e_data

        # 7. Bandit feedback endpoint
        res_bandit = client.post(
            "/api/v1/bandit/feedback",
            json={
                "item_id": "IND-001",
                "reward": 1.0,
                "event_type": "purchase_conversion",
            },
        )
        assert res_bandit.status_code == 200
        assert res_bandit.json()["status"] == "updated"

        # 8. Two-Tower contrastive verification endpoint
        res_tt = client.post(
            "/api/v1/two-tower/verify",
            json={"epochs": 8, "learning_rate": 0.05},
        )
        assert res_tt.status_code == 200
        assert res_tt.json()["status"] == "verified"
