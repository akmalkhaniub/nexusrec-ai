"""FastAPI Application & Strict Pydantic v2 API Contracts for NexusRec AI.

Follows `fullstack-fastapi-architecture` and `human-centered-ui-ux` skills:
- Uses `@asynccontextmanager async def lifespan(app: FastAPI)` to initialize shared singleton
  resources (`MultimodalCatalogStore`, `TwoTowerRecommender`, `UniversalHybridRetriever`,
  `CrossEncoderAndMMRReranker`, `ConversationalCommerceAssistant`, and `RecSysBenchmarkEvaluator`)
  in `app.state`.
- Enforces `model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)` on all request
  models.
- Mounts the interactive zero-build operator console (`src/static/index.html`) at `/` via
  `StaticFiles` so the full stack launches cleanly with `uvicorn src.api.main:app --reload`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.core.catalog import MultimodalCatalogStore, extract_query_visual_signature
from src.core.evaluator import RecSysBenchmarkEvaluator
from src.core.feature_store_and_export import (
    CrossFrameworkModelExporter,
    NoSQLOnlineSessionStore,
    SQLOfflineFeatureStore,
)
from src.core.hybrid_retriever import UniversalHybridRetriever
from src.core.nlp_assistant import ConversationalCommerceAssistant
from src.core.reranker import CrossEncoderAndMMRReranker
from src.core.sequential_sasrec import SASRecSequentialPredictor
from src.core.two_tower import TwoTowerRecommender

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class DiscoverySearchRequest(BaseModel):
    """Strict Pydantic v2 request schema for the 4-stage multimodal discovery funnel."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        default="waterproof alpine shell under $220 that pairs with crampon boots",
        min_length=1,
        max_length=500,
        description="Natural language conversational search or MRO part specification.",
    )
    persona_id: str = Field(
        default="alpine_mountaineer",
        description="Buyer persona ID encoding session history and commercial sensitivity.",
    )
    visual_preset: str | None = Field(
        default=None,
        description="Optional camera/visual reference upload preset ID.",
    )
    domain: Literal["retail", "industrial"] | None = Field(
        default=None,
        description="Optional explicit domain filter ('retail' or 'industrial').",
    )
    category: str | None = Field(
        default=None,
        description="Optional explicit category filter.",
    )
    max_price: float | None = Field(
        default=None,
        gt=0.0,
        le=10000.0,
        description="Optional explicit price ceiling in USD.",
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of final diversified recommendations to return.",
    )
    lambda_diversity: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Maximal Marginal Relevance (MMR) trade-off (1.0 = pure relevance, 0.0 = max diversity).",
    )
    bandit_exploration_weight: float = Field(
        default=0.12,
        ge=0.0,
        le=0.5,
        description="Weight assigned to Thompson Sampling cold-start exploration.",
    )
    in_stock_only: bool = Field(
        default=True,
        description="Whether to enforce Qdrant hard payload filter `in_stock == True`.",
    )


class BanditFeedbackRequest(BaseModel):
    """Strict Pydantic v2 request schema for recording online conversion/click bandit feedback."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    item_id: str = Field(..., min_length=3, max_length=32)
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Observed conversion/click reward in [0.0, 1.0] to update Beta(alpha, beta).",
    )
    event_type: Literal["click", "add_to_cart", "purchase_conversion", "dismiss"] = Field(
        default="purchase_conversion"
    )


class TwoTowerVerifyRequest(BaseModel):
    """Strict Pydantic v2 request schema for on-demand Two-Tower InfoNCE contrastive verification."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    epochs: int = Field(default=10, ge=1, le=50)
    learning_rate: float = Field(default=0.06, gt=0.001, le=0.5)


class SessionEventRequest(BaseModel):
    """Strict Pydantic v2 request schema for live clickstream trajectory simulation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    persona_id: str = Field(
        default="alpine_mountaineer",
        min_length=2,
        max_length=64,
        description="Active buyer persona ID.",
    )
    item_id: str = Field(
        ...,
        min_length=3,
        max_length=32,
        description="Clicked, added-to-cart, or skipped catalog item ID.",
    )
    event_type: Literal["click", "add_to_cart", "purchase_conversion", "skip"] = Field(
        default="click",
        description="User interaction type updating NoSQL session, SQL log, and SASRec sequence.",
    )
    rank_position: int = Field(
        default=1,
        ge=1,
        le=50,
        description="1-indexed rank position where the user interacted with the SKU.",
    )
    query: str = Field(
        default="waterproof alpine shell under $220 that pairs with crampon boots",
        min_length=1,
        max_length=500,
    )
    visual_preset: str | None = Field(default=None)
    top_k: int = Field(default=6, ge=1, le=20)
    lambda_diversity: float = Field(default=0.72, ge=0.0, le=1.0)


def _initialize_engine_state(app: FastAPI) -> None:
    """Populate `app.state` with singleton ML, Retrieval, Reranking, Feature Store & Evaluation engines."""
    catalog_store = MultimodalCatalogStore()
    two_tower = TwoTowerRecommender(catalog_store=catalog_store, seed=42)
    sasrec = SASRecSequentialPredictor(catalog_store=catalog_store, seed=42)
    sql_offline_store = SQLOfflineFeatureStore(catalog_store=catalog_store)
    nosql_online_store = NoSQLOnlineSessionStore(catalog_store=catalog_store)
    model_exporter = CrossFrameworkModelExporter(two_tower=two_tower, sasrec=sasrec)

    retriever = UniversalHybridRetriever(catalog_store=catalog_store, two_tower=two_tower)
    reranker = CrossEncoderAndMMRReranker(items=catalog_store.items, seed=42)
    assistant = ConversationalCommerceAssistant(
        catalog_store=catalog_store,
        retriever=retriever,
        reranker=reranker,
        sasrec=sasrec,
        online_store=nosql_online_store,
    )
    evaluator = RecSysBenchmarkEvaluator(
        catalog_store=catalog_store,
        assistant=assistant,
    )

    app.state.catalog_store = catalog_store
    app.state.two_tower = two_tower
    app.state.sasrec = sasrec
    app.state.sql_offline_store = sql_offline_store
    app.state.nosql_online_store = nosql_online_store
    app.state.model_exporter = model_exporter
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.assistant = assistant
    app.state.evaluator = evaluator
    app.state.cached_evaluation = evaluator.evaluate_all_models(k=10)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Async lifespan context manager initializing shared singleton ML & RecSys stores."""
    _initialize_engine_state(app)
    yield


app = FastAPI(
    title="NexusRec AI — Multimodal Visual-Semantic Retail & Industrial Discovery Engine",
    description=(
        "Production Two-Tower Neural Recommender, Qdrant-Style Hybrid Dense+BM25 RRF Retriever, "
        "Cross-Encoder + Thompson Sampling Bandit Reranker, and MMR Diversity Engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_state(app_instance: FastAPI) -> None:
    if not hasattr(app_instance.state, "catalog_store"):
        _initialize_engine_state(app_instance)


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    """Liveness and readiness probe with active Two-Tower & catalog telemetry."""
    _ensure_state(app)
    store: MultimodalCatalogStore = app.state.catalog_store
    tt: TwoTowerRecommender = app.state.two_tower
    return {
        "status": "healthy",
        "service": "nexusrec-ai",
        "version": "1.0.0",
        "two_tower_backend": tt.backend,
        "embedding_dim": tt.embedding_dim,
        "infonce_loss": tt.training_metrics.final_infonce_loss,
        "catalog_summary": store.summary_stats(),
    }


@app.get("/api/v1/catalog")
async def list_catalog(
    domain: Literal["retail", "industrial"] | None = Query(default=None),
    category: str | None = Query(default=None),
    in_stock_only: bool = Query(default=False),
) -> dict[str, Any]:
    """Return the multimodal catalog items, buyer personas, and visual query presets."""
    _ensure_state(app)
    store: MultimodalCatalogStore = app.state.catalog_store
    items = store.items
    if domain:
        items = [i for i in items if i.domain == domain]
    if category:
        items = [i for i in items if i.category.lower() == category.lower()]
    if in_stock_only:
        items = [i for i in items if i.in_stock]

    visual_presets = [
        extract_query_visual_signature("alpine_hardshell_jacket"),
        extract_query_visual_signature("crampon_mountaineering_boot"),
        extract_query_visual_signature("hydraulic_viton_seal_kit"),
        extract_query_visual_signature("cnc_stainless_manifold_valve"),
        extract_query_visual_signature("ceramic_spindle_bearing"),
    ]

    return {
        "summary": store.summary_stats(),
        "personas": [
            {
                "user_id": p.user_id,
                "name": p.name,
                "role": p.role,
                "preferred_domain": p.preferred_domain,
                "budget_ceiling": p.budget_ceiling,
                "price_sensitivity": p.price_sensitivity,
                "margin_affinity": p.margin_affinity,
                "recent_item_ids": p.recent_item_ids,
                "intent_keywords": p.intent_keywords,
                "description": p.description,
            }
            for p in store.personas.values()
        ],
        "visual_presets": [
            {k: v for k, v in vp.items() if k != "visual_embedding"} for vp in visual_presets
        ],
        "items": [i.to_dict() for i in items],
    }


@app.post("/api/v1/discover")
async def run_discovery_funnel(payload: DiscoverySearchRequest) -> dict[str, Any]:
    """Execute the 4-stage Conversational & Visual Discovery Recommendation Funnel."""
    _ensure_state(app)
    assistant: ConversationalCommerceAssistant = app.state.assistant
    result = assistant.execute_funnel(
        query=payload.query,
        persona_id=payload.persona_id,
        visual_preset=payload.visual_preset,
        explicit_domain=payload.domain,
        explicit_category=payload.category,
        explicit_max_price=payload.max_price,
        top_k=payload.top_k,
        lambda_diversity=payload.lambda_diversity,
        bandit_exploration_weight=payload.bandit_exploration_weight,
        in_stock_only=payload.in_stock_only,
    )
    return result


@app.get("/api/v1/evaluate")
async def get_benchmark_evaluation() -> dict[str, Any]:
    """Return offline RecSys benchmark metrics (NDCG@10, Recall@10, MRR, Coverage, ILD)."""
    _ensure_state(app)
    tt: TwoTowerRecommender = app.state.two_tower
    eval_report: dict[str, Any] = app.state.cached_evaluation
    return {
        **eval_report,
        "two_tower_training_verification": {
            "backend": tt.training_metrics.backend,
            "embedding_dim": tt.training_metrics.embedding_dim,
            "temperature": tt.training_metrics.temperature,
            "epochs": tt.training_metrics.epochs,
            "initial_infonce_loss": tt.training_metrics.initial_infonce_loss,
            "final_infonce_loss": tt.training_metrics.final_infonce_loss,
            "loss_reduction_pct": tt.training_metrics.loss_reduction_pct,
            "positive_pair_mean_cosine": tt.training_metrics.positive_pair_mean_cosine,
            "negative_pair_mean_cosine": tt.training_metrics.negative_pair_mean_cosine,
            "separation_margin": tt.training_metrics.separation_margin,
        },
    }


@app.post("/api/v1/bandit/feedback")
async def record_bandit_feedback(payload: BanditFeedbackRequest) -> dict[str, Any]:
    """Update the Thompson Sampling Contextual Bandit Beta(alpha, beta) posterior for an item."""
    _ensure_state(app)
    reranker: CrossEncoderAndMMRReranker = app.state.reranker
    try:
        updated_arm = reranker.bandit.update_arm(
            item_id=payload.item_id,
            reward=payload.reward,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Catalog item '{payload.item_id}' not found in Thompson Sampling bandit arms.",
        ) from exc

    return {
        "status": "updated",
        "event_type": payload.event_type,
        "item_id": updated_arm.item_id,
        "alpha": updated_arm.alpha,
        "beta": updated_arm.beta,
        "impressions": updated_arm.impressions,
        "posterior_mean": updated_arm.posterior_mean,
        "uncertainty_std": updated_arm.uncertainty_std,
        "cold_start": updated_arm.cold_start,
    }


@app.post("/api/v1/two-tower/verify")
async def verify_two_tower_training(payload: TwoTowerVerifyRequest) -> dict[str, Any]:
    """Run an on-demand contrastive InfoNCE training verification step on the Two-Tower network."""
    _ensure_state(app)
    tt: TwoTowerRecommender = app.state.two_tower
    metrics = tt.run_contrastive_training_step(
        epochs=payload.epochs,
        lr=payload.learning_rate,
    )
    return {
        "status": "verified",
        "backend": metrics.backend,
        "embedding_dim": metrics.embedding_dim,
        "epochs": metrics.epochs,
        "initial_infonce_loss": metrics.initial_infonce_loss,
        "final_infonce_loss": metrics.final_infonce_loss,
        "loss_reduction_pct": metrics.loss_reduction_pct,
        "positive_pair_mean_cosine": metrics.positive_pair_mean_cosine,
        "negative_pair_mean_cosine": metrics.negative_pair_mean_cosine,
        "separation_margin": metrics.separation_margin,
    }


@app.post("/api/v1/session/event")
async def record_live_session_event(payload: SessionEventRequest) -> dict[str, Any]:
    """Simulate a live clickstream event, update SQL/NoSQL stores & SASRec, and re-rank shelf."""
    _ensure_state(app)
    store: MultimodalCatalogStore = app.state.catalog_store
    item = store.get_item(payload.item_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Catalog item '{payload.item_id}' not found.",
        )

    nosql_store: NoSQLOnlineSessionStore = app.state.nosql_online_store
    sql_store: SQLOfflineFeatureStore = app.state.sql_offline_store
    reranker: CrossEncoderAndMMRReranker = app.state.reranker
    sasrec: SASRecSequentialPredictor = app.state.sasrec
    assistant: ConversationalCommerceAssistant = app.state.assistant

    # Capture pre-event SASRec next-item embedding to quantify live user embedding drift
    prev_session = nosql_store.get_session(payload.persona_id)
    prev_sasrec = sasrec.predict_next_items(list(prev_session.last_5_clicks))

    reward_map = {
        "purchase_conversion": 1.0,
        "add_to_cart": 0.85,
        "click": 0.65,
        "skip": 0.0,
    }
    reward_val = reward_map.get(payload.event_type, 0.5)

    # 1. Update NoSQL Online Session Store (last_5_clicks, session_price_anchor, real_time_ctr)
    updated_doc = nosql_store.record_realtime_event(
        user_id=payload.persona_id,
        item_id=payload.item_id,
        event_type=payload.event_type,
    )

    # 2. Log interaction with position propensity to SQL Offline Store
    sql_log = sql_store.log_interaction_event(
        user_id=payload.persona_id,
        item_id=payload.item_id,
        event_type=payload.event_type,
        rank_position=payload.rank_position,
        reward=reward_val,
    )

    # 3. Update Thompson Sampling Bandit posterior
    updated_arm = reranker.bandit.update_arm(item_id=payload.item_id, reward=reward_val)

    # 4. Recompute SASRec sequential attention and measure User Embedding Cosine Drift
    new_sasrec = sasrec.predict_next_items(list(updated_doc.last_5_clicks))
    import numpy as np

    v_prev = np.asarray(prev_sasrec.next_item_embedding_64d, dtype=np.float32)
    v_new = np.asarray(new_sasrec.next_item_embedding_64d, dtype=np.float32)
    cos_alignment = float(np.dot(v_prev, v_new))
    embedding_drift = round(max(0.0, 1.0 - cos_alignment), 4)

    # 5. Re-execute the recommendation funnel with the updated live session state
    updated_funnel = assistant.execute_funnel(
        query=payload.query,
        persona_id=payload.persona_id,
        visual_preset=payload.visual_preset,
        top_k=payload.top_k,
        lambda_diversity=payload.lambda_diversity,
    )

    return {
        "status": "session_updated",
        "interacted_item": {
            "item_id": item.item_id,
            "sku": item.sku,
            "title": item.title,
            "category": item.category,
            "event_type": payload.event_type,
            "reward": reward_val,
        },
        "embedding_drift_l2_cosine": embedding_drift,
        "online_session_document": updated_doc.to_dict(),
        "sql_offline_event_logged": sql_log,
        "bandit_arm_posterior": {
            "item_id": updated_arm.item_id,
            "alpha": updated_arm.alpha,
            "beta": updated_arm.beta,
            "posterior_mean": updated_arm.posterior_mean,
        },
        "sasrec_sequential_state": new_sasrec.to_dict(),
        "updated_funnel": updated_funnel,
    }


@app.get("/api/v1/feature-store/status")
async def get_feature_store_and_export_status() -> dict[str, Any]:
    """Return SQL point-in-time audit, NoSQL online cache telemetry, and PyTorch/TF/ONNX parity."""
    _ensure_state(app)
    sql_store: SQLOfflineFeatureStore = app.state.sql_offline_store
    nosql_store: NoSQLOnlineSessionStore = app.state.nosql_online_store
    exporter: CrossFrameworkModelExporter = app.state.model_exporter
    eval_report: dict[str, Any] = app.state.cached_evaluation

    # Run a live point-in-time join audit at historical cutoff 1700000250.0
    pit_audit = sql_store.point_in_time_join(
        user_id="alpine_mountaineer",
        as_of_timestamp=1700000250.0,
    )
    parity_report = exporter.verify_cross_framework_parity()

    return {
        "sql_offline_store": {
            **sql_store.get_offline_summary(),
            "point_in_time_join_audit": pit_audit,
        },
        "nosql_online_store": nosql_store.get_online_summary(),
        "cross_framework_parity": parity_report,
        "counterfactual_off_policy_summary": {
            "nexusrec_ips_score": eval_report["headline_lifts"]["nexusrec_ips_score"],
            "nexusrec_snips_score": eval_report["headline_lifts"]["nexusrec_snips_score"],
            "snips_lift_multiplier": eval_report["headline_lifts"]["snips_lift_multiplier"],
        },
    }


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the interactive human-centered UI/UX frontend at `/`."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
