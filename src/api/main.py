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
from src.core.hybrid_retriever import UniversalHybridRetriever
from src.core.nlp_assistant import ConversationalCommerceAssistant
from src.core.reranker import CrossEncoderAndMMRReranker
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


def _initialize_engine_state(app: FastAPI) -> None:
    """Populate `app.state` with singleton ML, Retrieval, Reranking, and Evaluation engines."""
    catalog_store = MultimodalCatalogStore()
    two_tower = TwoTowerRecommender(catalog_store=catalog_store, seed=42)
    retriever = UniversalHybridRetriever(catalog_store=catalog_store, two_tower=two_tower)
    reranker = CrossEncoderAndMMRReranker(items=catalog_store.items, seed=42)
    assistant = ConversationalCommerceAssistant(
        catalog_store=catalog_store,
        retriever=retriever,
        reranker=reranker,
    )
    evaluator = RecSysBenchmarkEvaluator(
        catalog_store=catalog_store,
        assistant=assistant,
    )

    app.state.catalog_store = catalog_store
    app.state.two_tower = two_tower
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


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the interactive human-centered UI/UX frontend at `/`."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
