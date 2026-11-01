"""LangChain / LangGraph-Pattern Conversational Intent Parser & Visual Search Router.

Executes a deterministic 4-Node Stateful Agent Graph:
- Node 1 (`parse_conversational_intent`): Extracts structured constraints (`max_price`,
  `min_price`, `domain`, `category`, `technical_attributes`, `compatibility_targets`,
  and `intent_mode`) from natural language queries such as:
  - "waterproof alpine shell under $220 that pairs with crampon boots"
  - "high-pressure hydraulic seal replacement under $150"
  - "ABEC-9 ceramic CNC spindle bearing and vibration telemetry under $250"
- Node 2 (`extract_multimodal_query_vectors`): Computes the 32-D OpenCV visual query signature
  (if a visual preset or color/texture cue is present) and 32-D semantic query vector.
- Node 3 (`execute_hybrid_retrieval`): Runs Qdrant-style pre-filtering + Two-Tower Dense
  Cosine + Sparse BM25/SPLADE + Reciprocal Rank Fusion (RRF).
- Node 4 (`rerank_and_synthesize_response`): Runs Cross-Encoder + Thompson Sampling Bandit
  + Maximal Marginal Relevance (MMR) diversification and synthesizes an actionable,
  human-centered procurement / merchandising brief.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.catalog import (
    MultimodalCatalogStore,
    extract_query_visual_signature,
    extract_semantic_descriptor,
)
from src.core.feature_store_and_export import NoSQLOnlineSessionStore
from src.core.hybrid_retriever import PayloadFilter, UniversalHybridRetriever
from src.core.reranker import CrossEncoderAndMMRReranker, RerankedRecommendation
from src.core.sequential_sasrec import SASRecSequentialPredictor


@dataclass(frozen=True)
class ParsedConversationalIntent:
    """Structured constraint envelope extracted from natural language + visual input."""

    raw_query: str
    detected_domain: str | None
    detected_category: str | None
    max_price: float | None
    min_price: float | None
    in_stock_only: bool
    extracted_tags: list[str]
    compatibility_anchor_ids: list[str]
    intent_mode: str  # "compatibility_cross_sell" | "mro_replacement" | "visual_similarity" | "discovery"
    visual_preset_used: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "detected_domain": self.detected_domain,
            "detected_category": self.detected_category,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "in_stock_only": self.in_stock_only,
            "extracted_tags": self.extracted_tags,
            "compatibility_anchor_ids": self.compatibility_anchor_ids,
            "intent_mode": self.intent_mode,
            "visual_preset_used": self.visual_preset_used,
        }


_CATEGORY_KEYWORD_RULES: list[tuple[list[str], str, str]] = [
    (["alpine shell", "hardshell", "storm shell", "anorak", "parka", "shell"], "retail", "Alpine Shells"),
    (["crampon boot", "mountaineering boot", "approach shoe", "footwear"], "retail", "Mountaineering Footwear"),
    (["midlayer", "down hoody", "fleece", "belay jacket", "insulated vest"], "retail", "Insulated Midlayers"),
    (["ice axe", "carabiner", "climbing harness", "12-point crampon"], "retail", "Climbing Hardware"),
    (["expedition pack", "summit pack", "alpine pack", "backpack"], "retail", "Expedition Packs"),
    (["baselayer", "merino", "gaiters", "climbing gloves"], "retail", "Technical Baselayers"),
    (["hydraulic seal", "piston seal", "rod seal", "wiper", "o-ring", "seal replacement"], "industrial", "Hydraulic Seals"),
    (["proportional valve", "relief valve", "cartridge valve", "solenoid valve", "counterbalance valve", "directional valve"], "industrial", "High-Pressure Valves"),
    (["ceramic bearing", "spindle bearing", "roller bearing", "linear guide", "rail carriage", "bearing"], "industrial", "Precision Bearings"),
    (["hydraulic cylinder", "orbital motor", "pneumatic cylinder", "ball screw", "actuator"], "industrial", "Industrial Actuators"),
    (["pressure transducer", "vibration", "accelerometer", "flow meter", "position sensor", "telemetry"], "industrial", "Sensor Telemetry"),
    (["servo coupling", "shaft coupling", "spin-on filter", "filter element"], "industrial", "Power Transmission"),
]

_RETAIL_SIGNALS = {
    "alpine", "shell", "gore-tex", "waterproof", "crampon", "boots", "mountaineering",
    "climbing", "fleece", "merino", "down", "hoody", "pack", "vibram", "gaiters", "ultralight",
}

_INDUSTRIAL_SIGNALS = {
    "hydraulic", "seal", "high-pressure", "viton", "ptfe", "valve", "manifold",
    "bearing", "ceramic", "spindle", "cnc", "actuator", "cylinder", "transducer",
    "sensor", "piston", "solenoid", "replacement", "bar", "psi",
}

_KNOWN_TAGS = sorted(_RETAIL_SIGNALS | _INDUSTRIAL_SIGNALS)


class ConversationalCommerceAssistant:
    """LangGraph-style 4-node conversational + visual search orchestrator."""

    def __init__(
        self,
        catalog_store: MultimodalCatalogStore,
        retriever: UniversalHybridRetriever,
        reranker: CrossEncoderAndMMRReranker,
        sasrec: SASRecSequentialPredictor | None = None,
        online_store: NoSQLOnlineSessionStore | None = None,
    ) -> None:
        self.catalog_store = catalog_store
        self.retriever = retriever
        self.reranker = reranker
        self.sasrec = sasrec or SASRecSequentialPredictor(catalog_store=catalog_store, seed=42)
        self.online_store = online_store or NoSQLOnlineSessionStore(catalog_store=catalog_store)

    def parse_intent(
        self,
        query: str,
        explicit_domain: str | None = None,
        explicit_category: str | None = None,
        explicit_max_price: float | None = None,
        visual_preset: str | None = None,
        in_stock_only: bool = True,
    ) -> ParsedConversationalIntent:
        """Node 1: Deterministic natural language constraint & compatibility parser."""
        q_lower = query.lower().strip()

        # Extract price ceiling patterns: "under $220", "below 150", "< $200", "max $180"
        max_price = explicit_max_price
        if max_price is None:
            price_match = re.search(
                r"(?:under|below|less\s+than|max|up\s+to|<=?)\s*\$?\s*(\d+(?:\.\d{1,2})?)",
                q_lower,
            )
            if price_match:
                max_price = float(price_match.group(1))

        min_price: float | None = None
        min_match = re.search(r"(?:over|above|min|at\s+least|>=?)\s*\$?\s*(\d+(?:\.\d{1,2})?)", q_lower)
        if min_match:
            min_price = float(min_match.group(1))

        # Detect compatibility phrasing (e.g., "that pairs with crampon boots")
        compat_anchors: list[str] = []
        primary_clause = q_lower
        for sep in ["that pairs with", "compatible with", "to pair with", "that fits", "for use with"]:
            if sep in q_lower:
                parts = q_lower.split(sep, 1)
                primary_clause = parts[0]
                compat_clause = parts[1]
                if "crampon" in compat_clause or "boot" in compat_clause:
                    compat_anchors.extend(["RET-005", "RET-006", "RET-013"])
                if "cylinder" in compat_clause or "actuator" in compat_clause:
                    compat_anchors.extend(["IND-013", "IND-005"])
                if "spindle" in compat_clause or "cnc" in compat_clause:
                    compat_anchors.extend(["IND-009", "IND-011"])
                break

        # Detect primary category and domain from the main subject clause
        detected_domain = explicit_domain
        detected_category = explicit_category

        if detected_category is None:
            for phrases, dom, cat in _CATEGORY_KEYWORD_RULES:
                if any(p in primary_clause for p in phrases):
                    detected_category = cat
                    if detected_domain is None:
                        detected_domain = dom
                    break

        # Infer domain from keyword polarity if still unspecified
        if detected_domain is None:
            ret_hits = sum(1 for tok in _RETAIL_SIGNALS if tok in q_lower)
            ind_hits = sum(1 for tok in _INDUSTRIAL_SIGNALS if tok in q_lower)
            if ret_hits > ind_hits:
                detected_domain = "retail"
            elif ind_hits > ret_hits:
                detected_domain = "industrial"

        # If a visual preset was uploaded, merge its visual taxonomy cues
        if visual_preset:
            vis_sig = extract_query_visual_signature(visual_preset)
            if detected_domain is None:
                detected_domain = vis_sig["detected_domain"]
            if detected_category is None and not q_lower:
                detected_category = vis_sig["detected_category"]

        # Extract canonical technical/merchandising tags
        extracted_tags = [tag for tag in _KNOWN_TAGS if tag in q_lower]
        if visual_preset:
            vis_sig = extract_query_visual_signature(visual_preset)
            for vtag in vis_sig["visual_tags"]:
                if vtag not in extracted_tags:
                    extracted_tags.append(vtag)

        # Classify high-level conversational intent mode
        if compat_anchors:
            intent_mode = "compatibility_cross_sell"
        elif "replacement" in q_lower or "kit" in q_lower or "repair" in q_lower:
            intent_mode = "mro_replacement"
        elif visual_preset:
            intent_mode = "visual_similarity"
        else:
            intent_mode = "discovery"

        return ParsedConversationalIntent(
            raw_query=query,
            detected_domain=detected_domain,
            detected_category=detected_category,
            max_price=max_price,
            min_price=min_price,
            in_stock_only=in_stock_only,
            extracted_tags=extracted_tags,
            compatibility_anchor_ids=compat_anchors,
            intent_mode=intent_mode,
            visual_preset_used=visual_preset,
        )

    def execute_funnel(
        self,
        query: str,
        persona_id: str = "alpine_mountaineer",
        visual_preset: str | None = None,
        explicit_domain: str | None = None,
        explicit_category: str | None = None,
        explicit_max_price: float | None = None,
        top_k: int = 6,
        lambda_diversity: float = 0.72,
        bandit_exploration_weight: float = 0.12,
        in_stock_only: bool = True,
    ) -> dict[str, Any]:
        """Run the complete 4-stage multimodal recommendation funnel and return trace + results."""
        t0 = time.perf_counter()
        base_persona = self.catalog_store.get_persona(persona_id)

        # Stage 0 / Node 1: Parse Conversational Intent
        t_parse_start = time.perf_counter()
        parsed_intent = self.parse_intent(
            query=query,
            explicit_domain=explicit_domain,
            explicit_category=explicit_category,
            explicit_max_price=explicit_max_price,
            visual_preset=visual_preset,
            in_stock_only=in_stock_only,
        )
        parse_ms = round((time.perf_counter() - t_parse_start) * 1000.0, 2)

        # Node 2: Extract Multimodal Query Descriptors (32-D Visual + 32-D Semantic)
        t_embed_start = time.perf_counter()
        visual_sig: dict[str, Any] | None = None
        query_vis_vec: np.ndarray | None = None
        if visual_preset:
            visual_sig = extract_query_visual_signature(visual_preset)
            query_vis_vec = np.asarray(visual_sig["visual_embedding"], dtype=np.float32)

        effective_domain = parsed_intent.detected_domain or base_persona.preferred_domain
        effective_category = parsed_intent.detected_category or (
            "Alpine Shells" if effective_domain == "retail" else "Hydraulic Seals"
        )
        query_sem_vec = extract_semantic_descriptor(
            sku=f"QUERY-{hash(query) % 10000}",
            domain=effective_domain,
            category=effective_category,
            tags=parsed_intent.extracted_tags or base_persona.intent_keywords,
            unit_price=parsed_intent.max_price or base_persona.budget_ceiling,
            gross_margin_pct=60.0,
        )
        embed_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 2)

        # Pull live session clickstream trajectory from NoSQL Online Feature Store
        session_doc = self.online_store.get_session(base_persona.user_id)
        live_sequence = list(session_doc.last_5_clicks)

        # Augment persona session with compatibility anchors if the user specified "pairs with..."
        active_persona = base_persona
        merged_recent = list(
            dict.fromkeys(
                parsed_intent.compatibility_anchor_ids
                + list(reversed(live_sequence))
                + base_persona.recent_item_ids
            )
        )
        from src.core.catalog import UserPersona

        active_persona = UserPersona(
            user_id=base_persona.user_id,
            name=base_persona.name,
            role=base_persona.role,
            preferred_domain=effective_domain,
            budget_ceiling=parsed_intent.max_price or base_persona.budget_ceiling,
            price_sensitivity=base_persona.price_sensitivity,
            margin_affinity=base_persona.margin_affinity,
            recent_item_ids=merged_recent[:6],
            intent_keywords=list(
                dict.fromkeys(parsed_intent.extracted_tags + base_persona.intent_keywords)
            ),
            description=base_persona.description,
        )

        # Compute SASRec Causal Self-Attention Next-Item Transition Scores
        sasrec_out = self.sasrec.predict_next_items(session_item_ids=live_sequence)

        # Stage 1 & 2 / Node 3: Hybrid Retrieval (Payload Filter + Two-Tower Dense + BM25 + RRF)
        t_ret_start = time.perf_counter()
        payload_filter = PayloadFilter(
            domain=parsed_intent.detected_domain,
            category=explicit_category,
            max_price=parsed_intent.max_price,
            min_price=parsed_intent.min_price,
            in_stock_only=in_stock_only,
        )
        retrieval_out = self.retriever.retrieve(
            persona=active_persona,
            query_text=query,
            payload_filter=payload_filter,
            query_visual_vec=query_vis_vec,
            query_semantic_vec=query_sem_vec,
            top_k=max(18, top_k * 3),
        )
        retrieve_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

        # Stage 3 & 4 / Node 4: Cross-Encoder + SASRec + Thompson Bandit + MMR Diversity Reranker
        t_rerank_start = time.perf_counter()
        rerank_out = self.reranker.rerank_with_mmr(
            candidates=retrieval_out["candidates"],
            persona=active_persona,
            top_k=top_k,
            lambda_diversity=lambda_diversity,
            bandit_exploration_weight=bandit_exploration_weight,
            price_ceiling=parsed_intent.max_price,
            sasrec_transition_scores=sasrec_out.item_transition_scores,
        )
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        recs: list[RerankedRecommendation] = rerank_out["recommendations"]
        synthesized_answer = self._synthesize_assistant_brief(
            parsed_intent=parsed_intent,
            recommendations=recs,
            ild=rerank_out["intra_list_diversity"],
        )

        return {
            "persona": {
                "user_id": active_persona.user_id,
                "name": active_persona.name,
                "role": active_persona.role,
                "preferred_domain": active_persona.preferred_domain,
                "budget_ceiling": active_persona.budget_ceiling,
                "recent_item_ids": active_persona.recent_item_ids,
            },
            "online_session_state": session_doc.to_dict(),
            "sasrec_sequential_prediction": sasrec_out.to_dict(),
            "parsed_intent": parsed_intent.to_dict(),
            "visual_query_signature": visual_sig,
            "funnel_telemetry": {
                "stage0_total_catalog": retrieval_out["total_catalog_size"],
                "stage1_filtered_candidates": retrieval_out["filtered_candidate_count"],
                "stage2_hybrid_rrf_retrieved": retrieval_out["retrieved_count"],
                "stage3_cross_encoder_scored": retrieval_out["retrieved_count"],
                "stage4_mmr_final_selected": len(recs),
                "intra_list_diversity": rerank_out["intra_list_diversity"],
                "unique_categories_surfaced": rerank_out["unique_categories_count"],
                "cold_start_items_surfaced": rerank_out["cold_start_items_surfaced"],
                "mean_expected_margin_usd": rerank_out["mean_expected_margin_usd"],
                "latency_breakdown_ms": {
                    "intent_parsing_ms": parse_ms,
                    "multimodal_encoding_ms": embed_ms,
                    "hybrid_rrf_retrieval_ms": retrieve_ms,
                    "cross_encoder_mmr_rerank_ms": rerank_ms,
                    "total_pipeline_ms": total_ms,
                },
            },
            "qdrant_filter_payload": retrieval_out["qdrant_payload_filter"],
            "user_tower_embedding_preview": retrieval_out["user_embedding_64d"][:12],
            "assistant_synthesis": synthesized_answer,
            "recommendations": [r.to_dict() for r in recs],
        }

    @staticmethod
    def _synthesize_assistant_brief(
        parsed_intent: ParsedConversationalIntent,
        recommendations: list[RerankedRecommendation],
        ild: float,
    ) -> str:
        if not recommendations:
            return (
                "No in-stock items matched your exact price and specification constraints. "
                "Try raising the maximum price filter or broadening the category."
            )
        top = recommendations[0]
        price_clause = (
            f"under ${parsed_intent.max_price:.0f}"
            if parsed_intent.max_price is not None
            else "within your target budget"
        )
        compat_note = (
            f" with verified compatibility across {', '.join(top.compatibility_matches)}"
            if top.compatibility_matches
            else ""
        )
        return (
            f"Top match {price_clause}: {top.item.title} (${top.item.unit_price:.2f}, "
            f"{top.item.gross_margin_pct:.1f}% gross margin){compat_note}. "
            f"Ranked {len(recommendations)} items via Two-Tower + BM25 RRF fusion and MMR "
            f"diversification (Intra-List Diversity = {ild:.2f})."
        )
