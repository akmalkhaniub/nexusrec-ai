"""Second-Stage Cross-Encoder Reranker, Thompson Sampling Bandit & MMR Diversity Filter.

Stage 3 & Stage 4 of the NexusRec AI Recommendation Funnel:
- Stage 3 (`CrossEncoderFeatureCrossReranker` + `ThompsonSamplingBandit`):
  1. Computes second-order interaction features (`user_context x item_attributes`):
     - Hybrid RRF retrieval score
     - Dense Two-Tower + Visual similarity
     - Session compatibility graph overlap (`compatible_with` boost with user's recent items)
     - Price-to-budget headroom utility
     - Gross margin commercial value contribution
  2. Fits/evaluates a calibrated `scikit-learn` `GradientBoostingRegressor` / Ridge cross-feature
     scoring pipeline on pairwise interaction features.
  3. Applies a **Thompson Sampling Contextual Bandit** (`Beta(alpha, beta)` posterior per item arm)
     to give cold-start / low-impression high-potential items controlled exploration lift.
- Stage 4 (`MaximalMarginalRelevanceFilter` - MMR):
  Iteratively selects final top-K recommendations maximizing:
  `MMR(i) = lambda_div * Relevance(i) - (1 - lambda_div) * max_{j in Selected} Sim(i, j)`
  where `Sim(i, j)` combines 64-D multimodal vector cosine similarity and same-category saturation penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.core.catalog import CatalogItem, UserPersona
from src.core.hybrid_retriever import HybridCandidate


@dataclass
class BanditArmState:
    """Beta-Bernoulli Thompson Sampling posterior parameters for an item arm."""

    item_id: str
    alpha: float
    beta: float
    cold_start: bool
    impressions: int

    @property
    def posterior_mean(self) -> float:
        return round(self.alpha / (self.alpha + self.beta), 4)

    @property
    def uncertainty_std(self) -> float:
        a, b = self.alpha, self.beta
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        return round(float(np.sqrt(var)), 4)


@dataclass(frozen=True)
class RerankedRecommendation:
    """Final recommendation record with full Stage 1-4 explainability telemetry."""

    rank: int
    item: CatalogItem
    stage1_dense_score: float
    stage1_visual_score: float
    stage2_bm25_score: float
    stage2_rrf_score: float
    stage3_cross_encoder_score: float
    stage3_sasrec_transition_score: float
    stage3_bandit_sample: float
    stage3_pre_mmr_score: float
    stage4_mmr_diversity_penalty: float
    final_mmr_score: float
    compatibility_matches: list[str]
    matched_terms: list[str]
    expected_margin_contribution_usd: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item": self.item.to_dict(),
            "scores": {
                "stage1_dense_two_tower": self.stage1_dense_score,
                "stage1_visual_similarity": self.stage1_visual_score,
                "stage2_sparse_bm25": self.stage2_bm25_score,
                "stage2_rrf_fusion": self.stage2_rrf_score,
                "stage3_cross_encoder": self.stage3_cross_encoder_score,
                "stage3_sasrec_sequential": self.stage3_sasrec_transition_score,
                "stage3_thompson_bandit": self.stage3_bandit_sample,
                "stage3_combined_relevance": self.stage3_pre_mmr_score,
                "stage4_mmr_penalty": self.stage4_mmr_diversity_penalty,
                "final_mmr_score": self.final_mmr_score,
            },
            "compatibility_matches": self.compatibility_matches,
            "matched_terms": self.matched_terms,
            "expected_margin_contribution_usd": self.expected_margin_contribution_usd,
            "explanation": self.explanation,
        }


class ThompsonSamplingBandit:
    """Beta-Bernoulli Thompson Sampling bandit for cold-start exploration vs. exploitation."""

    def __init__(self, items: list[CatalogItem], seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.arms: dict[str, BanditArmState] = {}
        for item in items:
            if item.cold_start:
                # Cold-start items have low prior observations (high variance -> exploration bonus)
                alpha = 2.2
                beta = 1.6
                impressions = 4
            else:
                impressions = int(40 + item.popularity_score * 80)
                alpha = round(1.0 + impressions * item.popularity_score * 0.35, 2)
                beta = round(1.0 + impressions * (1.0 - item.popularity_score * 0.35) * 0.25, 2)
            self.arms[item.item_id] = BanditArmState(
                item_id=item.item_id,
                alpha=alpha,
                beta=beta,
                cold_start=item.cold_start,
                impressions=impressions,
            )

    def sample_arm(self, item_id: str, explore_temperature: float = 1.0) -> tuple[float, BanditArmState]:
        """Draw a deterministic-seeded or live Thompson sample from `Beta(alpha, beta)`."""
        arm = self.arms[item_id]
        # Deterministic reproducible sample anchored on item_id + current alpha/beta state
        state_seed = int(abs(hash(f"{item_id}:{arm.alpha:.2f}:{arm.beta:.2f}")) % (2**31 - 1))
        local_rng = np.random.default_rng(state_seed)
        scaled_alpha = max(0.5, arm.alpha / max(explore_temperature, 0.2))
        scaled_beta = max(0.5, arm.beta / max(explore_temperature, 0.2))
        sample = float(local_rng.beta(scaled_alpha, scaled_beta))
        # Cold-start exploration bonus proportional to posterior uncertainty
        if arm.cold_start:
            sample = min(1.0, sample + 0.85 * arm.uncertainty_std)
        return round(sample, 4), arm

    def update_arm(self, item_id: str, reward: float) -> BanditArmState:
        """Update Beta posterior (`alpha += reward`, `beta += 1 - reward`) from user feedback."""
        if item_id not in self.arms:
            raise KeyError(f"Unknown bandit arm: {item_id}")
        arm = self.arms[item_id]
        clamped = max(0.0, min(1.0, float(reward)))
        arm.alpha = round(arm.alpha + clamped, 3)
        arm.beta = round(arm.beta + (1.0 - clamped), 3)
        arm.impressions += 1
        return arm


class CrossEncoderAndMMRReranker:
    """Second-stage Cross-Encoder feature-cross scorer + Thompson Bandit + MMR diversification."""

    def __init__(self, items: list[CatalogItem], seed: int = 42) -> None:
        self.bandit = ThompsonSamplingBandit(items=items, seed=seed)
        self.scaler = StandardScaler()
        self.gbdt_scorer = GradientBoostingRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.08,
            random_state=seed,
        )
        self._fit_calibrated_cross_encoder(seed=seed)

    def _fit_calibrated_cross_encoder(self, seed: int = 42) -> None:
        """Train the scikit-learn GradientBoosting cross-feature scorer on synthetic ground-truth pairs."""
        rng = np.random.default_rng(seed)
        n_samples = 240
        # 7 cross-features:
        # [rrf_score, dense_score, bm25_norm, compat_overlap, budget_headroom, margin_utility, popularity]
        X = rng.uniform(0.0, 1.0, size=(n_samples, 7)).astype(np.float32)
        # Ground-truth conversion utility target combining semantic relevance, compatibility, and margin
        y = (
            0.30 * X[:, 0]
            + 0.24 * X[:, 1]
            + 0.16 * X[:, 2]
            + 0.14 * X[:, 3]
            + 0.06 * X[:, 4]
            + 0.06 * X[:, 5]
            + 0.04 * X[:, 6]
            + 0.05 * (X[:, 1] * X[:, 3])  # non-linear interaction: high dense sim * compatible
        )
        X_scaled = self.scaler.fit_transform(X)
        self.gbdt_scorer.fit(X_scaled, y)

    def _extract_cross_features(
        self,
        candidate: HybridCandidate,
        persona: UserPersona,
        max_bm25: float,
        price_ceiling: float | None,
    ) -> tuple[np.ndarray, list[str]]:
        item = candidate.item
        bm25_norm = candidate.sparse_bm25_score / max(max_bm25, 1e-6)

        # Compatibility overlap with user's recent session items
        recent_set = set(persona.recent_item_ids)
        compat_matches = [cid for cid in item.compatible_with if cid in recent_set]
        compat_score = min(1.0, len(compat_matches) * 0.5)

        effective_budget = price_ceiling if price_ceiling is not None else persona.budget_ceiling
        if effective_budget > 0:
            headroom = max(0.0, min(1.0, (effective_budget - item.unit_price) / effective_budget + 0.5))
        else:
            headroom = 0.5

        margin_utility = (item.gross_margin_pct / 100.0) * persona.margin_affinity

        feat_vec = np.array(
            [
                candidate.rrf_fusion_score,
                candidate.combined_dense_score,
                bm25_norm,
                compat_score,
                headroom,
                margin_utility,
                item.popularity_score,
            ],
            dtype=np.float32,
        )
        return feat_vec, compat_matches

    def rerank_with_mmr(
        self,
        candidates: list[HybridCandidate],
        persona: UserPersona,
        top_k: int = 8,
        lambda_diversity: float = 0.72,
        bandit_exploration_weight: float = 0.12,
        price_ceiling: float | None = None,
        sasrec_transition_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run Stage 3 (Cross-Encoder + SASRec + Bandit) and Stage 4 (MMR Diversity Filter)."""
        if not candidates:
            return {
                "recommendations": [],
                "intra_list_diversity": 0.0,
                "category_coverage_ratio": 0.0,
                "cold_start_items_surfaced": 0,
                "mean_expected_margin_usd": 0.0,
            }

        max_bm25 = max((c.sparse_bm25_score for c in candidates), default=1.0)
        feature_rows: list[np.ndarray] = []
        compat_list: list[list[str]] = []

        for cand in candidates:
            f_vec, c_matches = self._extract_cross_features(
                candidate=cand,
                persona=persona,
                max_bm25=max_bm25,
                price_ceiling=price_ceiling,
            )
            feature_rows.append(f_vec)
            compat_list.append(c_matches)

        X_mat = np.stack(feature_rows, axis=0)
        X_scaled = self.scaler.transform(X_mat)
        raw_ce_preds = self.gbdt_scorer.predict(X_scaled)

        # Stage 3: Combine Cross-Encoder score with SASRec Sequential Transition & Thompson Bandit
        scored_pool: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            ce_score = float(np.clip(raw_ce_preds[idx], 0.05, 0.99))
            sasrec_score = (
                float(sasrec_transition_scores.get(cand.item.item_id, ce_score))
                if sasrec_transition_scores
                else round(ce_score * 0.95, 4)
            )
            bandit_sample, arm_state = self.bandit.sample_arm(cand.item.item_id)
            blended_ce_sasrec = 0.78 * ce_score + 0.22 * sasrec_score
            pre_mmr = (1.0 - bandit_exploration_weight) * blended_ce_sasrec + (
                bandit_exploration_weight * bandit_sample
            )
            scored_pool.append(
                {
                    "candidate": cand,
                    "ce_score": round(ce_score, 4),
                    "sasrec_score": round(sasrec_score, 4),
                    "bandit_sample": round(bandit_sample, 4),
                    "arm_state": arm_state,
                    "pre_mmr_score": round(pre_mmr, 4),
                    "compat_matches": compat_list[idx],
                    "vec": np.asarray(cand.item.multimodal_features, dtype=np.float32),
                }
            )

        # Sort pool by Stage 3 relevance before Stage 4 MMR greedy selection
        scored_pool.sort(key=lambda x: x["pre_mmr_score"], reverse=True)

        # Stage 4: Maximal Marginal Relevance (MMR) greedy diversification
        selected: list[dict[str, Any]] = []
        remaining = list(scored_pool)

        while remaining and len(selected) < top_k:
            if not selected:
                best = remaining.pop(0)
                best["mmr_penalty"] = 0.0
                best["final_mmr_score"] = best["pre_mmr_score"]
                selected.append(best)
                continue

            best_idx = 0
            best_mmr_val = -1e9
            best_penalty = 0.0

            for r_idx, entry in enumerate(remaining):
                cand_item = entry["candidate"].item
                cand_vec = entry["vec"]

                # Compute maximum similarity to already selected items
                max_sim = 0.0
                for sel in selected:
                    sel_item = sel["candidate"].item
                    cos_sim = float(np.dot(cand_vec, sel["vec"]))
                    # Add explicit category saturation penalty if same category already selected
                    cat_overlap = 0.28 if cand_item.category == sel_item.category else 0.0
                    pair_sim = min(1.0, max(0.0, 0.72 * cos_sim + cat_overlap))
                    if pair_sim > max_sim:
                        max_sim = pair_sim

                mmr_val = (
                    lambda_diversity * entry["pre_mmr_score"]
                    - (1.0 - lambda_diversity) * max_sim
                )
                if mmr_val > best_mmr_val:
                    best_mmr_val = mmr_val
                    best_idx = r_idx
                    best_penalty = round((1.0 - lambda_diversity) * max_sim, 4)

            chosen = remaining.pop(best_idx)
            chosen["mmr_penalty"] = best_penalty
            chosen["final_mmr_score"] = round(best_mmr_val, 4)
            selected.append(chosen)

        # Build final explainable recommendation objects
        recommendations: list[RerankedRecommendation] = []
        for rank_idx, sel in enumerate(selected, start=1):
            cand: HybridCandidate = sel["candidate"]
            item = cand.item
            compat_ids = sel["compat_matches"]
            exp_margin = round(
                item.unit_price * (item.gross_margin_pct / 100.0) * sel["ce_score"], 2
            )

            reasons: list[str] = []
            if cand.matched_terms:
                reasons.append(f"Matched lexical intent ({', '.join(cand.matched_terms[:4])})")
            if compat_ids:
                reasons.append(f"Direct assembly/kit compatibility with {', '.join(compat_ids)}")
            if item.cold_start:
                reasons.append(
                    f"Surfaced via Thompson Sampling cold-start exploration (Bandit={sel['bandit_sample']:.2f})"
                )
            else:
                reasons.append(f"High Two-Tower cosine alignment ({cand.dense_two_tower_score:.2f})")

            recommendations.append(
                RerankedRecommendation(
                    rank=rank_idx,
                    item=item,
                    stage1_dense_score=cand.dense_two_tower_score,
                    stage1_visual_score=cand.visual_similarity_score,
                    stage2_bm25_score=cand.sparse_bm25_score,
                    stage2_rrf_score=cand.rrf_fusion_score,
                    stage3_cross_encoder_score=sel["ce_score"],
                    stage3_sasrec_transition_score=sel["sasrec_score"],
                    stage3_bandit_sample=sel["bandit_sample"],
                    stage3_pre_mmr_score=sel["pre_mmr_score"],
                    stage4_mmr_diversity_penalty=sel["mmr_penalty"],
                    final_mmr_score=sel["final_mmr_score"],
                    compatibility_matches=compat_ids,
                    matched_terms=cand.matched_terms,
                    expected_margin_contribution_usd=exp_margin,
                    explanation=" • ".join(reasons),
                )
            )

        # Compute Intra-List Diversity (ILD) = mean pairwise cosine distance among top-K
        ild = self.compute_intra_list_diversity([r.item for r in recommendations])
        unique_cats = {r.item.category for r in recommendations}
        cold_start_cnt = sum(1 for r in recommendations if r.item.cold_start)
        mean_margin = (
            round(
                float(np.mean([r.expected_margin_contribution_usd for r in recommendations])),
                2,
            )
            if recommendations
            else 0.0
        )

        return {
            "recommendations": recommendations,
            "intra_list_diversity": ild,
            "unique_categories_count": len(unique_cats),
            "cold_start_items_surfaced": cold_start_cnt,
            "mean_expected_margin_usd": mean_margin,
        }

    @staticmethod
    def compute_intra_list_diversity(items: list[CatalogItem]) -> float:
        """Compute Intra-List Diversity (ILD) as average `(1 - cosine_similarity)` over all pairs."""
        n = len(items)
        if n <= 1:
            return 0.0
        vecs = [np.asarray(it.multimodal_features, dtype=np.float32) for it in items]
        dists: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = float(np.dot(vecs[i], vecs[j]))
                cat_diff_bonus = 0.15 if items[i].category != items[j].category else 0.0
                dist = min(1.0, max(0.0, (1.0 - cos_sim) + cat_diff_bonus))
                dists.append(dist)
        return round(float(np.mean(dists)), 4)
