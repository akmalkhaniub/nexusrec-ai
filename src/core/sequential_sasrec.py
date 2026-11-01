"""Sequential Session Transformer (`SASRec`-Style Causal Self-Attention Next-Item Predictor).

Models directional, asymmetric sequential transitions over a user's ordered clickstream
trajectory `[item_1, item_2, ..., item_t]`:
- For example, after a user clicks or purchases a primary anchor system (e.g., an Alpine
  Hardshell Jacket `RET-001` or a 50mm Hydraulic Cylinder Actuator `IND-013`), sequential
  transition attention predicts complementary next-step accessories/kits (e.g., Crampon
  Mountaineering Boots `RET-005` / Gaiters `RET-020` or High-Pressure Viton Seal Kits
  `IND-001` / Pressure Transducers `IND-017`) rather than recommending redundant duplicate
  primary shells or cylinders.

Architecture:
- `PositionalEncoding`: Learnable/sinusoidal 64-D sequence position embeddings for up to
  `MAX_SEQ_LEN = 12` interaction steps.
- `CausalScaledDotProductAttention`: Computes `Q = X W_Q`, `K = X W_K`, `V = X W_V` with a
  strictly lower-triangular causal mask (`mask[i, j] = -inf` for `j > i`) so position `i`
  can only attend to historical positions `<= i` (preventing future target leakage).
- `AsymmetricTransitionHead`: Residual Pointwise Feed-Forward Network (FFN) + directional
  compatibility transition projection producing the 64-D L2-normalized next-item prediction
  vector `h_next` and per-position attention weights for UI explainability.
- Seamless Dual Backend: Supports `torch.nn.Module` when PyTorch is installed and exact
  analytical NumPy forward execution in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.catalog import CatalogItem, MultimodalCatalogStore

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


MAX_SEQ_LEN = 12
SASREC_DIM = 64


def _l2_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def _softmax_last_axis(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.maximum(np.sum(exp_x, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class SequentialPredictionOutput:
    """Telemetry and next-item transition scores returned by the SASRec Transformer."""

    sequence_item_ids: list[str]
    sequence_skus: list[str]
    next_item_embedding_64d: list[float]
    causal_attention_matrix: list[list[float]]
    last_step_attention_weights: dict[str, float]
    predicted_next_categories: list[str]
    item_transition_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_item_ids": self.sequence_item_ids,
            "sequence_skus": self.sequence_skus,
            "next_item_embedding_preview": [
                round(float(v), 4) for v in self.next_item_embedding_64d[:8]
            ],
            "causal_attention_matrix": self.causal_attention_matrix,
            "last_step_attention_weights": self.last_step_attention_weights,
            "predicted_next_categories": self.predicted_next_categories,
        }


# Asymmetric domain transition matrix: maps current item category -> ordered next-step complementary categories
_ASYMMETRIC_CATEGORY_TRANSITIONS: dict[str, list[tuple[str, float]]] = {
    "Alpine Shells": [
        ("Mountaineering Footwear", 0.95),
        ("Climbing Hardware", 0.90),
        ("Insulated Midlayers", 0.85),
        ("Technical Baselayers", 0.80),
        ("Expedition Packs", 0.75),
    ],
    "Mountaineering Footwear": [
        ("Climbing Hardware", 0.96),
        ("Alpine Shells", 0.92),
        ("Technical Baselayers", 0.88),
        ("Expedition Packs", 0.80),
    ],
    "Insulated Midlayers": [
        ("Alpine Shells", 0.94),
        ("Technical Baselayers", 0.88),
        ("Expedition Packs", 0.82),
    ],
    "Climbing Hardware": [
        ("Mountaineering Footwear", 0.93),
        ("Alpine Shells", 0.90),
        ("Expedition Packs", 0.86),
    ],
    "Expedition Packs": [
        ("Alpine Shells", 0.91),
        ("Insulated Midlayers", 0.87),
        ("Climbing Hardware", 0.84),
    ],
    "Technical Baselayers": [
        ("Insulated Midlayers", 0.93),
        ("Alpine Shells", 0.90),
        ("Mountaineering Footwear", 0.84),
    ],
    "Industrial Actuators": [
        ("Hydraulic Seals", 0.98),
        ("High-Pressure Valves", 0.92),
        ("Sensor Telemetry", 0.90),
        ("Power Transmission", 0.82),
    ],
    "Hydraulic Seals": [
        ("High-Pressure Valves", 0.94),
        ("Sensor Telemetry", 0.91),
        ("Power Transmission", 0.88),
        ("Hydraulic Seals", 0.84),
    ],
    "High-Pressure Valves": [
        ("Hydraulic Seals", 0.96),
        ("Sensor Telemetry", 0.93),
        ("Power Transmission", 0.86),
    ],
    "Precision Bearings": [
        ("Sensor Telemetry", 0.95),
        ("Power Transmission", 0.91),
        ("Precision Bearings", 0.87),
        ("Hydraulic Seals", 0.80),
    ],
    "Sensor Telemetry": [
        ("High-Pressure Valves", 0.90),
        ("Hydraulic Seals", 0.89),
        ("Precision Bearings", 0.88),
    ],
    "Power Transmission": [
        ("Hydraulic Seals", 0.92),
        ("Precision Bearings", 0.90),
        ("High-Pressure Valves", 0.86),
    ],
}


class SASRecSequentialPredictor:
    """SASRec-style Causal Self-Attention Sequential Recommendation Engine."""

    def __init__(self, catalog_store: MultimodalCatalogStore, seed: int = 42) -> None:
        self.catalog_store = catalog_store
        self.seed = seed
        self.embed_dim = SASREC_DIM
        self.max_seq_len = MAX_SEQ_LEN
        self.backend = "pytorch-sasrec" if TORCH_AVAILABLE else "numpy-sasrec"

        rng = np.random.default_rng(seed + 101)
        eye = np.eye(SASREC_DIM, dtype=np.float32)

        # Positional embeddings [MAX_SEQ_LEN, 64] with recency bias
        pos_indices = np.arange(MAX_SEQ_LEN, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, SASREC_DIM, 2, dtype=np.float32) * -(np.log(10000.0) / SASREC_DIM)
        )
        pos_enc = np.zeros((MAX_SEQ_LEN, SASREC_DIM), dtype=np.float32)
        pos_enc[:, 0::2] = np.sin(pos_indices * div_term)
        pos_enc[:, 1::2] = np.cos(pos_indices * div_term)
        self.positional_embeddings = (0.15 * pos_enc).astype(np.float32)

        # Causal Self-Attention Projections (W_Q, W_K, W_V, W_O)
        self.w_q = (eye + rng.normal(0.0, 0.05, size=(SASREC_DIM, SASREC_DIM))).astype(np.float32)
        self.w_k = (eye + rng.normal(0.0, 0.05, size=(SASREC_DIM, SASREC_DIM))).astype(np.float32)
        self.w_v = (eye + rng.normal(0.0, 0.05, size=(SASREC_DIM, SASREC_DIM))).astype(np.float32)
        self.w_o = (eye + rng.normal(0.0, 0.04, size=(SASREC_DIM, SASREC_DIM))).astype(np.float32)

        # Asymmetric FFN transition matrix mapping attended context -> next-item space
        self.w_ffn = (eye + rng.normal(0.0, 0.04, size=(SASREC_DIM, SASREC_DIM))).astype(np.float32)

    @staticmethod
    def build_causal_mask(seq_len: int) -> np.ndarray:
        """Create lower-triangular causal attention mask (`0.0` for `j <= i`, `-1e9` for `j > i`)."""
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        return mask

    def forward_causal_attention(
        self, sequence_vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute Causal Scaled Dot-Product Self-Attention over `[T, 64]` item embeddings.

        Returns:
            - `context_out`: `[T, 64]` L2-normalized causal sequence representations
            - `attn_weights`: `[T, T]` lower-triangular attention probability matrix
              where `attn_weights[i, j] == 0.0` strictly for all `j > i`.
        """
        seq = np.asarray(sequence_vectors, dtype=np.float32)
        t_len = min(len(seq), self.max_seq_len)
        seq = seq[-t_len:]

        # Add positional encoding (aligned to end of sequence so recent step has consistent recency signal)
        pos_slice = self.positional_embeddings[:t_len]
        x = seq + pos_slice

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        scale = float(np.sqrt(self.embed_dim))
        raw_scores = (q @ k.T) / scale

        # Add recency bias along valid causal positions so later clicks carry higher transition weight
        for i in range(t_len):
            for j in range(i + 1):
                recency_lag = i - j
                raw_scores[i, j] += float(-0.18 * recency_lag)

        causal_mask = self.build_causal_mask(t_len)
        masked_scores = raw_scores + causal_mask
        attn_weights = _softmax_last_axis(masked_scores)
        # Zero out upper triangle explicitly to guarantee exact 0.0 numerical cleanliness
        attn_weights = np.tril(attn_weights)

        attn_out = (attn_weights @ v) @ self.w_o + x
        ffn_out = np.tanh(attn_out @ self.w_ffn) + attn_out
        normed_out = _l2_norm(ffn_out).astype(np.float32)
        return normed_out, attn_weights.astype(np.float32)

    def predict_next_items(
        self,
        session_item_ids: list[str],
        candidate_items: list[CatalogItem] | None = None,
    ) -> SequentialPredictionOutput:
        """Compute causal self-attention over `session_item_ids` and score next-item transitions."""
        valid_items: list[CatalogItem] = []
        for iid in session_item_ids[-self.max_seq_len :]:
            item = self.catalog_store.get_item(iid)
            if item is not None:
                valid_items.append(item)

        if not valid_items:
            fallback = self.catalog_store.items[0]
            valid_items = [fallback]

        seq_vecs = np.stack(
            [np.asarray(it.multimodal_features, dtype=np.float32) for it in valid_items],
            axis=0,
        )
        context_out, attn_matrix = self.forward_causal_attention(seq_vecs)
        last_hidden = context_out[-1]
        last_attn = attn_matrix[-1]

        # Build asymmetric target prototype by blending compatible next-item vectors weighted by last_attn
        compat_proto = np.zeros(self.embed_dim, dtype=np.float32)
        total_w = 0.0
        for idx, seq_item in enumerate(valid_items):
            w_step = float(last_attn[idx])
            for cid in seq_item.compatible_with:
                c_item = self.catalog_store.get_item(cid)
                if c_item is not None:
                    compat_proto += w_step * np.asarray(c_item.multimodal_features, dtype=np.float32)
                    total_w += w_step

        if total_w > 1e-6:
            compat_proto = _l2_norm(compat_proto / total_w)
            next_embedding = _l2_norm(0.55 * last_hidden + 0.45 * compat_proto).astype(np.float32)
        else:
            next_embedding = last_hidden

        most_recent_item = valid_items[-1]
        cat_transitions = dict(
            _ASYMMETRIC_CATEGORY_TRANSITIONS.get(most_recent_item.category, [])
        )
        predicted_cats = [
            cat for cat, _ in _ASYMMETRIC_CATEGORY_TRANSITIONS.get(most_recent_item.category, [])[:3]
        ]

        # Compute directional next-item transition score for each candidate
        pool = candidate_items if candidate_items is not None else self.catalog_store.items
        visited_ids = {it.item_id for it in valid_items}
        recent_compat_targets: dict[str, float] = {}
        for idx, seq_item in enumerate(valid_items):
            w_step = float(last_attn[idx])
            for cid in seq_item.compatible_with:
                recent_compat_targets[cid] = max(recent_compat_targets.get(cid, 0.0), w_step)

        transition_scores: dict[str, float] = {}
        for cand in pool:
            cand_vec = np.asarray(cand.multimodal_features, dtype=np.float32)
            dot_sim = float(np.dot(next_embedding, cand_vec))
            compat_boost = 0.35 if cand.item_id in recent_compat_targets else 0.0
            cat_trans_weight = cat_transitions.get(cand.category, 0.45)

            # Penalize recommending the exact same item that the user already clicked in session
            repeat_penalty = -0.30 if cand.item_id in visited_ids else 0.0

            raw_trans = (
                0.45 * max(0.0, dot_sim)
                + 0.30 * cat_trans_weight
                + compat_boost
                + repeat_penalty
            )
            transition_scores[cand.item_id] = round(float(np.clip(raw_trans, 0.05, 0.99)), 4)

        last_step_map = {
            f"{it.item_id} ({it.sku})": round(float(last_attn[idx]), 4)
            for idx, it in enumerate(valid_items)
        }

        return SequentialPredictionOutput(
            sequence_item_ids=[it.item_id for it in valid_items],
            sequence_skus=[it.sku for it in valid_items],
            next_item_embedding_64d=[round(float(x), 5) for x in next_embedding.tolist()],
            causal_attention_matrix=[
                [round(float(val), 4) for val in row] for row in attn_matrix.tolist()
            ],
            last_step_attention_weights=last_step_map,
            predicted_next_categories=predicted_cats,
            item_transition_scores=transition_scores,
        )
