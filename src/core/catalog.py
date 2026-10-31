"""Multimodal Catalog & Deterministic OpenCV/NumPy Visual Feature Extractor.

Provides a realistic 44-item enterprise catalog spanning two primary domains:
1. Retail Apparel & Technical Alpine Gear (`domain="retail"`)
2. Industrial Spare Parts & Hydraulic/Mechanical Components (`domain="industrial"`)

Each catalog item includes:
- Structured commercial telemetry (`unit_price`, `gross_margin_pct`, `in_stock`, `stock_qty`)
- Rich semantic description and technical attributes/tags
- Deterministic 32-dim Visual Descriptor (`visual_embedding`) combining simulated HSV color
  histograms, Sobel edge orientation histograms, and Gabor surface texture descriptors
- Deterministic 32-dim Semantic Descriptor (`semantic_embedding`) derived from domain
  taxonomy, technical specs, and compatibility clusters
- Combined 64-dim raw multimodal feature vector (`multimodal_features`) fed into the ItemTower
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CatalogItem:
    """Immutable multimodal catalog item with visual, semantic, and commercial attributes."""

    item_id: str
    sku: str
    title: str
    domain: str  # "retail" | "industrial"
    category: str
    subcategory: str
    brand: str
    unit_price: float
    gross_margin_pct: float
    in_stock: bool
    stock_qty: int
    description: str
    attributes: dict[str, str]
    tags: list[str]
    compatible_with: list[str]
    color_hex: str
    surface_finish: str  # e.g., "gore-tex-ripstop", "cnc-anodized-steel", "viton-elastomer"
    popularity_score: float  # 0.0 to 1.0 historical conversion popularity
    cold_start: bool = False
    visual_embedding: list[float] = field(default_factory=list)
    semantic_embedding: list[float] = field(default_factory=list)
    multimodal_features: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize item metadata and summary feature statistics."""
        return {
            "item_id": self.item_id,
            "sku": self.sku,
            "title": self.title,
            "domain": self.domain,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "unit_price": self.unit_price,
            "gross_margin_pct": self.gross_margin_pct,
            "unit_margin_usd": round(self.unit_price * (self.gross_margin_pct / 100.0), 2),
            "in_stock": self.in_stock,
            "stock_qty": self.stock_qty,
            "description": self.description,
            "attributes": self.attributes,
            "tags": self.tags,
            "compatible_with": self.compatible_with,
            "color_hex": self.color_hex,
            "surface_finish": self.surface_finish,
            "popularity_score": self.popularity_score,
            "cold_start": self.cold_start,
            "visual_norm": round(float(np.linalg.norm(self.visual_embedding)), 4),
            "semantic_norm": round(float(np.linalg.norm(self.semantic_embedding)), 4),
        }


@dataclass(frozen=True)
class UserPersona:
    """Benchmark buyer persona with historical click/purchase session sequences."""

    user_id: str
    name: str
    role: str
    preferred_domain: str
    budget_ceiling: float
    price_sensitivity: float  # 0.0 (premium buyer) to 1.0 (value hunter)
    margin_affinity: float
    recent_item_ids: list[str]
    intent_keywords: list[str]
    description: str


# Domain/category latent anchors to ensure meaningful geometric clustering in feature space
_CATEGORY_LATENT_SEEDS: dict[str, int] = {
    "Alpine Shells": 101,
    "Mountaineering Footwear": 102,
    "Insulated Midlayers": 103,
    "Climbing Hardware": 104,
    "Expedition Packs": 105,
    "Technical Baselayers": 106,
    "Hydraulic Seals": 201,
    "High-Pressure Valves": 202,
    "Precision Bearings": 203,
    "Industrial Actuators": 204,
    "Sensor Telemetry": 205,
    "Power Transmission": 206,
}

_FINISH_TEXTURE_PROFILES: dict[str, tuple[float, float, float]] = {
    # (edge_sharpness, specular_reflectance, surface_roughness)
    "gore-tex-3l-ripstop": (0.72, 0.25, 0.68),
    "cordura-ballistic-weave": (0.85, 0.18, 0.88),
    "vibram-lug-rubber": (0.91, 0.12, 0.94),
    "pertex-quantum-ultralight": (0.45, 0.62, 0.28),
    "forged-7075-aluminum": (0.94, 0.85, 0.19),
    "merino-grid-fleece": (0.35, 0.10, 0.76),
    "viton-fluoroelastomer": (0.65, 0.30, 0.22),
    "cnc-316-stainless": (0.98, 0.92, 0.11),
    "ceramic-hybrid-nitride": (0.96, 0.89, 0.06),
    "hard-anodized-alloy": (0.88, 0.74, 0.25),
    "ptfe-teflon-composite": (0.58, 0.48, 0.14),
    "ip68-potted-polymer": (0.76, 0.40, 0.35),
}


def _hex_to_rgb_normalized(color_hex: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' hex string into normalized [0, 1] RGB channels."""
    clean = color_hex.lstrip("#")
    if len(clean) != 6:
        return (0.5, 0.5, 0.5)
    r = int(clean[0:2], 16) / 255.0
    g = int(clean[2:4], 16) / 255.0
    b = int(clean[4:6], 16) / 255.0
    return (r, g, b)


def extract_visual_descriptor(
    sku: str,
    domain: str,
    category: str,
    color_hex: str,
    surface_finish: str,
) -> np.ndarray:
    """Compute a deterministic 32-dim OpenCV/NumPy-style visual descriptor.

    Layout of the 32-D visual vector:
    - Dims [0:12]: 12-bin HSV/RGB chromaticity histogram derived from item color_hex
    - Dims [12:22]: 10-bin Sobel edge orientation & structural contour profile by category
    - Dims [22:32]: 10-bin Gabor surface texture & reflectance energy from surface_finish
    """
    r, g, b = _hex_to_rgb_normalized(color_hex)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    saturation = 0.0 if max_c == 0 else delta / max_c
    value = max_c

    # 12-bin color histogram representation
    color_bins = np.array(
        [
            r,
            g,
            b,
            saturation,
            value,
            abs(r - g),
            abs(g - b),
            abs(b - r),
            0.5 * (r + g),
            0.5 * (g + b),
            1.0 if domain == "retail" else 0.2,
            1.0 if domain == "industrial" else 0.2,
        ],
        dtype=np.float32,
    )

    # 10-bin structural contour & Sobel orientation descriptor anchored on category
    cat_seed = _CATEGORY_LATENT_SEEDS.get(category, 999)
    cat_rng = np.random.default_rng(cat_seed)
    sku_hash = int(hashlib.sha256(sku.encode("utf-8")).hexdigest()[:8], 16)
    item_rng = np.random.default_rng(sku_hash)
    contour_bins = cat_rng.normal(loc=0.5, scale=0.25, size=10).astype(np.float32)
    contour_bins += item_rng.normal(loc=0.0, scale=0.04, size=10).astype(np.float32)

    # 10-bin Gabor texture & specular reflectance profile
    sharpness, specular, roughness = _FINISH_TEXTURE_PROFILES.get(
        surface_finish, (0.7, 0.5, 0.5)
    )
    freq_angles = np.linspace(0.2, 1.8, 7, dtype=np.float32)
    gabor_energy = np.sin(freq_angles * sharpness * np.pi) * roughness + 0.3 * specular
    texture_bins = np.concatenate(
        [
            np.array([sharpness, specular, roughness], dtype=np.float32),
            gabor_energy.astype(np.float32),
        ]
    )

    vec = np.concatenate([color_bins, contour_bins, texture_bins])
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.astype(np.float32)


def extract_semantic_descriptor(
    sku: str,
    domain: str,
    category: str,
    tags: list[str],
    unit_price: float,
    gross_margin_pct: float,
) -> np.ndarray:
    """Compute a deterministic 32-dim semantic embedding encoding taxonomy & attributes."""
    cat_seed = _CATEGORY_LATENT_SEEDS.get(category, 500)
    cat_rng = np.random.default_rng(cat_seed * 17)
    base_cat_vec = cat_rng.normal(loc=0.0, scale=0.6, size=32).astype(np.float32)

    # Domain polarity in first 4 dimensions
    if domain == "retail":
        base_cat_vec[0:4] += np.array([1.2, 0.8, -0.9, -0.7], dtype=np.float32)
    else:
        base_cat_vec[0:4] += np.array([-0.9, -0.7, 1.3, 0.9], dtype=np.float32)

    # Accumulate tag embeddings deterministically
    tag_accum = np.zeros(32, dtype=np.float32)
    for tag in tags:
        t_seed = int(hashlib.md5(tag.lower().encode("utf-8")).hexdigest()[:8], 16)
        t_rng = np.random.default_rng(t_seed)
        tag_accum += t_rng.normal(loc=0.0, scale=0.35, size=32).astype(np.float32)

    if tags:
        tag_accum /= np.sqrt(len(tags))

    # Price and margin signal in final 2 dimensions
    price_scaled = min(unit_price / 500.0, 2.0)
    margin_scaled = gross_margin_pct / 100.0
    sku_seed = int(hashlib.md5(sku.encode("utf-8")).hexdigest()[:8], 16)
    jitter = np.random.default_rng(sku_seed).normal(0.0, 0.05, size=32).astype(np.float32)

    vec = base_cat_vec + tag_accum + jitter
    vec[30] = float(price_scaled)
    vec[31] = float(margin_scaled)

    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.astype(np.float32)


def extract_query_visual_signature(preset_name: str) -> dict[str, Any]:
    """Generate a visual query signature from a camera/image upload preset."""
    presets: dict[str, dict[str, Any]] = {
        "alpine_hardshell_jacket": {
            "preset_id": "alpine_hardshell_jacket",
            "label": "Visual Upload: Gore-Tex Alpine Hardshell Jacket (Glacier Blue)",
            "detected_domain": "retail",
            "detected_category": "Alpine Shells",
            "dominant_color_hex": "#1E6B9B",
            "surface_finish": "gore-tex-3l-ripstop",
            "visual_tags": ["waterproof", "gore-tex", "alpine", "shell", "windproof", "breathable"],
            "confidence": 0.96,
        },
        "crampon_mountaineering_boot": {
            "preset_id": "crampon_mountaineering_boot",
            "label": "Visual Upload: Rigid Crampon-Compatible Mountaineering Boot",
            "detected_domain": "retail",
            "detected_category": "Mountaineering Footwear",
            "dominant_color_hex": "#D9531E",
            "surface_finish": "vibram-lug-rubber",
            "visual_tags": ["crampon", "mountaineering", "vibram", "waterproof", "insulated", "boots"],
            "confidence": 0.95,
        },
        "hydraulic_viton_seal_kit": {
            "preset_id": "hydraulic_viton_seal_kit",
            "label": "Visual Upload: High-Pressure Viton Hydraulic Cylinder Seal Kit",
            "detected_domain": "industrial",
            "detected_category": "Hydraulic Seals",
            "dominant_color_hex": "#2B2D2F",
            "surface_finish": "viton-fluoroelastomer",
            "visual_tags": ["hydraulic", "seal", "viton", "high-pressure", "o-ring", "piston"],
            "confidence": 0.98,
        },
        "cnc_stainless_manifold_valve": {
            "preset_id": "cnc_stainless_manifold_valve",
            "label": "Visual Upload: 316 Stainless Proportional Directional Valve",
            "detected_domain": "industrial",
            "detected_category": "High-Pressure Valves",
            "dominant_color_hex": "#8A959E",
            "surface_finish": "cnc-316-stainless",
            "visual_tags": ["valve", "manifold", "stainless", "high-pressure", "hydraulic", "directional"],
            "confidence": 0.94,
        },
        "ceramic_spindle_bearing": {
            "preset_id": "ceramic_spindle_bearing",
            "label": "Visual Upload: ABEC-9 Hybrid Ceramic CNC Spindle Bearing",
            "detected_domain": "industrial",
            "detected_category": "Precision Bearings",
            "dominant_color_hex": "#6E7B85",
            "surface_finish": "ceramic-hybrid-nitride",
            "visual_tags": ["bearing", "ceramic", "spindle", "cnc", "high-speed", "angular-contact"],
            "confidence": 0.97,
        },
    }
    meta = presets.get(preset_name, presets["alpine_hardshell_jacket"])
    vis_vec = extract_visual_descriptor(
        sku=f"QUERY-{meta['preset_id']}",
        domain=meta["detected_domain"],
        category=meta["detected_category"],
        color_hex=meta["dominant_color_hex"],
        surface_finish=meta["surface_finish"],
    )
    return {
        **meta,
        "visual_embedding": vis_vec.tolist(),
    }


_RAW_CATALOG_SPECS: list[dict[str, Any]] = [
    # ==================== RETAIL APPAREL & TECHNICAL ALPINE GEAR (22 items) ====================
    {
        "item_id": "RET-001",
        "sku": "ARC-SHL-001",
        "title": "AetherPro 3L Gore-Tex Alpine Hardshell Jacket",
        "domain": "retail",
        "category": "Alpine Shells",
        "subcategory": "3-Layer Hardshells",
        "brand": "NordicPeak",
        "unit_price": 198.00,
        "gross_margin_pct": 58.0,
        "in_stock": True,
        "stock_qty": 42,
        "description": "Ultralight 3-layer waterproof alpine shell engineered with 40D Gore-Tex Pro ripstop, helmet-compatible storm hood, pit zips, and high-reach articulated gussets for ice climbing and crampon boot ascents.",
        "attributes": {"hydrostatic_head": "28,000mm", "weight": "385g", "membrane": "Gore-Tex 3L"},
        "tags": ["waterproof", "alpine", "shell", "gore-tex", "hardshell", "windproof", "crampon", "climbing", "jacket"],
        "compatible_with": ["RET-005", "RET-006", "RET-009", "RET-013"],
        "color_hex": "#1E6B9B",
        "surface_finish": "gore-tex-3l-ripstop",
        "popularity_score": 0.94,
        "cold_start": False,
    },
    {
        "item_id": "RET-002",
        "sku": "ARC-SHL-002",
        "title": "CirrusRidge Waterproof Alpine Storm Shell",
        "domain": "retail",
        "category": "Alpine Shells",
        "subcategory": "3-Layer Hardshells",
        "brand": "ValhallaGear",
        "unit_price": 215.00,
        "gross_margin_pct": 61.5,
        "in_stock": True,
        "stock_qty": 29,
        "description": "High-altitude waterproof breathable alpine shell under $220 with reinforced Cordura shoulders, harness-friendly Napoleon pockets, and snow gaiter integration for technical mountaineering.",
        "attributes": {"hydrostatic_head": "25,000mm", "weight": "410g", "membrane": "Dermizax EV"},
        "tags": ["waterproof", "alpine", "shell", "mountaineering", "storm", "breathable", "crampon", "jacket"],
        "compatible_with": ["RET-005", "RET-007", "RET-010", "RET-014"],
        "color_hex": "#16537E",
        "surface_finish": "gore-tex-3l-ripstop",
        "popularity_score": 0.89,
        "cold_start": False,
    },
    {
        "item_id": "RET-003",
        "sku": "ARC-SHL-003",
        "title": "K2 Expedition Extreme 8000m Gore-Tex Pro Parka Shell",
        "domain": "retail",
        "category": "Alpine Shells",
        "subcategory": "Expedition Shells",
        "brand": "NordicPeak",
        "unit_price": 465.00,
        "gross_margin_pct": 54.0,
        "in_stock": True,
        "stock_qty": 14,
        "description": "Heavy-duty 80D Gore-Tex Pro expedition shell built for Himalayan mixed faces, featuring RECCO avalanche reflector and two-way Aquaguard zippers.",
        "attributes": {"hydrostatic_head": "30,000mm", "weight": "590g", "membrane": "Gore-Tex Pro 80D"},
        "tags": ["waterproof", "alpine", "shell", "expedition", "gore-tex", "hardshell", "avalanche", "jacket"],
        "compatible_with": ["RET-005", "RET-008", "RET-015"],
        "color_hex": "#D9381E",
        "surface_finish": "gore-tex-3l-ripstop",
        "popularity_score": 0.82,
        "cold_start": False,
    },
    {
        "item_id": "RET-004",
        "sku": "ARC-SHL-004",
        "title": "ZephyrLite Packable Wind & Rain Trail Anorak",
        "domain": "retail",
        "category": "Alpine Shells",
        "subcategory": "Ultralight Shells",
        "brand": "TerraPace",
        "unit_price": 149.00,
        "gross_margin_pct": 64.0,
        "in_stock": True,
        "stock_qty": 55,
        "description": "Sub-200g packable waterproof alpine trail shell with Pertex Shield Air 2.5L laminate for fastpacking, ridge scrambles, and summer alpine approaches.",
        "attributes": {"hydrostatic_head": "20,000mm", "weight": "185g", "membrane": "Pertex Shield Air"},
        "tags": ["waterproof", "alpine", "shell", "ultralight", "rain", "packable", "trail", "jacket"],
        "compatible_with": ["RET-006", "RET-011", "RET-018"],
        "color_hex": "#2A8C68",
        "surface_finish": "pertex-quantum-ultralight",
        "popularity_score": 0.78,
        "cold_start": True,
    },
    {
        "item_id": "RET-005",
        "sku": "FTW-MNT-001",
        "title": "GlacierTech B3 Rigid Crampon Mountaineering Boots",
        "domain": "retail",
        "category": "Mountaineering Footwear",
        "subcategory": "B3 Automatic Crampon Boots",
        "brand": "DolomiteAlps",
        "unit_price": 389.00,
        "gross_margin_pct": 52.0,
        "in_stock": True,
        "stock_qty": 19,
        "description": "Full-shank carbon fiber B3 mountaineering boot with Vibram Mont ice compound, Gore-Tex Insulated Comfort lining, and automatic step-in crampon welts.",
        "attributes": {"crampon_rating": "B3 Automatic", "sole": "Vibram Mont", "insulation": "Primaloft Gold 200g"},
        "tags": ["crampon", "boots", "mountaineering", "vibram", "waterproof", "ice-climbing", "alpine", "footwear"],
        "compatible_with": ["RET-001", "RET-002", "RET-013", "RET-014"],
        "color_hex": "#D9531E",
        "surface_finish": "vibram-lug-rubber",
        "popularity_score": 0.91,
        "cold_start": False,
    },
    {
        "item_id": "RET-006",
        "sku": "FTW-MNT-002",
        "title": "CrestRunner B2 Semi-Rigid Alpine Crampon Boots",
        "domain": "retail",
        "category": "Mountaineering Footwear",
        "subcategory": "B2 Semi-Automatic Boots",
        "brand": "DolomiteAlps",
        "unit_price": 212.00,
        "gross_margin_pct": 56.5,
        "in_stock": True,
        "stock_qty": 34,
        "description": "Versatile B2 semi-automatic crampon boot under $220 combining rock-scrambling agility with rigid heel welt stability for glacier travel and mixed alpine ridges.",
        "attributes": {"crampon_rating": "B2 Semi-Auto", "sole": "Vibram Mulaz", "weight": "640g"},
        "tags": ["crampon", "boots", "alpine", "mountaineering", "waterproof", "vibram", "glacier", "footwear"],
        "compatible_with": ["RET-001", "RET-002", "RET-004", "RET-013"],
        "color_hex": "#C84B31",
        "surface_finish": "vibram-lug-rubber",
        "popularity_score": 0.87,
        "cold_start": False,
    },
    {
        "item_id": "RET-007",
        "sku": "FTW-MNT-003",
        "title": "TalusGrip Gore-Tex Scrambling & Via Ferrata Approach Shoes",
        "domain": "retail",
        "category": "Mountaineering Footwear",
        "subcategory": "Approach Footwear",
        "brand": "ValhallaGear",
        "unit_price": 168.00,
        "gross_margin_pct": 59.0,
        "in_stock": True,
        "stock_qty": 48,
        "description": "Sticky Vibram Megagrip approach shoe with climbing-zone toe rand and waterproof Gore-Tex Extended Comfort membrane for technical boulder approaches.",
        "attributes": {"crampon_rating": "B0 Strap-On", "sole": "Vibram Megagrip", "upper": "1.8mm Suede"},
        "tags": ["approach", "boots", "shoes", "vibram", "waterproof", "climbing", "alpine", "footwear"],
        "compatible_with": ["RET-002", "RET-004", "RET-016"],
        "color_hex": "#3D5A45",
        "surface_finish": "vibram-lug-rubber",
        "popularity_score": 0.83,
        "cold_start": False,
    },
    {
        "item_id": "RET-008",
        "sku": "FTW-MNT-004",
        "title": "Polaris Double-Boot High Altitude 6000m Expedition Boot",
        "domain": "retail",
        "category": "Mountaineering Footwear",
        "subcategory": "Double Expedition Boots",
        "brand": "DolomiteAlps",
        "unit_price": 620.00,
        "gross_margin_pct": 49.0,
        "in_stock": False,
        "stock_qty": 0,
        "description": "Removable thermo-moldable inner boot with integrated Cordura gaiter and carbon honeycomb midsole for extreme sub-zero ascents.",
        "attributes": {"crampon_rating": "B3 Automatic", "temp_rating": "-40C", "weight": "960g"},
        "tags": ["crampon", "boots", "expedition", "mountaineering", "insulated", "alpine"],
        "compatible_with": ["RET-003", "RET-013"],
        "color_hex": "#E03C31",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.72,
        "cold_start": False,
    },
    {
        "item_id": "RET-009",
        "sku": "MID-INS-001",
        "title": "Helios Loft 850-Fill Hydrophobic Down Alpine Hoody",
        "domain": "retail",
        "category": "Insulated Midlayers",
        "subcategory": "Down Insulation",
        "brand": "NordicPeak",
        "unit_price": 189.00,
        "gross_margin_pct": 62.0,
        "in_stock": True,
        "stock_qty": 37,
        "description": "RDS-certified 850-fill Nikwax hydrophobic goose down midlayer engineered to layer cleanly beneath 3L Gore-Tex alpine shells during belays.",
        "attributes": {"fill_power": "850FP Down", "shell_fabric": "Pertex Quantum 10D", "weight": "310g"},
        "tags": ["insulated", "down", "midlayer", "alpine", "hoody", "warm", "packable", "jacket"],
        "compatible_with": ["RET-001", "RET-002", "RET-005"],
        "color_hex": "#E69F00",
        "surface_finish": "pertex-quantum-ultralight",
        "popularity_score": 0.90,
        "cold_start": False,
    },
    {
        "item_id": "RET-010",
        "sku": "MID-INS-002",
        "title": "AeroTherm Coreloft Active Synthetic Belay Jacket",
        "domain": "retail",
        "category": "Insulated Midlayers",
        "subcategory": "Synthetic Active Insulation",
        "brand": "ValhallaGear",
        "unit_price": 174.00,
        "gross_margin_pct": 63.5,
        "in_stock": True,
        "stock_qty": 44,
        "description": "Breathable air-permeable synthetic midlayer with stretch fleece side panels that dumps excess heat during steep crampon ascents.",
        "attributes": {"insulation": "Coreloft 60g/m2", "breathability": "40 CFM", "weight": "340g"},
        "tags": ["insulated", "synthetic", "midlayer", "alpine", "breathable", "climbing", "crampon", "jacket"],
        "compatible_with": ["RET-001", "RET-002", "RET-006"],
        "color_hex": "#2B5B84",
        "surface_finish": "pertex-quantum-ultralight",
        "popularity_score": 0.88,
        "cold_start": False,
    },
    {
        "item_id": "RET-011",
        "sku": "MID-INS-003",
        "title": "FjordGrid Polartec Power Grid Technical Half-Zip Fleece",
        "domain": "retail",
        "category": "Insulated Midlayers",
        "subcategory": "Technical Fleece",
        "brand": "TerraPace",
        "unit_price": 118.00,
        "gross_margin_pct": 66.0,
        "in_stock": True,
        "stock_qty": 63,
        "description": "Geometric bi-component grid fleece that wicks moisture rapidly under waterproof shells while maintaining high warmth-to-weight efficiency.",
        "attributes": {"fabric": "Polartec Power Grid", "weight": "245g", "fit": "Athletic Trim"},
        "tags": ["fleece", "midlayer", "alpine", "breathable", "lightweight", "layering"],
        "compatible_with": ["RET-001", "RET-004", "RET-019"],
        "color_hex": "#3B4D54",
        "surface_finish": "merino-grid-fleece",
        "popularity_score": 0.85,
        "cold_start": False,
    },
    {
        "item_id": "RET-012",
        "sku": "MID-INS-004",
        "title": "Borealis Graphene-Infused Aerogel Alpine Vest",
        "domain": "retail",
        "category": "Insulated Midlayers",
        "subcategory": "Insulated Vests",
        "brand": "NordicPeak",
        "unit_price": 159.00,
        "gross_margin_pct": 65.0,
        "in_stock": True,
        "stock_qty": 22,
        "description": "Next-gen zero-bulk core warmer utilizing encapsulated silica aerogel fibers for windproof thermal regulation under alpine hardshells.",
        "attributes": {"insulation": "PrimaLoft Cross Core Aerogel", "weight": "220g"},
        "tags": ["vest", "insulated", "midlayer", "alpine", "aerogel", "windproof"],
        "compatible_with": ["RET-001", "RET-002"],
        "color_hex": "#243447",
        "surface_finish": "pertex-quantum-ultralight",
        "popularity_score": 0.74,
        "cold_start": True,
    },
    {
        "item_id": "RET-013",
        "sku": "CLM-HRD-001",
        "title": "LynxPro 12-Point Chromoly Modular Ice & Mixed Crampons",
        "domain": "retail",
        "category": "Climbing Hardware",
        "subcategory": "Ice Crampons",
        "brand": "CruxForge",
        "unit_price": 195.00,
        "gross_margin_pct": 55.0,
        "in_stock": True,
        "stock_qty": 27,
        "description": "Forged chromoly steel 12-point crampons with dual/mono modular front points, anti-balling plates, and interchangeable B2/B3 bindings for mountaineering boots.",
        "attributes": {"points": "12-Point Modular", "material": "4130 Chromoly Steel", "binding": "LeverLock Universal"},
        "tags": ["crampon", "ice-climbing", "mountaineering", "alpine", "hardware", "boots", "glacier"],
        "compatible_with": ["RET-005", "RET-006", "RET-001", "RET-014"],
        "color_hex": "#E87722",
        "surface_finish": "forged-7075-aluminum",
        "popularity_score": 0.92,
        "cold_start": False,
    },
    {
        "item_id": "RET-014",
        "sku": "CLM-HRD-002",
        "title": "Vortex Carbon-Shaft Technical Ice Axe (50cm)",
        "domain": "retail",
        "category": "Climbing Hardware",
        "subcategory": "Ice Axes & Tools",
        "brand": "CruxForge",
        "unit_price": 184.00,
        "gross_margin_pct": 57.5,
        "in_stock": True,
        "stock_qty": 31,
        "description": "T-rated hot-forged chromoly pick paired with a curved ergonomic aluminum/carbon composite shaft for steep couloirs and glacier self-arrest.",
        "attributes": {"rating": "CEN-T Technical", "length": "50cm", "weight": "430g"},
        "tags": ["ice-axe", "climbing", "mountaineering", "alpine", "crampon", "glacier", "hardware"],
        "compatible_with": ["RET-005", "RET-013", "RET-001"],
        "color_hex": "#A2ADAF",
        "surface_finish": "forged-7075-aluminum",
        "popularity_score": 0.86,
        "cold_start": False,
    },
    {
        "item_id": "RET-015",
        "sku": "CLM-HRD-003",
        "title": "Aegis Ultralight Dyneema Alpine Climbing Harness",
        "domain": "retail",
        "category": "Climbing Hardware",
        "subcategory": "Harnesses",
        "brand": "CruxForge",
        "unit_price": 112.00,
        "gross_margin_pct": 61.0,
        "in_stock": True,
        "stock_qty": 40,
        "description": "Drop-leg buckle construction allow donning while wearing crampons or skis without lifting feet off the snow.",
        "attributes": {"weight": "215g", "gear_loops": "4 + 2 Ice Clipper Slots", "material": "UHMWPE Dyneema"},
        "tags": ["harness", "climbing", "alpine", "crampon", "mountaineering", "ultralight"],
        "compatible_with": ["RET-001", "RET-002", "RET-013"],
        "color_hex": "#1B998B",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.84,
        "cold_start": False,
    },
    {
        "item_id": "RET-016",
        "sku": "CLM-HRD-004",
        "title": "TitanLock HMS Triple-Action Belay Carabiner 3-Pack",
        "domain": "retail",
        "category": "Climbing Hardware",
        "subcategory": "Carabiners & Draws",
        "brand": "CruxForge",
        "unit_price": 68.00,
        "gross_margin_pct": 64.5,
        "in_stock": True,
        "stock_qty": 85,
        "description": "Hot-forged 7075 I-beam pear carabiners with keylock nose and glove-friendly triple auto-locking sleeves.",
        "attributes": {"major_axis_strength": "24 kN", "gate_opening": "22mm", "certification": "UIAA 121"},
        "tags": ["carabiner", "climbing", "belay", "hardware", "alpine"],
        "compatible_with": ["RET-015", "RET-014"],
        "color_hex": "#7A8B99",
        "surface_finish": "forged-7075-aluminum",
        "popularity_score": 0.81,
        "cold_start": False,
    },
    {
        "item_id": "RET-017",
        "sku": "PCK-EXP-001",
        "title": "Serac 45L Cordura Waterproof Alpine Pack",
        "domain": "retail",
        "category": "Expedition Packs",
        "subcategory": "Alpine Climbing Packs",
        "brand": "ValhallaGear",
        "unit_price": 208.00,
        "gross_margin_pct": 57.0,
        "in_stock": True,
        "stock_qty": 24,
        "description": "Roll-top seam-taped 420D Cordura pack with dedicated external crampon pouch, dual ice tool toggles, and removable bivy frame sheet.",
        "attributes": {"volume": "45 Liters", "fabric": "420D Carbonate Cordura", "weight": "1,120g"},
        "tags": ["pack", "backpack", "alpine", "waterproof", "crampon", "mountaineering", "cordura"],
        "compatible_with": ["RET-001", "RET-005", "RET-013", "RET-014"],
        "color_hex": "#2E4057",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.88,
        "cold_start": False,
    },
    {
        "item_id": "RET-018",
        "sku": "PCK-EXP-002",
        "title": "DynaPeak 28L Ultra-Weave Summit Assault Pack",
        "domain": "retail",
        "category": "Expedition Packs",
        "subcategory": "Summit Packs",
        "brand": "TerraPace",
        "unit_price": 164.00,
        "gross_margin_pct": 60.0,
        "in_stock": True,
        "stock_qty": 33,
        "description": "Featherweight Challenge Ultra-200 woven UHMWPE summit pack for single-push alpine ascents and ski mountaineering.",
        "attributes": {"volume": "28 Liters", "fabric": "Ultra-200 UHMWPE", "weight": "540g"},
        "tags": ["pack", "summit", "ultralight", "alpine", "waterproof", "climbing"],
        "compatible_with": ["RET-004", "RET-006", "RET-015"],
        "color_hex": "#ECEFF1",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.77,
        "cold_start": True,
    },
    {
        "item_id": "RET-019",
        "sku": "BSL-TEC-001",
        "title": "MerinoCore 260g Zip-Neck Cold Weather Baselayer Top",
        "domain": "retail",
        "category": "Technical Baselayers",
        "subcategory": "Merino Tops",
        "brand": "NordicPeak",
        "unit_price": 105.00,
        "gross_margin_pct": 67.0,
        "in_stock": True,
        "stock_qty": 72,
        "description": "18.5-micron core-spun New Zealand merino wool baselayer offering natural odor resistance and moisture management next to skin.",
        "attributes": {"fabric": "87% Merino / 13% Nylon Core", "weight_class": "260 g/m2"},
        "tags": ["baselayer", "merino", "wool", "thermal", "alpine", "warm", "breathable"],
        "compatible_with": ["RET-001", "RET-009", "RET-011"],
        "color_hex": "#1F3A52",
        "surface_finish": "merino-grid-fleece",
        "popularity_score": 0.89,
        "cold_start": False,
    },
    {
        "item_id": "RET-020",
        "sku": "BSL-TEC-002",
        "title": "GlacierShield Gore-Tex Crocodile Mountaineering Gaiters",
        "domain": "retail",
        "category": "Technical Baselayers",
        "subcategory": "Alpine Gaiters & Accessories",
        "brand": "DolomiteAlps",
        "unit_price": 84.00,
        "gross_margin_pct": 63.0,
        "in_stock": True,
        "stock_qty": 51,
        "description": "1000D Cordura inner ankle kick-patch protects against sharp crampon points while 3L Gore-Tex upper seals out deep powder and wet slush.",
        "attributes": {"lower_fabric": "1000D Ballistic Cordura", "upper_fabric": "Gore-Tex 3L", "instep_strap": "BioThane"},
        "tags": ["gaiters", "crampon", "waterproof", "gore-tex", "boots", "alpine", "mountaineering"],
        "compatible_with": ["RET-005", "RET-006", "RET-013", "RET-001"],
        "color_hex": "#1C1E21",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.90,
        "cold_start": False,
    },
    {
        "item_id": "RET-021",
        "sku": "ARC-SHL-005",
        "title": "Couloir Pro 3L Gore-Tex Bib Hardshell Pants",
        "domain": "retail",
        "category": "Alpine Shells",
        "subcategory": "Alpine Shell Pants",
        "brand": "NordicPeak",
        "unit_price": 218.00,
        "gross_margin_pct": 56.0,
        "in_stock": True,
        "stock_qty": 26,
        "description": "Full side-zip waterproof alpine shell pants under $220 with Kevlar crampon instep guards and internal boot snow gaiters.",
        "attributes": {"hydrostatic_head": "28,000mm", "crampon_guard": "Keprotec Kevlar", "weight": "495g"},
        "tags": ["waterproof", "alpine", "shell", "pants", "gore-tex", "crampon", "boots", "mountaineering"],
        "compatible_with": ["RET-001", "RET-005", "RET-006", "RET-013"],
        "color_hex": "#263238",
        "surface_finish": "gore-tex-3l-ripstop",
        "popularity_score": 0.87,
        "cold_start": False,
    },
    {
        "item_id": "RET-022",
        "sku": "BSL-TEC-003",
        "title": "IceDex Gore-Tex Grip Waterproof Technical Ice Climbing Gloves",
        "domain": "retail",
        "category": "Technical Baselayers",
        "subcategory": "Alpine Gloves",
        "brand": "CruxForge",
        "unit_price": 129.00,
        "gross_margin_pct": 60.5,
        "in_stock": True,
        "stock_qty": 39,
        "description": "Pre-curved Pittards goat leather palm bonded directly to a Gore-Tex Grip membrane for zero-slip ice axe handling in wet conditions.",
        "attributes": {"palm": "Pittards Oiltac Leather", "insulation": "PrimaLoft Gold 100g", "membrane": "Gore-Tex Grip"},
        "tags": ["gloves", "waterproof", "ice-climbing", "alpine", "gore-tex", "crampon"],
        "compatible_with": ["RET-001", "RET-014", "RET-013"],
        "color_hex": "#2C3539",
        "surface_finish": "cordura-ballistic-weave",
        "popularity_score": 0.85,
        "cold_start": False,
    },
    # ==================== INDUSTRIAL SPARE PARTS & MRO COMPONENTS (22 items) ====================
    {
        "item_id": "IND-001",
        "sku": "HYD-SEL-001",
        "title": "HydroFlex Viton FKM High-Pressure Hydraulic Piston Seal Kit (50mm Bore)",
        "domain": "industrial",
        "category": "Hydraulic Seals",
        "subcategory": "Piston & Rod Seal Kits",
        "brand": "RexrothFluidics",
        "unit_price": 118.00,
        "gross_margin_pct": 64.0,
        "in_stock": True,
        "stock_qty": 95,
        "description": "Heavy-duty high-pressure hydraulic seal replacement kit rated to 400 bar (5,800 PSI) and 200C continuous operating temperature with PTFE anti-extrusion backup rings.",
        "attributes": {"max_pressure": "400 bar / 5800 PSI", "bore_diameter": "50mm", "elastomer": "Viton FKM 90 Shore A"},
        "tags": ["hydraulic", "seal", "high-pressure", "replacement", "viton", "piston", "cylinder", "o-ring"],
        "compatible_with": ["IND-005", "IND-009", "IND-013", "IND-017"],
        "color_hex": "#2B2D2F",
        "surface_finish": "viton-fluoroelastomer",
        "popularity_score": 0.95,
        "cold_start": False,
    },
    {
        "item_id": "IND-002",
        "sku": "HYD-SEL-002",
        "title": "DuraGlide PTFE Bronze-Filled Rod Wiper & Chevron Packing Set",
        "domain": "industrial",
        "category": "Hydraulic Seals",
        "subcategory": "Chevron Vee Packings",
        "brand": "ParkerMotion",
        "unit_price": 142.00,
        "gross_margin_pct": 61.0,
        "in_stock": True,
        "stock_qty": 64,
        "description": "Low-friction 40% bronze-filled PTFE rod seal assembly under $150 designed for high-cycle hydraulic press cylinders and excavator boom actuators.",
        "attributes": {"max_pressure": "450 bar / 6500 PSI", "rod_diameter": "36mm", "material": "Bronze-PTFE + FKM"},
        "tags": ["hydraulic", "seal", "high-pressure", "replacement", "ptfe", "rod-seal", "wiper", "actuator"],
        "compatible_with": ["IND-005", "IND-006", "IND-013"],
        "color_hex": "#6E5A3E",
        "surface_finish": "ptfe-teflon-composite",
        "popularity_score": 0.91,
        "cold_start": False,
    },
    {
        "item_id": "IND-003",
        "sku": "HYD-SEL-003",
        "title": "CryoSeal HNBR Sub-Zero High-Pressure Rotary Shaft Seal",
        "domain": "industrial",
        "category": "Hydraulic Seals",
        "subcategory": "Rotary Shaft Seals",
        "brand": "RexrothFluidics",
        "unit_price": 89.00,
        "gross_margin_pct": 68.0,
        "in_stock": True,
        "stock_qty": 120,
        "description": "Hydrogenated Nitrile (HNBR) stainless-cased double-lip rotary shaft seal engineered for high-pressure hydraulic pumps and motors.",
        "attributes": {"max_pressure": "250 bar", "temp_range": "-40C to +150C", "material": "HNBR + 316L Case"},
        "tags": ["hydraulic", "seal", "high-pressure", "replacement", "rotary", "shaft", "pump"],
        "compatible_with": ["IND-006", "IND-014", "IND-021"],
        "color_hex": "#1E4D2B",
        "surface_finish": "viton-fluoroelastomer",
        "popularity_score": 0.87,
        "cold_start": False,
    },
    {
        "item_id": "IND-004",
        "sku": "HYD-SEL-004",
        "title": "Kalrez Perfluoroelastomer FFKM Ultra-Chemical Manifold O-Ring Kit",
        "domain": "industrial",
        "category": "Hydraulic Seals",
        "subcategory": "Chemical Flange Seals",
        "brand": "ChemShield",
        "unit_price": 310.00,
        "gross_margin_pct": 53.0,
        "in_stock": True,
        "stock_qty": 18,
        "description": "Extreme sour-gas and aggressive ester fluid resistant FFKM seal assortment for petrochemical manifold blocks up to 315C.",
        "attributes": {"max_pressure": "500 bar", "temp_range": "-15C to +315C", "material": "FFKM Perfluoroelastomer"},
        "tags": ["hydraulic", "seal", "high-pressure", "ffkm", "chemical", "manifold", "o-ring"],
        "compatible_with": ["IND-005", "IND-007"],
        "color_hex": "#1A1A1A",
        "surface_finish": "viton-fluoroelastomer",
        "popularity_score": 0.76,
        "cold_start": True,
    },
    {
        "item_id": "IND-005",
        "sku": "HYD-VLV-001",
        "title": "ProFlow CETOP-03 316L Proportional Directional Control Valve",
        "domain": "industrial",
        "category": "High-Pressure Valves",
        "subcategory": "Proportional Valves",
        "brand": "RexrothFluidics",
        "unit_price": 345.00,
        "gross_margin_pct": 55.0,
        "in_stock": True,
        "stock_qty": 22,
        "description": "CNC-machined 316 stainless directional spool valve with integrated LVDT position feedback and Viton manifold interface seals.",
        "attributes": {"max_pressure": "350 bar", "flow_rate": "65 L/min", "mounting": "ISO 4401 CETOP-03"},
        "tags": ["valve", "hydraulic", "high-pressure", "proportional", "directional", "manifold", "stainless"],
        "compatible_with": ["IND-001", "IND-002", "IND-013", "IND-017"],
        "color_hex": "#8A959E",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.92,
        "cold_start": False,
    },
    {
        "item_id": "IND-006",
        "sku": "HYD-VLV-002",
        "title": "SafeGuard Direct-Acting High-Pressure Relief Cartridge Valve",
        "domain": "industrial",
        "category": "High-Pressure Valves",
        "subcategory": "Pressure Relief Cartridges",
        "brand": "ParkerMotion",
        "unit_price": 136.00,
        "gross_margin_pct": 62.5,
        "in_stock": True,
        "stock_qty": 54,
        "description": "Hardened steel poppet safety relief cartridge valve under $150 with Viton high-pressure seal rings for hydraulic power pack overpressure protection.",
        "attributes": {"max_pressure": "420 bar / 6000 PSI", "cavity": "SAE-10-2", "seal_kit": "Viton FKM Included"},
        "tags": ["valve", "hydraulic", "high-pressure", "relief", "cartridge", "seal", "replacement"],
        "compatible_with": ["IND-001", "IND-002", "IND-013"],
        "color_hex": "#7D8790",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.89,
        "cold_start": False,
    },
    {
        "item_id": "IND-007",
        "sku": "HYD-VLV-003",
        "title": "ZeroLeak Pilot-Operated Dual Counterbalance Holding Valve",
        "domain": "industrial",
        "category": "High-Pressure Valves",
        "subcategory": "Load Holding Valves",
        "brand": "ParkerMotion",
        "unit_price": 195.00,
        "gross_margin_pct": 58.0,
        "in_stock": True,
        "stock_qty": 31,
        "description": "Prevents hydraulic cylinder drift and runaway loads on boom actuators with zero-leakage lapped seat geometry.",
        "attributes": {"max_pressure": "380 bar", "pilot_ratio": "3:1", "body": "Zinc-Nickel Plated Steel"},
        "tags": ["valve", "hydraulic", "high-pressure", "counterbalance", "cylinder", "actuator"],
        "compatible_with": ["IND-001", "IND-013", "IND-015"],
        "color_hex": "#9BA4AB",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.84,
        "cold_start": False,
    },
    {
        "item_id": "IND-008",
        "sku": "HYD-VLV-004",
        "title": "MicroPulse High-Speed 24VDC Solenoid Selector Valve",
        "domain": "industrial",
        "category": "High-Pressure Valves",
        "subcategory": "Solenoid Valves",
        "brand": "RexrothFluidics",
        "unit_price": 128.00,
        "gross_margin_pct": 63.0,
        "in_stock": True,
        "stock_qty": 47,
        "description": "IP69K wet-pin armature solenoid valve under $150 with manual override and Viton port seals for automated hydraulic clamping circuits.",
        "attributes": {"max_pressure": "315 bar", "voltage": "24V DC", "response_time": "18ms"},
        "tags": ["valve", "hydraulic", "high-pressure", "solenoid", "24vdc", "seal", "replacement"],
        "compatible_with": ["IND-001", "IND-017", "IND-018"],
        "color_hex": "#5C6670",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.86,
        "cold_start": False,
    },
    {
        "item_id": "IND-009",
        "sku": "BRG-PRC-001",
        "title": "AeroSpin 7008C Hybrid Silicon Nitride Ceramic Spindle Bearing",
        "domain": "industrial",
        "category": "Precision Bearings",
        "subcategory": "Angular Contact Bearings",
        "brand": "SKF-Precision",
        "unit_price": 144.00,
        "gross_margin_pct": 60.0,
        "in_stock": True,
        "stock_qty": 41,
        "description": "ABEC-9 (ISO P2) 15-degree angular contact CNC spindle bearing with Si3N4 ceramic balls and PEEK phenolic cage for 32,000 RPM machining centers.",
        "attributes": {"precision": "ABEC-9 / ISO P2", "bore_od_width": "40x68x15mm", "max_rpm": "32,000 RPM"},
        "tags": ["bearing", "ceramic", "spindle", "cnc", "precision", "angular-contact", "replacement"],
        "compatible_with": ["IND-014", "IND-019", "IND-021"],
        "color_hex": "#6E7B85",
        "surface_finish": "ceramic-hybrid-nitride",
        "popularity_score": 0.93,
        "cold_start": False,
    },
    {
        "item_id": "IND-010",
        "sku": "BRG-PRC-002",
        "title": "TitanLoad 22210 Spherical Roller Heavy Axial Bearing",
        "domain": "industrial",
        "category": "Precision Bearings",
        "subcategory": "Spherical Roller Bearings",
        "brand": "SKF-Precision",
        "unit_price": 115.00,
        "gross_margin_pct": 59.5,
        "in_stock": True,
        "stock_qty": 58,
        "description": "Self-aligning double-row spherical roller bearing with machined brass cage and Viton triple-lip labyrinth seals for high-vibration crushers and conveyors.",
        "attributes": {"dynamic_load": "129 kN", "bore_od_width": "50x90x23mm", "sealing": "Viton Contact Seal"},
        "tags": ["bearing", "roller", "spherical", "heavy-duty", "seal", "replacement", "conveyor"],
        "compatible_with": ["IND-003", "IND-021", "IND-022"],
        "color_hex": "#808B96",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.88,
        "cold_start": False,
    },
    {
        "item_id": "IND-011",
        "sku": "BRG-PRC-003",
        "title": "LinearGlide HGH25CA Preloaded Recirculating Ball Rail Carriage",
        "domain": "industrial",
        "category": "Precision Bearings",
        "subcategory": "Linear Motion Guides",
        "brand": "THK-Linear",
        "unit_price": 94.00,
        "gross_margin_pct": 65.0,
        "in_stock": True,
        "stock_qty": 76,
        "description": "Heavy-preload Z1 linear guideway block with double end scraper seals to keep metal swarf out of CNC gantry axes.",
        "attributes": {"rail_size": "25mm", "preload_class": "Z1 Medium-Heavy", "dynamic_rating": "32.7 kN"},
        "tags": ["bearing", "linear", "cnc", "carriage", "rail", "seal", "replacement"],
        "compatible_with": ["IND-009", "IND-016", "IND-020"],
        "color_hex": "#4A5568",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.85,
        "cold_start": False,
    },
    {
        "item_id": "IND-012",
        "sku": "BRG-PRC-004",
        "title": "CryoClean Full-Complement ZrO2 Zirconia Ceramic Wet-Process Bearing",
        "domain": "industrial",
        "category": "Precision Bearings",
        "subcategory": "Corrosion-Proof Bearings",
        "brand": "SKF-Precision",
        "unit_price": 178.00,
        "gross_margin_pct": 62.0,
        "in_stock": True,
        "stock_qty": 19,
        "description": "Non-magnetic, lubricant-free zirconia ceramic bearing for submerged acid pumps and semiconductor wafer wash stations.",
        "attributes": {"races_and_balls": "100% ZrO2 Ceramic", "max_temp": "400C", "cage": "PTFE"},
        "tags": ["bearing", "ceramic", "chemical", "ptfe", "pump", "corrosion-proof"],
        "compatible_with": ["IND-004", "IND-003"],
        "color_hex": "#EAECEE",
        "surface_finish": "ceramic-hybrid-nitride",
        "popularity_score": 0.75,
        "cold_start": True,
    },
    {
        "item_id": "IND-013",
        "sku": "ACT-IND-001",
        "title": "ForceMax Tie-Rod Hydraulic Cylinder Actuator (50mm Bore x 200mm Stroke)",
        "domain": "industrial",
        "category": "Industrial Actuators",
        "subcategory": "Hydraulic Cylinders",
        "brand": "ParkerMotion",
        "unit_price": 285.00,
        "gross_margin_pct": 54.0,
        "in_stock": True,
        "stock_qty": 21,
        "description": "Induction-hardened chrome piston rod hydraulic cylinder factory-fitted with Viton high-pressure seal glands and adjustable end-stroke cushioning.",
        "attributes": {"bore_stroke": "50mm x 200mm", "working_pressure": "250 bar", "seal_kit_ref": "IND-001"},
        "tags": ["actuator", "hydraulic", "cylinder", "high-pressure", "piston", "seal"],
        "compatible_with": ["IND-001", "IND-002", "IND-005", "IND-006", "IND-017"],
        "color_hex": "#C0392B",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.90,
        "cold_start": False,
    },
    {
        "item_id": "IND-014",
        "sku": "ACT-IND-002",
        "title": "TorqueMaster High-Pressure Gerotor Hydraulic Orbital Motor",
        "domain": "industrial",
        "category": "Industrial Actuators",
        "subcategory": "Hydraulic Motors",
        "brand": "RexrothFluidics",
        "unit_price": 248.00,
        "gross_margin_pct": 56.5,
        "in_stock": True,
        "stock_qty": 17,
        "description": "High starting torque hydraulic orbital motor with needle-roller output bearing and HNBR high-pressure shaft seal allowing case-drain-free operation.",
        "attributes": {"displacement": "160 cc/rev", "max_torque": "410 Nm", "shaft_seal_ref": "IND-003"},
        "tags": ["actuator", "hydraulic", "motor", "high-pressure", "rotary", "seal"],
        "compatible_with": ["IND-003", "IND-005", "IND-009"],
        "color_hex": "#283747",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.84,
        "cold_start": False,
    },
    {
        "item_id": "IND-015",
        "sku": "ACT-IND-003",
        "title": "PneumoGrip ISO-15552 Hard-Anodized Pneumatic Cylinder (63mm Bore)",
        "domain": "industrial",
        "category": "Industrial Actuators",
        "subcategory": "Pneumatic Cylinders",
        "brand": "FestoDynamics",
        "unit_price": 124.00,
        "gross_margin_pct": 63.0,
        "in_stock": True,
        "stock_qty": 46,
        "description": "Hard-anodized aluminum profile barrel with magnetic piston ring and low-friction polyurethane lip seals for packaging robotics.",
        "attributes": {"bore_stroke": "63mm x 150mm", "max_pressure": "12 bar", "ports": "G3/8 BSP"},
        "tags": ["actuator", "pneumatic", "cylinder", "seal", "replacement", "automation"],
        "compatible_with": ["IND-018", "IND-020"],
        "color_hex": "#5D6D7E",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.82,
        "cold_start": False,
    },
    {
        "item_id": "IND-016",
        "sku": "ACT-IND-004",
        "title": "ServoDrive Precision Ground Ball Screw Actuator Assembly (SFU2005)",
        "domain": "industrial",
        "category": "Industrial Actuators",
        "subcategory": "Electromechanical Actuators",
        "brand": "THK-Linear",
        "unit_price": 165.00,
        "gross_margin_pct": 59.0,
        "in_stock": False,
        "stock_qty": 0,
        "description": "C5 ground precision ball screw with zero-backlash preloaded nut and BK15 angular contact support bearing blocks.",
        "attributes": {"lead_accuracy": "C5 (+/- 0.018mm)", "diameter_lead": "20mm x 5mm", "length": "600mm"},
        "tags": ["actuator", "cnc", "ball-screw", "linear", "bearing", "precision"],
        "compatible_with": ["IND-009", "IND-011"],
        "color_hex": "#A6ACAF",
        "surface_finish": "cnc-316-stainless",
        "popularity_score": 0.79,
        "cold_start": False,
    },
    {
        "item_id": "IND-017",
        "sku": "SNS-TEL-001",
        "title": "PressSense 0-600 Bar Thin-Film Stainless Hydraulic Pressure Transducer",
        "domain": "industrial",
        "category": "Sensor Telemetry",
        "subcategory": "Pressure Transmitters",
        "brand": "IFM-Sensing",
        "unit_price": 134.00,
        "gross_margin_pct": 66.0,
        "in_stock": True,
        "stock_qty": 68,
        "description": "Sputtered thin-film 316L stainless steel pressure transducer under $150 with integrated Viton FKM G1/4 port seal and 4-20mA / IO-Link telemetry output.",
        "attributes": {"measuring_range": "0 to 600 bar", "output": "4-20mA + IO-Link 1.1", "accuracy": "+/- 0.25% FS"},
        "tags": ["sensor", "hydraulic", "high-pressure", "transducer", "seal", "telemetry", "replacement"],
        "compatible_with": ["IND-001", "IND-005", "IND-006", "IND-013"],
        "color_hex": "#D35400",
        "surface_finish": "ip68-potted-polymer",
        "popularity_score": 0.91,
        "cold_start": False,
    },
    {
        "item_id": "IND-018",
        "sku": "SNS-TEL-002",
        "title": "VibeGuard Triaxial MEMS Vibration & Bearing Fault Accelerometer",
        "domain": "industrial",
        "category": "Sensor Telemetry",
        "subcategory": "Condition Monitoring",
        "brand": "IFM-Sensing",
        "unit_price": 148.00,
        "gross_margin_pct": 67.5,
        "in_stock": True,
        "stock_qty": 39,
        "description": "Real-time FFT envelope bearing wear and hydraulic cavitation detector with M12 stainless base and Modbus RTU / IO-Link diagnostics.",
        "attributes": {"bandwidth": "2 Hz to 10 kHz", "protection": "IP68 / IP69K", "temp_sensor": "Integrated PT100"},
        "tags": ["sensor", "vibration", "bearing", "hydraulic", "telemetry", "predictive-maintenance"],
        "compatible_with": ["IND-009", "IND-010", "IND-014"],
        "color_hex": "#E67E22",
        "surface_finish": "ip68-potted-polymer",
        "popularity_score": 0.87,
        "cold_start": True,
    },
    {
        "item_id": "IND-019",
        "sku": "SNS-TEL-003",
        "title": "ThermoFlow Inline Hydraulic Oil Flow & Temperature Meter",
        "domain": "industrial",
        "category": "Sensor Telemetry",
        "subcategory": "Flow Sensors",
        "brand": "IFM-Sensing",
        "unit_price": 210.00,
        "gross_margin_pct": 61.0,
        "in_stock": True,
        "stock_qty": 25,
        "description": "High-pressure turbine bypassed calorimetric flow sensor for detecting internal hydraulic cylinder seal bypass leakage in real time.",
        "attributes": {"max_pressure": "350 bar", "flow_range": "1 to 100 L/min", "seals": "Viton FKM"},
        "tags": ["sensor", "hydraulic", "flow", "high-pressure", "seal", "leak-detection"],
        "compatible_with": ["IND-001", "IND-005", "IND-013"],
        "color_hex": "#CA6F1E",
        "surface_finish": "ip68-potted-polymer",
        "popularity_score": 0.83,
        "cold_start": False,
    },
    {
        "item_id": "IND-020",
        "sku": "SNS-TEL-004",
        "title": "MagTrack Non-Contact Magnetostrictive Linear Stroke Position Sensor",
        "domain": "industrial",
        "category": "Sensor Telemetry",
        "subcategory": "Position Encoders",
        "brand": "FestoDynamics",
        "unit_price": 182.00,
        "gross_margin_pct": 63.0,
        "in_stock": True,
        "stock_qty": 30,
        "description": "5-micron resolution absolute linear encoder for closed-loop hydraulic press cylinders and CNC gantry stages.",
        "attributes": {"resolution": "5 um", "stroke_length": "250mm", "interface": "SSI / 0-10V"},
        "tags": ["sensor", "linear", "actuator", "cylinder", "hydraulic", "cnc"],
        "compatible_with": ["IND-013", "IND-015", "IND-011"],
        "color_hex": "#34495E",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.80,
        "cold_start": False,
    },
    {
        "item_id": "IND-021",
        "sku": "PWR-TRN-001",
        "title": "FlexJaw Zero-Backlash Polyurethane Spider Servo Coupling",
        "domain": "industrial",
        "category": "Power Transmission",
        "subcategory": "Shaft Couplings",
        "brand": "RexrothFluidics",
        "unit_price": 76.00,
        "gross_margin_pct": 68.5,
        "in_stock": True,
        "stock_qty": 88,
        "description": "7075-T6 CNC clamping hubs with 98 Shore A red polyurethane vibration-damping spider for hydraulic pump drives and CNC ball screws.",
        "attributes": {"bore_sizes": "19mm x 24mm", "nominal_torque": "160 Nm", "insert": "98 ShA Polyurethane"},
        "tags": ["coupling", "shaft", "pump", "cnc", "spindle", "replacement", "bearing"],
        "compatible_with": ["IND-003", "IND-009", "IND-014"],
        "color_hex": "#B03A2E",
        "surface_finish": "forged-7075-aluminum",
        "popularity_score": 0.86,
        "cold_start": False,
    },
    {
        "item_id": "IND-022",
        "sku": "PWR-TRN-002",
        "title": "HydroClean 10-Micron High-Pressure Microglass Spin-On Filter Element",
        "domain": "industrial",
        "category": "Power Transmission",
        "subcategory": "Hydraulic Filtration",
        "brand": "ParkerMotion",
        "unit_price": 64.00,
        "gross_margin_pct": 69.0,
        "in_stock": True,
        "stock_qty": 140,
        "description": "Beta-1000 microglass high-pressure inline filter element with Viton bowl seal to protect proportional valves and piston seals from particulate scoring.",
        "attributes": {"filtration_rating": "10 um Absolute (Beta 1000)", "collapse_pressure": "210 bar", "seal": "Viton FKM"},
        "tags": ["hydraulic", "filter", "high-pressure", "seal", "valve", "replacement", "maintenance"],
        "compatible_with": ["IND-001", "IND-005", "IND-006", "IND-013"],
        "color_hex": "#1B4F72",
        "surface_finish": "hard-anodized-alloy",
        "popularity_score": 0.92,
        "cold_start": False,
    },
]


BENCHMARK_PERSONAS: dict[str, UserPersona] = {
    "alpine_mountaineer": UserPersona(
        user_id="alpine_mountaineer",
        name="Elena Vance — Technical Alpinist & Guide",
        role="Outdoor Retail Pro Buyer",
        preferred_domain="retail",
        budget_ceiling=220.0,
        price_sensitivity=0.55,
        margin_affinity=0.65,
        recent_item_ids=["RET-005", "RET-013", "RET-019"],
        intent_keywords=["waterproof", "alpine", "shell", "gore-tex", "crampon", "boots", "mountaineering"],
        description="Preparing for a winter mixed route; recently purchased rigid crampon mountaineering boots and 12-point crampons, seeking a waterproof alpine shell under $220 and compatible layering gear.",
    ),
    "plant_reliability_lead": UserPersona(
        user_id="plant_reliability_lead",
        name="Marcus Thorne — Hydraulic Plant Reliability Engineer",
        role="Industrial MRO Procurement",
        preferred_domain="industrial",
        budget_ceiling=150.0,
        price_sensitivity=0.60,
        margin_affinity=0.70,
        recent_item_ids=["IND-013", "IND-005", "IND-022"],
        intent_keywords=["high-pressure", "hydraulic", "seal", "replacement", "viton", "valve", "transducer"],
        description="Managing preventive maintenance on 400-bar hydraulic press cylinders; searching for high-pressure hydraulic seal replacements and relief/sensing components under $150.",
    ),
    "cnc_automation_architect": UserPersona(
        user_id="cnc_automation_architect",
        name="Dr. Kenji Sato — Precision CNC & Robotics Lead",
        role="Factory Automation Buyer",
        preferred_domain="industrial",
        budget_ceiling=250.0,
        price_sensitivity=0.35,
        margin_affinity=0.75,
        recent_item_ids=["IND-009", "IND-011", "IND-018"],
        intent_keywords=["ceramic", "bearing", "spindle", "cnc", "linear", "vibration", "sensor"],
        description="Upgrading 5-axis CNC machining centers with ABEC-9 ceramic hybrid bearings, preloaded linear rail carriages, and MEMS vibration condition monitoring.",
    ),
    "ultralight_trail_buyer": UserPersona(
        user_id="ultralight_trail_buyer",
        name="Maya Lin — Fastpacking & Ultralight Gear Specialist",
        role="Specialty Retail Merchandiser",
        preferred_domain="retail",
        budget_ceiling=200.0,
        price_sensitivity=0.50,
        margin_affinity=0.80,
        recent_item_ids=["RET-004", "RET-011", "RET-018"],
        intent_keywords=["ultralight", "packable", "shell", "breathable", "midlayer", "merino", "summit"],
        description="Curating high-margin ultralight alpine shells, grid fleeces, and UHMWPE summit packs under $200 for fast-and-light mountain endurance kits.",
    ),
}


def build_catalog() -> list[CatalogItem]:
    """Instantiate the full 44-item multimodal catalog with computed 64-D feature vectors."""
    items: list[CatalogItem] = []
    for spec in _RAW_CATALOG_SPECS:
        vis_vec = extract_visual_descriptor(
            sku=spec["sku"],
            domain=spec["domain"],
            category=spec["category"],
            color_hex=spec["color_hex"],
            surface_finish=spec["surface_finish"],
        )
        sem_vec = extract_semantic_descriptor(
            sku=spec["sku"],
            domain=spec["domain"],
            category=spec["category"],
            tags=spec["tags"],
            unit_price=spec["unit_price"],
            gross_margin_pct=spec["gross_margin_pct"],
        )
        multimodal = np.concatenate([vis_vec, sem_vec])
        norm = np.linalg.norm(multimodal)
        if norm > 1e-8:
            multimodal = multimodal / norm

        items.append(
            CatalogItem(
                item_id=spec["item_id"],
                sku=spec["sku"],
                title=spec["title"],
                domain=spec["domain"],
                category=spec["category"],
                subcategory=spec["subcategory"],
                brand=spec["brand"],
                unit_price=spec["unit_price"],
                gross_margin_pct=spec["gross_margin_pct"],
                in_stock=spec["in_stock"],
                stock_qty=spec["stock_qty"],
                description=spec["description"],
                attributes=spec["attributes"],
                tags=spec["tags"],
                compatible_with=spec["compatible_with"],
                color_hex=spec["color_hex"],
                surface_finish=spec["surface_finish"],
                popularity_score=spec["popularity_score"],
                cold_start=spec["cold_start"],
                visual_embedding=vis_vec.tolist(),
                semantic_embedding=sem_vec.tolist(),
                multimodal_features=multimodal.tolist(),
            )
        )
    return items


class MultimodalCatalogStore:
    """In-memory indexed store over the 44-item multimodal catalog and personas."""

    def __init__(self) -> None:
        self._items: list[CatalogItem] = build_catalog()
        self._by_id: dict[str, CatalogItem] = {item.item_id: item for item in self._items}
        self._personas: dict[str, UserPersona] = dict(BENCHMARK_PERSONAS)

    @property
    def items(self) -> list[CatalogItem]:
        return self._items

    @property
    def personas(self) -> dict[str, UserPersona]:
        return self._personas

    def get_item(self, item_id: str) -> CatalogItem | None:
        return self._by_id.get(item_id)

    def get_persona(self, persona_id: str) -> UserPersona:
        return self._personas.get(persona_id, self._personas["alpine_mountaineer"])

    def list_categories(self, domain: str | None = None) -> list[str]:
        cats = {
            item.category
            for item in self._items
            if domain is None or item.domain == domain
        }
        return sorted(cats)

    def summary_stats(self) -> dict[str, Any]:
        retail_count = sum(1 for i in self._items if i.domain == "retail")
        ind_count = sum(1 for i in self._items if i.domain == "industrial")
        in_stock_count = sum(1 for i in self._items if i.in_stock)
        cold_start_count = sum(1 for i in self._items if i.cold_start)
        return {
            "total_items": len(self._items),
            "retail_items": retail_count,
            "industrial_items": ind_count,
            "in_stock_items": in_stock_count,
            "cold_start_items": cold_start_count,
            "categories": self.list_categories(),
            "visual_dim": 32,
            "semantic_dim": 32,
            "multimodal_dim": 64,
        }
