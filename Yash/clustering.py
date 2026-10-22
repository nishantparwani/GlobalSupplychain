"""
CLUSTERING (Owner: Yash)
========================
K-Means clustering (k-means++ init, silhouette score) of commodities by price
volatility and supply strain, with noisy-OR supply exposure.
"""

import math
import random
from typing import Any, Dict, List

from Abhik import ai_models as ai
from Nishant.core import CHOKE_POINTS, STATE
from Nishant.routing import effective_risk


# =============================================================================
# UNIT 5: K-MEANS CLUSTERING (pure Python, 2D, k-means++ init)
# =============================================================================
# code matches data_sources.MARKET_SERIES; vol/stress are the modelled fallback used before any
# market data has been ingested. exposure = share of seaborne supply routed through each choke point.
COMMODITIES = [
    {"name": "Brent crude",        "code": "BRENT", "vol": 72, "stress": 58, "exposure": {"HORMUZ": 0.9, "BAB_EL_MANDEB": 0.4, "SUEZ": 0.3, "MALACCA": 0.3}},
    {"name": "LNG / EU gas (TTF)", "code": "TTF",   "vol": 68, "stress": 55, "exposure": {"HORMUZ": 0.8, "BAB_EL_MANDEB": 0.4, "SUEZ": 0.3, "PANAMA": 0.2}},
    {"name": "US natural gas",     "code": "HH",    "vol": 55, "stress": 30, "exposure": {"PANAMA": 0.4}},
    {"name": "Tanker freight",     "code": "TANKER_FRT", "vol": 64, "stress": 60, "exposure": {"HORMUZ": 0.8, "BAB_EL_MANDEB": 0.5, "SUEZ": 0.4, "MALACCA": 0.3}},
    {"name": "Dry-bulk freight",   "code": "BULK_FRT",   "vol": 58, "stress": 45, "exposure": {"PANAMA": 0.4, "MALACCA": 0.4, "SUEZ": 0.2}},
    {"name": "Wheat",              "code": "WHEAT", "vol": 46, "stress": 38, "exposure": {"BAB_EL_MANDEB": 0.3, "SUEZ": 0.4}},
    {"name": "Soybeans",           "code": "SOY",   "vol": 33, "stress": 35, "exposure": {"PANAMA": 0.6}},
    {"name": "Corn",               "code": "CORN",  "vol": 35, "stress": 33, "exposure": {"PANAMA": 0.6}},
    {"name": "Copper",             "code": "CU",    "vol": 34, "stress": 30, "exposure": {"PANAMA": 0.3, "MALACCA": 0.2}},
    {"name": "Aluminium",          "code": "AL",    "vol": 31, "stress": 28, "exposure": {"SUEZ": 0.2, "HORMUZ": 0.2}},
    {"name": "Iron ore",           "code": "FE",    "vol": 30, "stress": 26, "exposure": {"MALACCA": 0.3}},
]


def commodity_vectors(live: Dict[str, float]) -> List[Dict[str, Any]]:
    """Feature vector [volatility, supply-chain stress], each 0..100.

    With market data (after an ingestion):
      volatility = 30-day realised volatility, annualised, scaled so 80% -> 100
      stress     = 100 * (0.6 * noisy-OR P(supply hit) + 0.4 * price shock)
                   noisy-OR over exposed choke points with P(c) = effective risk / 10,
                   price shock = clamp(3-month price z-score / 3, 0, 1)
    Without market data: modelled snapshot coupled to live choke-point severity."""
    p_choke = {c: effective_risk(c, live) / 10.0 for c in CHOKE_POINTS}
    markets = STATE.get("markets") or {}
    out = []
    for c in COMMODITIES:
        m = markets.get(c["code"], {}).get("stats")
        supply_p = ai.noisy_or(c["exposure"], p_choke)
        if m:
            vol = min(100.0, m["vol_30d_ann"] * 125.0)
            shock = ai.clamp(m["z_score_3m"] / 3.0, 0.0, 1.0)
            stress = 100.0 * (0.6 * supply_p + 0.4 * shock)
            source = "MARKET"
        else:
            vol, stress = float(c["vol"]), float(c["stress"])
            for choke, w in c["exposure"].items():
                excess = max(0.0, live.get(choke, 0.0) - 3.0)
                vol += w * excess * 2.2
                stress += w * excess * 3.0
            source = "MODELLED"
        out.append({"name": c["name"], "code": c["code"],
                    "volatility": round(min(100.0, vol), 1),
                    "stress": round(min(100.0, stress), 1),
                    "supply_hit_probability": round(supply_p, 3),
                    "market": {k: m[k] for k in ("last", "ret_1m", "vol_30d_ann", "z_score_3m")} if m else None,
                    "feature_source": source,
                    "exposure": c["exposure"]})
    return out


def _dist(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def kmeans(points: List[List[float]], k: int, eps: float = 1e-3, max_iter: int = 20, seed: int = 42) -> Dict[str, Any]:
    n = len(points)
    k = max(1, min(k, n))
    rng = random.Random(seed)

    # k-means++ initialisation (deterministic seed for reproducible grading)
    centroids = [list(points[rng.randrange(n)])]
    while len(centroids) < k:
        d2 = [min(_dist(p, c) ** 2 for c in centroids) for p in points]
        total = sum(d2)
        if total == 0:
            centroids.append(list(points[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(d2):
            acc += w
            if acc >= r:
                centroids.append(list(points[i]))
                break

    history = [[[round(x, 3) for x in c] for c in centroids]]
    labels = [0] * n
    iterations = 0
    converged = False
    final_shift = None
    for it in range(1, max_iter + 1):
        iterations = it
        # assignment step (Euclidean)
        labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in points]
        # update step
        new_centroids = []
        for j in range(k):
            members = [points[i] for i in range(n) if labels[i] == j]
            if members:
                new_centroids.append([sum(m[0] for m in members) / len(members),
                                      sum(m[1] for m in members) / len(members)])
            else:  # empty cluster: re-seed at the point farthest from its centroid
                far = max(range(n), key=lambda i: _dist(points[i], centroids[labels[i]]))
                new_centroids.append(list(points[far]))
        shift = max(_dist(centroids[j], new_centroids[j]) for j in range(k))
        centroids = new_centroids
        history.append([[round(x, 3) for x in c] for c in centroids])
        final_shift = shift
        if shift < eps:
            converged = True
            break

    labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in points]
    inertia = sum(_dist(points[i], centroids[labels[i]]) ** 2 for i in range(n))

    # silhouette coefficient
    sil_values = []
    for i in range(n):
        same = [points[j] for j in range(n) if labels[j] == labels[i] and j != i]
        if not same:
            sil_values.append(0.0)
            continue
        a = sum(_dist(points[i], q) for q in same) / len(same)
        b = math.inf
        for j in range(k):
            if j == labels[i]:
                continue
            other = [points[t] for t in range(n) if labels[t] == j]
            if other:
                b = min(b, sum(_dist(points[i], q) for q in other) / len(other))
        if b is math.inf:
            sil_values.append(0.0)
        else:
            sil_values.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    silhouette = sum(sil_values) / n if n else 0.0

    return {"centroids": centroids, "labels": labels, "iterations": iterations, "converged": converged,
            "final_shift": final_shift, "inertia": inertia, "silhouette": silhouette, "history": history}


def cluster_commodities(live: Dict[str, float], k: int = 2) -> Dict[str, Any]:
    k = 3 if k == 3 else 2
    vectors = commodity_vectors(live)
    points = [[v["volatility"], v["stress"]] for v in vectors]
    res = kmeans(points, k)

    # name clusters by centroid magnitude (higher = riskier)
    order = sorted(range(k), key=lambda j: -(res["centroids"][j][0] + res["centroids"][j][1]))
    names = ["HIGH_SYSTEMIC_RISK", "STABLE_NORMAL_FLOW"] if k == 2 else \
            ["HIGH_SYSTEMIC_RISK", "ELEVATED_WATCH", "STABLE_NORMAL_FLOW"]
    label_of = {cluster_idx: names[rank] for rank, cluster_idx in enumerate(order)}

    commodities = []
    for i, v in enumerate(vectors):
        j = res["labels"][i]
        commodities.append({**v, "cluster": label_of[j],
                            "distance_to_centroid": round(_dist(points[i], res["centroids"][j]), 2)})
    clusters = []
    for rank, j in enumerate(order):
        members = [c for c in commodities if c["cluster"] == label_of[j]]
        clusters.append({"label": label_of[j],
                         "centroid": [round(res["centroids"][j][0], 2), round(res["centroids"][j][1], 2)],
                         "size": len(members), "members": [m["name"] for m in members]})
    return {
        "status": "success",
        "algorithm": "K-Means (Lloyd's algorithm), k-means++ init, seed 42",
        "features": ["volatility_score_0_100", "supply_chain_stress_index_0_100"],
        "distance": "Euclidean",
        "k": k,
        "epsilon": 0.001,
        "max_iterations": 20,
        "iterations": res["iterations"],
        "converged": res["converged"],
        "final_centroid_shift": round(res["final_shift"], 6) if res["final_shift"] is not None else None,
        "inertia_wcss": round(res["inertia"], 2),
        "silhouette": round(res["silhouette"], 3),
        "centroid_history": res["history"],
        "clusters": clusters,
        "commodities": commodities,
        "high_risk_cluster": [c for c in commodities if c["cluster"] == "HIGH_SYSTEMIC_RISK"],
        "stable_cluster": [c for c in commodities if c["cluster"] == "STABLE_NORMAL_FLOW"],
        "data_note": ("Volatility = 30-day realised volatility from daily market prices; stress = noisy-OR supply exposure "
                      "to choke-point risk + 3-month price shock."
                      if any(v["feature_source"] == "MARKET" for v in vectors) else
                      "Indicator snapshot is modelled until market data is ingested: base values + coupling to choke-point risk."),
    }
