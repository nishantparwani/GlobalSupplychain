"""
ROUTING (Owner: Nishant)
========================
A* search over the maritime graph: f(n) = g(n) + h(n), with risk in the edge cost
and an admissible great-circle heuristic. Produces the safest and the fastest route.
"""

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

from Nishant.core import (
    ADJ, CANAL_DWELL_DAYS, CHOKE_POINTS, EXPOSURE_THRESHOLD, NM_PER_DAY, NODES, RISK_ALPHA,
    RISK_BETA, SAFETY_DECAY, SHIP_SPEED_KNOTS, coord, haversine_nm,
)


# =============================================================================
# UNIT 2: A* SEARCH  --  f(n) = g(n) + h(n)
# =============================================================================
def effective_risk(node: str, live: Dict[str, float]) -> float:
    """Risk level r in [0,10] = max(structural base hazard, live scraped severity)."""
    return max(NODES[node]["base_hazard"], live.get(node, 0.0))


def risk_penalty_days(r: float) -> float:
    """Exponential geopolitical risk penalty in equivalent days: P(r) = a(e^(b r) - 1)."""
    return RISK_ALPHA * (math.exp(RISK_BETA * r) - 1.0)


def edge_cost(u: str, v: str, live: Dict[str, float], risk_weight: float) -> Dict[str, float]:
    nm = ADJ[u][v]["nm"]
    transit = nm / NM_PER_DAY
    dwell = CANAL_DWELL_DAYS.get(v, 0.0)
    penalty = risk_penalty_days(effective_risk(v, live))
    return {
        "nm": nm,
        "transit_days": transit,
        "dwell_days": dwell,
        "risk_penalty_days": penalty,
        "cost": transit + dwell + risk_weight * penalty,
    }


def heuristic_days(n: str, goal: str) -> float:
    """h(n): great-circle distance to goal at service speed. Admissible and consistent
    because every edge is a polyline >= the great-circle chord, and all other cost terms
    (canal dwell, risk penalty) are non-negative."""
    return haversine_nm(coord(n), coord(goal)) / NM_PER_DAY


def a_star(start: str, goal: str, live: Dict[str, float], risk_weight: float) -> Optional[Dict[str, Any]]:
    counter = 0
    open_heap: List[Tuple[float, float, int, str]] = [(heuristic_days(start, goal), 0.0, counter, start)]
    g_score: Dict[str, float] = {start: 0.0}
    came_from: Dict[str, str] = {}
    closed: set = set()
    expansion_order: List[str] = []
    generated = 1
    max_open = 1

    while open_heap:
        _, g_cur, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return {
                "path": path,
                "objective_cost": g_cur,
                "nodes_expanded": len(expansion_order),
                "nodes_generated": generated,
                "max_open_set": max_open,
                "expansion_order": expansion_order,
            }
        closed.add(current)
        expansion_order.append(current)
        for nbr in ADJ[current]:
            if nbr in closed:
                continue
            tentative = g_cur + edge_cost(current, nbr, live, risk_weight)["cost"]
            if tentative < g_score.get(nbr, math.inf):
                g_score[nbr] = tentative
                came_from[nbr] = current
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic_days(nbr, goal), tentative, counter, nbr))
                generated += 1
        max_open = max(max_open, len(open_heap))
    return None


def unwrap_longitudes(points: List[Tuple[float, float]]) -> List[List[float]]:
    """Make longitudes continuous so trans-Pacific polylines do not wrap across the map."""
    out: List[List[float]] = []
    offset = 0.0
    prev: Optional[float] = None
    for lat, lng in points:
        if prev is not None:
            d = (lng + offset) - prev
            if d > 180:
                offset -= 360
            elif d < -180:
                offset += 360
        x = lng + offset
        out.append([round(lat, 4), round(x, 4)])
        prev = x
    return out


def route_metrics(label: str, search: Dict[str, Any], live: Dict[str, float]) -> Dict[str, Any]:
    path = search["path"]
    legs = []
    geometry_pts: List[Tuple[float, float]] = []
    total_nm = total_transit = total_dwell = total_penalty = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c = edge_cost(a, b, live, 1.0)
        total_nm += c["nm"]
        total_transit += c["transit_days"]
        total_dwell += c["dwell_days"]
        total_penalty += c["risk_penalty_days"]
        legs.append({
            "from": a, "to": b,
            "nm": round(c["nm"], 0),
            "transit_days": round(c["transit_days"], 2),
            "dwell_days": round(c["dwell_days"], 2),
            "risk_level": round(effective_risk(b, live), 1),
            "risk_penalty_days": round(c["risk_penalty_days"], 2),
        })
        seg = ADJ[a][b]["geometry"]
        geometry_pts.extend(seg if not geometry_pts else seg[1:])

    exposure_sum = sum(effective_risk(n, live) ** 2 for n in path)
    safety = 100.0 * math.exp(-exposure_sum / SAFETY_DECAY)
    threats = [
        {"node": n, "name": NODES[n]["name"], "severity": round(live.get(n, 0.0), 1)}
        for n in path if n in CHOKE_POINTS and live.get(n, 0.0) >= EXPOSURE_THRESHOLD
    ]
    return {
        "label": label,
        "path": path,
        "path_names": [NODES[n]["name"] for n in path],
        "eta_days": round(total_transit + total_dwell, 1),
        "transit_days": round(total_transit, 1),
        "canal_dwell_days": round(total_dwell, 1),
        "nautical_miles": int(round(total_nm, -1)),
        "safety_index": round(safety, 1),
        "risk_exposure_sum": round(exposure_sum, 1),
        "latent_risk_penalty_days": round(total_penalty, 1),
        "threat_exposure": threats,
        "choke_transits": [n for n in path if n in CHOKE_POINTS],
        "legs": legs,
        "geometry": unwrap_longitudes(geometry_pts),
        "search": {
            "objective_cost_days": round(search["objective_cost"], 2),
            "nodes_expanded": search["nodes_expanded"],
            "nodes_generated": search["nodes_generated"],
            "max_open_set": search["max_open_set"],
            "expansion_order": search["expansion_order"],
        },
    }


def compute_routes(start: str, end: str, live: Dict[str, float]) -> Dict[str, Any]:
    safe_search = a_star(start, end, live, risk_weight=1.0)
    direct_search = a_star(start, end, live, risk_weight=0.0)
    if not safe_search or not direct_search:
        return {"status": "error", "message": f"No navigable path between {start} and {end}."}
    safe = route_metrics("SAFE_A_STAR_DIVERSION", safe_search, live)
    direct = route_metrics("STANDARD_COMMERCIAL_DIRECT", direct_search, live)
    converged = safe["path"] == direct["path"]
    avoided = [t for t in direct["threat_exposure"] if t["node"] not in safe["path"]]
    d_days = round(safe["eta_days"] - direct["eta_days"], 1)
    d_safety = round(safe["safety_index"] - direct["safety_index"], 1)
    if converged:
        verdict = "Both options take the same path. No current threat on this lane is worth a detour."
    else:
        names = ", ".join(t["name"] for t in avoided) or "elevated-risk nodes"
        verdict = (f"Take the safer route. It adds {d_days} days but raises safety by {d_safety} points by avoiding {names}.")
    return {
        "status": "success",
        "start": start,
        "end": end,
        "safe_route": safe,
        "commercial_route": direct,
        "routes": [safe, direct],
        "comparison": {
            "converged": converged,
            "delta_eta_days": d_days,
            "delta_nautical_miles": safe["nautical_miles"] - direct["nautical_miles"],
            "delta_safety": d_safety,
            "hazards_avoided": avoided,
            "residual_exposure": safe["threat_exposure"],
            "verdict": verdict,
        },
    }


COST_MODEL = {
    "objective": "f(n) = g(n) + h(n)",
    "g": "g(n) = sum over edges of [ nm / (24 * v) + canal_dwell(v) + lambda * P(r_v) ]  (days)",
    "h": "h(n) = great_circle_nm(n, goal) / (24 * v)  (admissible, consistent lower bound)",
    "risk_penalty": f"P(r) = {RISK_ALPHA} * (e^({RISK_BETA} * r) - 1) days, r = max(base_hazard, live_severity) in [0,10]",
    "lambda_safe": 1.0,
    "lambda_commercial": 0.0,
    "safety_index": f"100 * exp(-sum(r_i^2) / {SAFETY_DECAY:.0f}) over all path nodes",
    "speed_knots": SHIP_SPEED_KNOTS,
    "design_note": "Risk is placed in the edge cost g(n), not in h(n): a risk term inside h(n) would "
                   "overestimate remaining cost, break admissibility and void A* optimality.",
}
