"""
AI MODELS for Global Chain AI (pure Python, no numpy, so every step is inspectable)
==================================================================================

  (e) TF-IDF vector space model + cosine similarity
        - story clustering: groups articles from different outlets that report the same event
          (leader-follower clustering on centroids), giving a corroboration count per story
        - retrieval: top-k article search used as the Copilot's RAG memory
  (f) Time-series intelligence on AIS-derived transit counts (IMF PortWatch)
        - change-point detection (least-squares single split) finds a lasting regime shift,
          such as a closure, so the "normal" baseline is taken from before it
        - robust anomaly detection: 7-day mean vs. median/MAD of that baseline
        - Holt's linear exponential smoothing, (alpha, beta) fitted by grid search on
          one-step-ahead squared error, 14-day forecast with an 80% band
  (g) Market feature extraction (realised volatility, momentum, price z-score, drawdown)
  (h) Naive-Bayes evidence fusion: per choke point, independent evidence sources
        (prior hazard, news, traffic anomaly, rerouting, markets) are combined as
        log-likelihood ratios into a posterior disruption probability
  (i) Noisy-OR supply-chain exposure: P(a commodity's supply is hit) =
        1 - prod_c (1 - w_c * P(disruption at c))
"""

import math
import re
from collections import Counter
from datetime import date, datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# (e) TF-IDF VECTOR SPACE MODEL
# =============================================================================
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if in
into is it its itself just more most my no nor not now of off on once only or other our out over own same she should so some
such than that the their them then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your new says said say week year years day days amid via per one two three
""".split())


def tokenize(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\-]+", (text or "").lower()):
        w = w.strip("-")
        if len(w) < 3 or w in STOPWORDS or w.isdigit():
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def doc_text(a: Dict[str, Any]) -> str:
    # the headline carries most of the signal, so it is counted twice
    return f"{a['title']} {a['title']} {a.get('full_summary', '')}"


class TfidfIndex:
    """Sublinear-tf, smoothed-idf, L2-normalised sparse vectors."""

    def __init__(self, docs: List[str]):
        self.tokens = [tokenize(d) for d in docs]
        n = len(docs)
        df: Counter = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vectors = [self.vectorize_tokens(t) for t in self.tokens]

    def vectorize_tokens(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        v = {t: (1.0 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def vectorize(self, text: str) -> Dict[str, float]:
        return self.vectorize_tokens(tokenize(text))

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q = self.vectorize(query)
        scored = [(i, cosine(q, v)) for i, v in enumerate(self.vectors)]
        scored = [s for s in scored if s[1] > 0.04]
        scored.sort(key=lambda s: -s[1])
        return scored[:k]


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _normalise(v: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


def cluster_stories(articles: List[Dict[str, Any]], threshold: float = 0.33) -> List[Dict[str, Any]]:
    """Leader-follower clustering: each article (in relevance order) joins the most similar
    story centroid if cosine >= threshold, otherwise it starts a new story. Centroids are the
    re-normalised sum of member vectors. O(n * stories)."""
    if not articles:
        return []
    index = TfidfIndex([doc_text(a) for a in articles])
    order = sorted(range(len(articles)), key=lambda i: -articles[i].get("relevance", 0))
    stories: List[Dict[str, Any]] = []
    for i in order:
        v = index.vectors[i]
        best, best_sim = None, 0.0
        for s in stories:
            sim = cosine(v, s["centroid"])
            if sim > best_sim:
                best, best_sim = s, sim
        if best is not None and best_sim >= threshold:
            best["members"].append(i)
            best["sims"].append(round(best_sim, 3))
            acc = dict(best["_sum"])
            for t, x in v.items():
                acc[t] = acc.get(t, 0.0) + x
            best["_sum"] = acc
            best["centroid"] = _normalise(acc)
        else:
            stories.append({"members": [i], "sims": [1.0], "_sum": dict(v), "centroid": dict(v)})

    out = []
    for s in stories:
        members = [articles[i] for i in s["members"]]
        publishers = sorted({m["publisher_source"] for m in members})
        # representative: prefer trade press (richer summaries), then relevance
        rep = max(members, key=lambda m: (m.get("feed_kind") != "aggregator", m.get("relevance", 0),
                                           len(m.get("full_summary", ""))))
        stamps = sorted(m["published_timestamp"] for m in members if m.get("published_timestamp"))
        top_terms = sorted(s["centroid"].items(), key=lambda kv: -kv[1])[:6]
        out.append({
            "representative": rep,
            "members": members,
            "publishers": publishers,
            "corroboration": len(publishers),
            "first_seen": stamps[0] if stamps else None,
            "last_seen": stamps[-1] if stamps else None,
            "relevance": max(m.get("relevance", 0) for m in members),
            "key_terms": [t for t, _ in top_terms],
            "cohesion": round(sum(s["sims"]) / len(s["sims"]), 3),
        })
    out.sort(key=lambda s: -(s["relevance"] * (1 + math.log(s["corroboration"]))))
    return out


# =============================================================================
# (f) TRAFFIC: ANOMALY DETECTION + HOLT FORECAST
# =============================================================================
def rolling_mean(xs: List[float], w: int) -> List[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out


def holt_fit(series: List[float]) -> Dict[str, Any]:
    """Holt's linear trend method. level_t = a*y_t + (1-a)(level + trend);
    trend_t = b*(level_t - level_{t-1}) + (1-b)*trend. Grid search on one-step-ahead SSE."""
    best = None
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for b in [0.01, 0.05, 0.1, 0.2, 0.3]:
            level, trend = series[0], (series[min(7, len(series) - 1)] - series[0]) / max(1, min(7, len(series) - 1))
            sse, n = 0.0, 0
            for y in series[1:]:
                pred = level + trend
                sse += (y - pred) ** 2
                n += 1
                new_level = a * y + (1 - a) * (level + trend)
                trend = b * (new_level - level) + (1 - b) * trend
                level = new_level
            if best is None or sse < best["sse"]:
                best = {"alpha": a, "beta": b, "sse": sse, "level": level, "trend": trend,
                        "rmse": math.sqrt(sse / max(1, n))}
    return best


def holt_forecast(fit: Dict[str, Any], horizon: int = 14) -> List[Dict[str, float]]:
    out = []
    for h in range(1, horizon + 1):
        y = max(0.0, fit["level"] + h * fit["trend"])
        band = 1.2816 * fit["rmse"] * math.sqrt(1 + (h - 1) * fit["alpha"] ** 2)  # 80% interval
        out.append({"h": h, "mean": round(y, 2), "lo": round(max(0.0, y - band), 2), "hi": round(y + band, 2)})
    return out


def _robust_baseline(values: List[float]) -> Tuple[float, float]:
    med = median(values)
    mad = median([abs(v - med) for v in values])
    scale = max(1.4826 * mad, 0.05 * abs(med), 0.5)
    return med, scale


CHANGE_MIN_SEGMENT_DAYS = 30
CHANGE_MIN_SHIFT = 0.35


def detect_change_point(xs: List[float]) -> Optional[Dict[str, Any]]:
    """Single change-point by least squares: the split k that minimises the summed squared error of two
    constant-mean segments (equivalently, maximises between-segment variance). Accepted only if both
    segments last CHANGE_MIN_SEGMENT_DAYS and the level shifts by at least CHANGE_MIN_SHIFT."""
    n, m = len(xs), CHANGE_MIN_SEGMENT_DAYS
    if n < 2 * m:
        return None
    total = sum(xs)
    sq_total = sum(x * x for x in xs)
    best_k, best_gain, left = None, 0.0, 0.0
    for k in range(1, n):
        left += xs[k - 1]
        if k < m or n - k < m:
            continue
        right = total - left
        # SSE(split) = sq_total - left^2/k - right^2/(n-k); minimising it maximises the gain below
        gain = left * left / k + right * right / (n - k) - total * total / n
        if gain > best_gain:
            best_k, best_gain = k, gain
    if best_k is None:
        return None
    before, after = sum(xs[:best_k]) / best_k, sum(xs[best_k:]) / (n - best_k)
    shift = (after - before) / before if before > 0 else 0.0
    if abs(shift) < CHANGE_MIN_SHIFT:
        return None
    explained = best_gain / max(1e-9, sq_total - total * total / n)
    return {"index": best_k, "before_mean": before, "after_mean": after, "shift": shift, "r2": explained}


def analyse_traffic(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """rows: daily transit records, oldest first."""
    if len(rows) < 60:
        return None
    totals = [float(r["total"]) for r in rows]
    tankers = [float(r["tanker"]) for r in rows]
    containers = [float(r["container"]) for r in rows]
    roll = rolling_mean(totals, 7)
    roll_tank = rolling_mean(tankers, 7)
    roll_cont = rolling_mean(containers, 7)
    # baseline: 7-day means from ~1 year ago up to 5 weeks ago (excludes the current episode) ...
    lo = max(7, len(rows) - 372)
    hi = len(rows) - 35
    # ... unless a lasting regime shift (e.g. a closure) began inside that window: then "normal" is the
    # year before the shift, otherwise the disrupted level would slowly become the baseline
    cp = detect_change_point(roll[7:])
    change_point = None
    if cp:
        k = cp["index"] + 7
        change_point = {"date": rows[k]["date"], "before_mean": round(cp["before_mean"], 1),
                        "after_mean": round(cp["after_mean"], 1), "shift": round(cp["shift"], 3),
                        "variance_explained": round(cp["r2"], 2), "days_since": len(rows) - k}
        if k < hi and k - 7 >= 30:
            lo, hi = max(7, k - 365), k
            change_point["baseline_from"] = "before change point"
    base = roll[lo:hi] if hi - lo >= 30 else roll[7:max(8, len(rows) - 14)]
    base_tank = roll_tank[lo:hi] if hi - lo >= 30 else roll_tank[7:max(8, len(rows) - 14)]
    base_cont = roll_cont[lo:hi] if hi - lo >= 30 else roll_cont[7:max(8, len(rows) - 14)]
    med, scale = _robust_baseline(base)
    recent = roll[-1]
    z = (recent - med) / scale
    pct = (recent / med - 1.0) if med > 0 else 0.0
    t_med = median(base_tank)
    c_med = median(base_cont)

    fit = holt_fit(totals[-120:])
    fc = holt_forecast(fit, 14)
    last_date = rows[-1]["date"]
    try:
        stale_days = (date.today() - date.fromisoformat(last_date)).days
    except ValueError:
        stale_days = None
    trend_per_week = fit["trend"] * 7
    direction = "falling" if trend_per_week < -0.04 * max(med, 1) else "rising" if trend_per_week > 0.04 * max(med, 1) else "stable"
    return {
        "as_of": last_date,
        "data_lag_days": stale_days,
        "recent_7d_mean": round(recent, 1),
        "baseline_median": round(med, 1),
        "baseline_scale": round(scale, 2),
        "robust_z": round(z, 2),
        "pct_vs_baseline": round(pct, 3),
        "tanker_pct_vs_baseline": round(roll_tank[-1] / t_med - 1.0, 3) if t_med > 0 else None,
        "container_pct_vs_baseline": round(roll_cont[-1] / c_med - 1.0, 3) if c_med > 0 else None,
        "anomaly": abs(z) >= 3.0,
        "direction": direction,
        "change_point": change_point,
        "forecast_model": {"method": "Holt linear exponential smoothing", "alpha": fit["alpha"],
                           "beta": fit["beta"], "rmse": round(fit["rmse"], 2), "fit_window_days": min(120, len(totals))},
        "forecast": fc,
        "series": [{"date": r["date"], "total": r["total"], "ma7": round(m, 1)} for r, m in zip(rows[-300:], roll[-300:])],
    }


# =============================================================================
# (g) MARKET FEATURES
# =============================================================================
def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def analyse_market(series: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    closes = [p["close"] for p in series if p["close"] and p["close"] > 0]
    if len(closes) < 30:
        return None
    last = closes[-1]
    logret = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    vol30 = _std(logret[-21:]) * math.sqrt(252)
    vol1y = _std(logret) * math.sqrt(252)
    w63 = closes[-63:]
    sd63 = _std(w63)
    z90 = (last - sum(w63) / len(w63)) / sd63 if sd63 > 0 else 0.0
    ret = lambda n: (last / closes[-1 - n] - 1.0) if len(closes) > n else None
    hi = max(closes)
    rank = sum(1 for c in closes if c <= last) / len(closes)
    return {
        "last": round(last, 4),
        "as_of": series[-1]["date"],
        "ret_1d": round(ret(1), 4) if ret(1) is not None else None,
        "ret_1m": round(ret(21), 4) if ret(21) is not None else None,
        "ret_3m": round(ret(63), 4) if ret(63) is not None else None,
        "vol_30d_ann": round(vol30, 4),
        "vol_1y_ann": round(vol1y, 4),
        "vol_ratio": round(vol30 / vol1y, 2) if vol1y > 0 else None,
        "z_score_3m": round(z90, 2),
        "drawdown_from_1y_high": round(last / hi - 1.0, 4),
        "percentile_1y": round(rank, 3),
        "spark": [round(c, 4) for c in closes[-90:]],
    }


# =============================================================================
# (h) NAIVE-BAYES EVIDENCE FUSION (log-odds)
# =============================================================================
def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# which market series carry information about which choke point
MARKET_LINKS: Dict[str, List[str]] = {
    "HORMUZ": ["BRENT", "TTF", "TANKER_FRT", "TANK_EQ"],
    "BAB_EL_MANDEB": ["CONT_EQ", "TANKER_FRT", "BRENT"],
    "SUEZ": ["CONT_EQ", "TANKER_FRT", "BRENT"],
    "PANAMA": ["SOY", "CORN", "HH", "BULK_FRT"],
    "MALACCA": ["BRENT", "BULK_FRT", "FE"],
}
DIVERSION_TARGETS = {"SUEZ", "BAB_EL_MANDEB"}  # Cape of Good Hope surges signal Red Sea avoidance

FUSION_MODEL = {
    "type": "Naive-Bayes log-odds fusion (evidence assumed conditionally independent)",
    "prior": "p0 = 0.04 + 0.06 * base_hazard",
    "news": "LLR = 0.55 * (s - 4) * c,  s = top story severity, c = min(1.4, 0.7 + 0.15 * outlets); no story: -0.4",
    "traffic": "LLR = clamp(3.5 * drop, -0.8, 3.0) * min(1, |z| / 2),  drop = 1 - recent7 / baseline_median",
    "rerouting": "Suez/Bab-el-Mandeb only: LLR = clamp(2 * Cape traffic increase, 0, 1.2)",
    "market": "LLR = clamp(0.35 * mean(clamp(z_3m, -2, 3)) over linked series, -0.4, 0.9)",
    "posterior": "P = sigmoid(logit(p0) + sum LLR);  risk = 10 * P",
}


def fuse(node: str, base_hazard: float, stories: List[Dict[str, Any]],
         traffic: Dict[str, Dict[str, Any]], markets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    p0 = 0.04 + 0.06 * base_hazard
    evidence = [{"source": "prior", "label": "Structural hazard", "llr": round(logit(p0), 3),
                 "detail": f"Base hazard {base_hazard:.0f}/10 gives prior P = {p0:.2f}"}]

    mine = [s for s in stories if s.get("choke_point_targeted") == node]
    if mine:
        top = max(mine, key=lambda s: (s["threat_severity"], s.get("corroboration", 1)))
        outlets = top.get("corroboration", 1)
        c = min(1.4, 0.7 + 0.15 * outlets)
        llr = 0.55 * (top["threat_severity"] - 4) * c
        evidence.append({"source": "news", "label": "News reporting", "llr": round(llr, 3),
                         "detail": f"{len(mine)} stories; top severity {top['threat_severity']}/10 "
                                   f"reported by {outlets} outlet{'s' if outlets != 1 else ''}"})
    else:
        evidence.append({"source": "news", "label": "News reporting", "llr": -0.4,
                         "detail": "No current stories target this passage"})

    t = traffic.get(node)
    if t:
        drop = -t["pct_vs_baseline"]
        llr = clamp(3.5 * drop, -0.8, 3.0) * min(1.0, abs(t["robust_z"]) / 2.0)
        evidence.append({"source": "traffic", "label": "Ship traffic (AIS)", "llr": round(llr, 3),
                         "detail": f"{t['recent_7d_mean']:.0f} transits/day vs normal {t['baseline_median']:.0f} "
                                   f"({t['pct_vs_baseline'] * 100:+.0f}%, z = {t['robust_z']:+.1f})"
                                   + (f"; level shift since {t['change_point']['date']}" if t.get("change_point") else "")})
    if node in DIVERSION_TARGETS and traffic.get("CAPE_OF_GOOD_HOPE"):
        cape = traffic["CAPE_OF_GOOD_HOPE"]
        inc = cape["pct_vs_baseline"]
        llr = clamp(2.0 * inc, 0.0, 1.2)
        evidence.append({"source": "rerouting", "label": "Cape of Good Hope diversions", "llr": round(llr, 3),
                         "detail": f"Cape transits {inc * 100:+.0f}% vs normal"})

    linked = [markets[c] for c in MARKET_LINKS.get(node, []) if c in markets and markets[c].get("stats")]
    if linked:
        zs = [clamp(m["stats"]["z_score_3m"], -2.0, 3.0) for m in linked]
        score = sum(zs) / len(zs)
        llr = clamp(0.35 * score, -0.4, 0.9)
        names = ", ".join(f"{m['name'].split(' (')[0]} {m['stats']['z_score_3m']:+.1f}σ" for m in linked)
        evidence.append({"source": "market", "label": "Market pricing", "llr": round(llr, 3),
                         "detail": f"3-month price z-scores: {names}"})

    total = sum(e["llr"] for e in evidence)
    p = sigmoid(total)
    observed = {e["source"] for e in evidence} - {"prior"}
    if not mine:
        observed.discard("news")
    return {
        "node": node,
        "prior": round(p0, 3),
        "posterior": round(p, 3),
        "risk": round(10 * p, 1),
        "log_odds": round(total, 3),
        "evidence": evidence,
        "signals_observed": sorted(observed),
        "confidence": "high" if len(observed) >= 3 else "medium" if len(observed) == 2 else "low",
    }


# =============================================================================
# (i) NOISY-OR SUPPLY EXPOSURE
# =============================================================================
def noisy_or(exposure: Dict[str, float], p_by_choke: Dict[str, float]) -> float:
    q = 1.0
    for c, w in exposure.items():
        q *= 1.0 - w * p_by_choke.get(c, 0.0)
    return 1.0 - q
