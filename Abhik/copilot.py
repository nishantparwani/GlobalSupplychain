"""
COPILOT (Owner: Abhik)
======================
AI analyst (Copilot): a deterministic rule-based analyst that is always available, and a
tool-using Gemini agent (function calling over the platform's own models, with TF-IDF RAG).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from Abhik import ai_models as ai
from Nishant.core import (
    AGENT_MAX_STEPS, AIS_ZONES, CHOKE_POINTS, EXPOSURE_THRESHOLD, LAST_ROUTE, LLM, NODES, STATE,
    STRATEGIC_NOTES, TRACKER, live_severity,
)
from Nishant.routing import compute_routes
from Yash.clustering import cluster_commodities
from Yash.data_sources import MARKET_SERIES, PORTWATCH_IDS


# =============================================================================
# COPILOT: DETERMINISTIC ANALYST (always available) + GEMINI REFINEMENT
# =============================================================================
ALIASES = {
    "shanghai": "SHANGHAI", "ningbo": "SHANGHAI", "singapore strait": "MALACCA", "singapore": "SINGAPORE",
    "tokyo": "TOKYO", "yokohama": "TOKYO", "mumbai": "MUMBAI", "bombay": "MUMBAI", "nhava sheva": "MUMBAI",
    "dubai": "DUBAI", "jebel ali": "DUBAI", "rotterdam": "ROTTERDAM", "london": "LONDON",
    "felixstowe": "LONDON", "new york": "NEW_YORK", "nyc": "NEW_YORK", "los angeles": "LOS_ANGELES",
    "long beach": "LOS_ANGELES", "suez": "SUEZ", "hormuz": "HORMUZ", "bab el mandeb": "BAB_EL_MANDEB",
    "bab-el-mandeb": "BAB_EL_MANDEB", "red sea": "BAB_EL_MANDEB", "gulf of aden": "BAB_EL_MANDEB",
    "panama": "PANAMA", "malacca": "MALACCA", "cape of good hope": "CAPE_OF_GOOD_HOPE",
    "good hope": "CAPE_OF_GOOD_HOPE", "gibraltar": "GIBRALTAR",
}
REPRESENTATIVE_LANES = {
    "HORMUZ": [("DUBAI", "ROTTERDAM"), ("DUBAI", "SHANGHAI")],
    "SUEZ": [("SHANGHAI", "ROTTERDAM"), ("MUMBAI", "LONDON")],
    "BAB_EL_MANDEB": [("SHANGHAI", "ROTTERDAM"), ("DUBAI", "NEW_YORK")],
    "PANAMA": [("TOKYO", "NEW_YORK"), ("LOS_ANGELES", "NEW_YORK")],
    "MALACCA": [("SHANGHAI", "MUMBAI"), ("SHANGHAI", "ROTTERDAM")],
}


def find_mentions(text: str) -> List[str]:
    found: List[Tuple[int, str]] = []
    taken: List[Tuple[int, int]] = []
    for alias in sorted(ALIASES, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in taken):
                continue
            taken.append(span)
            found.append((m.start(), ALIASES[alias]))
    out: List[str] = []
    for _, node in sorted(found):
        if node not in out:
            out.append(node)
    return out


def fmt_path(path: List[str]) -> str:
    return " → ".join(NODES[n]["name"] for n in path)


GROUP_NAMES = {"HIGH_SYSTEMIC_RISK": "High risk", "ELEVATED_WATCH": "Watch", "STABLE_NORMAL_FLOW": "Stable"}
EVENT_NAMES = {"MILITARY_CONFLICT": "conflict", "PIRACY": "piracy", "DROUGHT_WEATHER": "drought / weather",
               "STRIKE_LABOR": "labour action", "GENERAL_ADVISORY": "advisory"}


def route_brief(start: str, end: str, live: Dict[str, float], title: str = "") -> str:
    res = compute_routes(start, end, live)
    if res["status"] != "success":
        return res["message"]
    s, d, c = res["safe_route"], res["commercial_route"], res["comparison"]
    lines = [f"### {title or 'Route comparison'}: {NODES[start]['name']} to {NODES[end]['name']}",
             f"- **Safest route:** {fmt_path(s['path'])}",
             f"  **{s['eta_days']} days**, {s['nautical_miles']:,} nm, safety **{s['safety_index']}/100**",
             f"- **Fastest route:** {fmt_path(d['path'])}",
             f"  **{d['eta_days']} days**, {d['nautical_miles']:,} nm, safety **{d['safety_index']}/100**",
             f"- **Recommendation:** {c['verdict']}"]
    if s["threat_exposure"]:
        lines.append("- **Still exposed to:** " +
                     ", ".join(f"{t['name']} (risk {t['severity']:.0f}/10)" for t in s["threat_exposure"]) +
                     ". There is no way around it from this port.")
    return "\n".join(lines)


def explain_route() -> str:
    start = LAST_ROUTE.get("start", "SHANGHAI")
    end = LAST_ROUTE.get("end", "ROTTERDAM")
    live = LAST_ROUTE.get("live") or live_severity()
    res = compute_routes(start, end, live)
    if res["status"] != "success":
        return res["message"]
    s, d, c = res["safe_route"], res["commercial_route"], res["comparison"]
    lines = [f"### Why this route: {NODES[start]['name']} to {NODES[end]['name']}"]
    if c["converged"]:
        lines += [f"Both options follow the same path: {fmt_path(s['path'])}.",
                  f"No hotspot on this lane is dangerous enough to justify a detour, so the fastest route "
                  f"(**{s['eta_days']} days**) is also the recommended one."]
        return "\n".join(lines)
    risky = [leg for leg in d["legs"] if leg["risk_level"] >= EXPOSURE_THRESHOLD]
    lines.append(f"The fastest route takes **{d['eta_days']} days** but passes through:")
    for leg in risky:
        lines.append(f"- **{NODES[leg['to']]['name']}**, currently at risk **{leg['risk_level']:.0f}/10**")
    lines += ["",
              f"The safest route goes {fmt_path(s['path'])} instead. It takes **{s['eta_days']} days** "
              f"(**{c['delta_eta_days']:+} days**) and raises the safety score from "
              f"**{d['safety_index']}** to **{s['safety_index']}** out of 100.",
              "",
              "The system weighs extra sailing time against danger. Risk grows sharply as a hotspot heats up, "
              "so a few extra days at sea is the better trade once a passage reaches the high-risk range."]
    return "\n".join(lines)


def scenario_closure(chokes: List[str], ports: List[str]) -> str:
    base_live = live_severity()
    scen_live = dict(base_live)
    for c in chokes:
        scen_live[c] = 10.0
    lanes = [(ports[0], ports[1])] if len(ports) >= 2 else REPRESENTATIVE_LANES.get(chokes[0], [])
    names = ", ".join(NODES[c]["name"] for c in chokes)
    lines = [f"### What if {names} closes?"]
    for a, b in lanes:
        base = compute_routes(a, b, base_live)
        scen = compute_routes(a, b, scen_live)
        if base["status"] != "success" or scen["status"] != "success":
            continue
        bs, ss = base["safe_route"], scen["safe_route"]
        lines.append(f"**{NODES[a]['name']} to {NODES[b]['name']}**")
        lines.append(f"- Today: {fmt_path(bs['path'])} ({bs['eta_days']} days, safety {bs['safety_index']})")
        lines.append(f"- After closure: {fmt_path(ss['path'])} ({ss['eta_days']} days, safety {ss['safety_index']})")
        trapped = [c for c in chokes if c in ss["path"]]
        if trapped:
            lines.append(f"- **No way around:** every route from this port passes "
                         f"{', '.join(NODES[c]['name'] for c in trapped)}. Options are to hold cargo, "
                         f"move it overland to another port, or sail with war-risk cover.")
        else:
            lines.append(f"- Extra time: **{round(ss['eta_days'] - bs['eta_days'], 1):+} days**")
    base_cl = {c["name"]: c for c in cluster_commodities(base_live)["commodities"]}
    scen_cl = cluster_commodities(scen_live)
    exposed = [c for c in scen_cl["commodities"] if any(ch in c["exposure"] for ch in chokes)]
    if exposed:
        lines.append("**Goods affected**")
        for c in sorted(exposed, key=lambda x: -x["stress"]):
            b = base_cl[c["name"]]
            moved = "" if b["cluster"] == c["cluster"] else f" (was {GROUP_NAMES[b['cluster']]})"
            lines.append(f"- {c['name']}: volatility {b['volatility']} to **{c['volatility']}**, "
                         f"supply stress {b['stress']} to **{c['stress']}**, now **{GROUP_NAMES[c['cluster']]}**{moved}")
    if "HORMUZ" in chokes:
        lines.append("- **For tankers:** stop new oil and LNG bookings loading inside the Gulf, shift loading to "
                     "Fujairah or Yanbu where possible, and expect insurance and freight costs to jump.")
    return "\n".join(lines)


def cluster_brief() -> str:
    cl = cluster_commodities(live_severity())
    lines = ["### Goods at risk right now"]
    for c in cl["clusters"]:
        lines.append(f"- **{GROUP_NAMES[c['label']]}:** {', '.join(c['members'])}")
    top = max(cl["commodities"], key=lambda x: x["volatility"] + x["stress"])
    lines.append(f"Most exposed: **{top['name']}** (volatility {top['volatility']}, supply stress {top['stress']}).")
    return "\n".join(lines)


def threat_brief(chokes: List[str]) -> str:
    events = STATE["events"]
    if chokes:
        events = [e for e in events if e["choke_point_targeted"] in chokes] or events
    lines = ["### Current situation"]
    for e in events[:5]:
        where = NODES[e["choke_point_targeted"]]["name"] if e["choke_point_targeted"] in NODES else "General"
        lines.append(f"- **{where}, risk {e['threat_severity']}/10 ({EVENT_NAMES.get(e['event_type'], 'advisory')}):** {e['headline']}")
        lines.append(f"  Action: {e['tactical_directive']}")
    sev = live_severity()
    lines.append("Risk by passage: " + ", ".join(f"{NODES[c]['name']} {max(sev[c], NODES[c]['base_hazard']):.0f}" for c in CHOKE_POINTS))
    return "\n".join(lines)


def current_fusion() -> Dict[str, Dict[str, Any]]:
    """Fused risk per choke point; before any ingestion it is computed from baseline news only."""
    if STATE.get("fusion"):
        return STATE["fusion"]
    return {c: ai.fuse(c, NODES[c]["base_hazard"], STATE["events"], {}, {}) for c in CHOKE_POINTS}


def signals_brief(chokes: List[str]) -> str:
    fusion = current_fusion()
    lines = ["### Why the risk levels are what they are"]
    for c in chokes or sorted(CHOKE_POINTS, key=lambda x: -fusion[x]["risk"]):
        f = fusion[c]
        lines.append(f"**{NODES[c]['name']}: risk {f['risk']:.1f}/10** (probability of disruption "
                     f"{f['posterior'] * 100:.0f}%, confidence {f['confidence']})")
        for e in f["evidence"]:
            arrow = "raises" if e["llr"] > 0.05 else "lowers" if e["llr"] < -0.05 else "neutral"
            lines.append(f"- {e['label']} ({arrow}, {e['llr']:+.2f}): {e['detail']}")
    return "\n".join(lines)


def traffic_brief(chokes: List[str]) -> str:
    traffic = STATE.get("traffic") or {}
    if not traffic:
        return "### Ship traffic\nNo traffic data yet. Press **Get latest news** to load AIS transit counts."
    nodes = chokes or [n for n in PORTWATCH_IDS if n in traffic]
    lines = ["### Ship traffic through the passages (daily transits, 7-day average)"]
    for n in nodes:
        t = traffic.get(n)
        if not t:
            continue
        f14 = t["forecast"][-1]
        lines.append(f"- **{NODES[n]['name']}:** {t['recent_7d_mean']:.0f}/day vs normal {t['baseline_median']:.0f} "
                     f"(**{t['pct_vs_baseline'] * 100:+.0f}%**{', unusual' if t['anomaly'] else ''}"
                     f"{', disrupted since ' + t['change_point']['date'] if t.get('change_point') else ''}); trend {t['direction']}, "
                     f"about {f14['mean']:.0f}/day expected in 2 weeks (data to {t['as_of']})")
    if TRACKER.running:
        z = TRACKER.zone_counts(AIS_ZONES)
        live = [f"{NODES[k]['name']} {v['vessels']}" for k, v in z.items() if v["vessels"]]
        if live:
            lines.append("Live AIS positions right now: " + ", ".join(live) + " vessels in range.")
    return "\n".join(lines)


def market_brief() -> str:
    markets = STATE.get("markets") or {}
    if not markets:
        return "### Markets\nNo market data yet. Press **Get latest news** to load prices."
    rows = sorted(markets.values(), key=lambda m: -abs(m["stats"]["z_score_3m"]))
    lines = ["### Market signals (biggest moves vs. the last 3 months first)"]
    for m in rows[:8]:
        s = m["stats"]
        lines.append(f"- **{m['name']}:** {s['last']:,.2f} {m['unit']}, 1 month {s['ret_1m'] * 100:+.1f}%, "
                     f"{s['z_score_3m']:+.1f}σ vs 3-month average, volatility {s['vol_30d_ann'] * 100:.0f}%")
    return "\n".join(lines)


def search_articles(query: str, k: int = 5) -> List[Dict[str, Any]]:
    idx = STATE.get("rag")
    if not idx:
        return []
    out = []
    for i, score in idx.search(query, k):
        d = STATE["rag_docs"][i]
        out.append({"title": d["title"], "publisher": d["publisher_source"], "url": d["canonical_url"],
                    "published": d["published_timestamp"], "summary": (d["full_summary"] or "")[:400],
                    "similarity": round(score, 3)})
    return out


def related_reporting(query: str) -> str:
    hits = search_articles(query, 3)
    if not hits:
        return ""
    lines = ["", "**Related reporting**"]
    for h in hits:
        lines.append(f"- [{h['title']}]({h['url']}) ({h['publisher']})" if h["url"] else f"- {h['title']} ({h['publisher']})")
    return "\n".join(lines)


HELP_TEXT = """### Things you can ask
- `Why avoid Suez between Shanghai and Rotterdam?`
- `What happens to oil tankers if Hormuz closes?`
- `How much traffic is going through Hormuz?`
- `Why is the Red Sea rated so high?`
- `Which goods are most at risk?`
- `What are oil and freight markets doing?`"""


def rule_copilot(cmd: str) -> str:
    t = cmd.lower()
    mentions = find_mentions(t)
    ports = [m for m in mentions if NODES[m]["kind"] == "PORT"]
    chokes = [m for m in mentions if m in CHOKE_POINTS]
    if re.search(r"\b(clos|block|shut|seal|mined|blockade|halt)", t) and chokes:
        return scenario_closure(chokes, ports)
    if re.search(r"\b(why|explain|reason|justif|avoid)", t) and re.search(r"(rout|divert|cape|detour|path)", t):
        if len(ports) >= 2:
            LAST_ROUTE.update(start=ports[0], end=ports[1], live=live_severity())
        return explain_route()
    if len(ports) >= 2:
        return route_brief(ports[0], ports[1], live_severity())
    if re.search(r"(traffic|transit|ships? (are|going)|vessels?|\bais\b|how many ships|congest)", t):
        return traffic_brief(chokes)
    if re.search(r"(market|price|brent|oil price|freight rate|stock|futures|ttf)", t):
        return market_brief()
    if re.search(r"(why|evidence|how sure|confiden|signal|rated|score)", t) and (chokes or "risk" in t):
        return signals_brief(chokes)
    if re.search(r"(cluster|k-?means|commodit|volatil|goods|cargo|grain|wheat)", t):
        return cluster_brief()
    if re.search(r"(threat|intel|risk|status|situation|brief|happening|news)", t) or chokes:
        return threat_brief(chokes) + related_reporting(cmd)
    return HELP_TEXT + "\n\n" + threat_brief([])


# =============================================================================
# COPILOT AGENT: Gemini function calling over the platform's own models (+ RAG)
# =============================================================================
NODE_ENUM = list(NODES)
TOOL_DECLARATIONS = [
    {"name": "get_chokepoint_risk",
     "description": "Fused disruption risk for a choke point with the evidence behind it (news, ship-traffic "
                    "anomaly, Cape rerouting, market pricing), plus its latest news events.",
     "parameters": {"type": "object", "properties": {
         "chokepoint": {"type": "string", "enum": CHOKE_POINTS}}, "required": ["chokepoint"]}},
    {"name": "compare_routes",
     "description": "Safest vs fastest sea route between two locations under current risk, optionally with "
                    "choke points forced closed (what-if). Returns days, nautical miles, safety score, path.",
     "parameters": {"type": "object", "properties": {
         "origin": {"type": "string", "enum": NODE_ENUM},
         "destination": {"type": "string", "enum": NODE_ENUM},
         "closed_chokepoints": {"type": "array", "items": {"type": "string", "enum": CHOKE_POINTS}}},
         "required": ["origin", "destination"]}},
    {"name": "search_news",
     "description": "Search the ingested maritime news (TF-IDF retrieval). Returns titles, outlets, URLs, summaries.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "get_ship_traffic",
     "description": "Daily vessel transits through a passage from AIS data: recent level vs normal, anomaly score, "
                    "trend, 14-day forecast, and live AIS vessel counts when the live map is running.",
     "parameters": {"type": "object", "properties": {
         "chokepoint": {"type": "string", "enum": list(PORTWATCH_IDS)}}, "required": ["chokepoint"]}},
    {"name": "get_market_data",
     "description": "Latest market prices and statistics (1-month change, volatility, z-score vs 3 months) for "
                    "energy, grains, metals, freight indices and shipping equities.",
     "parameters": {"type": "object", "properties": {
         "codes": {"type": "array", "items": {"type": "string", "enum": [s["code"] for s in MARKET_SERIES]}}}}},
    {"name": "get_goods_at_risk",
     "description": "Commodities grouped by price volatility and supply-chain stress, with the probability that "
                    "each one's seaborne supply is hit.",
     "parameters": {"type": "object", "properties": {}}},
]


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "get_chokepoint_risk":
            c = str(args.get("chokepoint", "")).upper()
            if c not in CHOKE_POINTS:
                return {"error": f"unknown choke point {c}"}
            f = current_fusion()[c]
            events = [{k: e[k] for k in ("headline", "threat_severity", "event_type", "tactical_directive", "corroboration")}
                      | {"url": e["verified_citation"]["url"], "publisher": e["verified_citation"]["publisher"]}
                      for e in STATE["events"] if e["choke_point_targeted"] == c][:4]
            return {"name": NODES[c]["name"], "risk_0_10": f["risk"], "disruption_probability": f["posterior"],
                    "confidence": f["confidence"], "evidence": f["evidence"], "events": events,
                    "strategic_note": STRATEGIC_NOTES.get(c)}
        if name == "compare_routes":
            a, b = str(args.get("origin", "")).upper(), str(args.get("destination", "")).upper()
            if a not in NODES or b not in NODES or a == b:
                return {"error": "origin and destination must be two different known locations", "valid": NODE_ENUM}
            live = live_severity()
            closed = [c for c in (args.get("closed_chokepoints") or []) if c in CHOKE_POINTS]
            for c in closed:
                live[c] = 10.0
            res = compute_routes(a, b, live)
            if res["status"] != "success":
                return {"error": res["message"]}
            if not closed:
                LAST_ROUTE.clear()
                LAST_ROUTE.update(start=a, end=b, live=dict(live), override={})
            pick = lambda r: {"path": [NODES[n]["name"] for n in r["path"]], "eta_days": r["eta_days"],
                              "nautical_miles": r["nautical_miles"], "safety_0_100": r["safety_index"],
                              "hotspots_crossed": [t["name"] for t in r["threat_exposure"]]}
            return {"closed": closed, "safest": pick(res["safe_route"]), "fastest": pick(res["commercial_route"]),
                    "verdict": res["comparison"]["verdict"]}
        if name == "search_news":
            hits = search_articles(str(args.get("query", "")), int(args.get("k") or 5))
            return {"results": hits} if hits else {"results": [], "note": "no ingested news matches (run an ingestion first)"}
        if name == "get_ship_traffic":
            n = str(args.get("chokepoint", "")).upper()
            t = (STATE.get("traffic") or {}).get(n)
            if not t:
                return {"error": "no traffic data loaded for this passage yet"}
            out = {k: t.get(k) for k in ("as_of", "recent_7d_mean", "baseline_median", "pct_vs_baseline", "robust_z",
                                     "anomaly", "direction", "tanker_pct_vs_baseline", "container_pct_vs_baseline", "change_point")}
            out["forecast_7d"] = t["forecast"][6]
            out["forecast_14d"] = t["forecast"][13]
            if TRACKER.running and n in AIS_ZONES:
                out["live_ais_now"] = TRACKER.zone_counts({n: AIS_ZONES[n]})[n]
            return out
        if name == "get_market_data":
            markets = STATE.get("markets") or {}
            codes = args.get("codes") or list(markets)
            return {"markets": [{"code": c, "name": markets[c]["name"], "unit": markets[c]["unit"],
                                 **{k: markets[c]["stats"][k] for k in ("last", "as_of", "ret_1d", "ret_1m", "ret_3m",
                                                                        "vol_30d_ann", "z_score_3m", "percentile_1y")}}
                                for c in codes if c in markets]} if markets else {"error": "no market data loaded yet"}
        if name == "get_goods_at_risk":
            cl = cluster_commodities(live_severity())
            return {"groups": [{"group": GROUP_NAMES[c["label"]], "members": c["members"]} for c in cl["clusters"]],
                    "items": [{k: c[k] for k in ("name", "volatility", "stress", "supply_hit_probability", "cluster")}
                              for c in cl["commodities"]]}
        return {"error": f"unknown tool {name}"}
    except Exception as exc:  # a tool failure must never break the agent loop
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


AGENT_SYSTEM = """You are the analyst inside GLOBAL CHAIN AI, a shipping risk monitor.
Use the tools to get facts before answering; call several tools in one turn when useful.
Answer in concise markdown (short heading, bullets, bold key figures, no tables, max ~250 words).
Only use numbers returned by tools. When you use news, cite it as [title](url).
Explain in everyday shipping terms; say "ship-traffic data", "market prices", "news reports" rather than
naming algorithms. If the tools cannot answer, say so plainly."""


async def agent_answer(question: str) -> Optional[Dict[str, Any]]:
    """ReAct-style loop: Gemini decides which tools to call, the server executes them, results go
    back to Gemini, until it writes the final answer (max AGENT_MAX_STEPS model calls)."""
    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": question}]}]
    trace: List[Dict[str, Any]] = []
    for step in range(AGENT_MAX_STEPS):
        last = step == AGENT_MAX_STEPS - 1
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": AGENT_SYSTEM}]},
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {"temperature": 0.2},
        }
        if last:
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        content = await LLM.call(body)
        if not content:
            return None
        content["role"] = "model"
        contents.append(content)  # echoed verbatim so any thought signatures are preserved
        calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not calls:
            text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought")).strip()
            return {"text": text, "trace": trace} if text else None
        responses = []
        for c in calls:
            args = c.get("args") or {}
            result = run_tool(c["name"], args)
            trace.append({"tool": c["name"], "args": args, "ok": "error" not in result})
            fr: Dict[str, Any] = {"name": c["name"], "response": {"result": result}}
            if c.get("id"):
                fr["id"] = c["id"]
            responses.append({"functionResponse": fr})
        contents.append({"role": "user", "parts": responses})
    return None
