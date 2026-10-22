"""
INTELLIGENCE (Owner: Yash)
==========================
Threat intelligence pipeline: news relevance filter, forward-chaining rule-based
expert system, Gemini threat rating with a strict JSON schema, and the ingestion round.
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from Abhik import ai_models as ai
from Nishant.core import (
    CHOKE_POINTS, LLM, MAX_STORIES, MAX_STORIES_FOR_LLM, NODES, SCAN_DEBOUNCE_S, SCAN_LOCK,
    SCAN_WALL_CLOCK_S, STATE, _finalise_offline_events, live_severity,
)
from Yash.data_sources import FEEDS, sweep


# =============================================================================
# NEWS RELEVANCE (the sensors themselves live in Yash/data_sources.py)
# =============================================================================
RELEVANCE_TERMS = {
    r"\bship(s|ping)?\b": 2, r"\bvessels?\b": 2, r"\btankers?\b": 3, r"\bcontainer": 2, r"\bports?\b": 1,
    r"\bcanal\b": 3, r"\bstrait\b": 3, r"\bred sea\b": 4, r"\bsuez\b": 4, r"\bhormuz\b": 4, r"\bpanama\b": 3,
    r"\bmalacca\b": 4, r"\bbab[\s-]?el[\s-]?mandeb\b": 4, r"\bhouthi": 4, r"\bpira(te|cy)": 4, r"\bfreight\b": 2,
    r"\bmaritime\b": 2, r"\bnav(y|al)\b": 2, r"\bcargo\b": 1, r"\bbunker": 1, r"\blng\b": 2, r"\bcrude\b": 2,
    r"\battack": 2, r"\bdrone": 2, r"\bmissile": 2, r"\bseiz": 2, r"\bdrought\b": 2, r"\bstrike\b": 1,
    r"\bsupply chain": 2, r"\btrade\b": 1,
}


def relevance_score(article: Dict[str, Any]) -> int:
    text = f"{article['title']} {article['full_summary']}".lower()
    return sum(w for pat, w in RELEVANCE_TERMS.items() if re.search(pat, text))


def rank_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for a in articles:
        key = re.sub(r"[^a-z0-9]", "", a["title"].lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        a["relevance"] = relevance_score(a)
        if a["relevance"] >= 2:
            unique.append(a)
    unique.sort(key=lambda a: (a["relevance"], a["published_timestamp"] or ""), reverse=True)
    return unique


# =============================================================================
# UNIT 6: FORWARD-CHAINING RULE-BASED EXPERT SYSTEM
# =============================================================================
KEYWORD_FACTS: Dict[str, List[str]] = {
    "LOC_SUEZ": [r"\bsuez\b", r"\bport said\b"],
    "LOC_BAB_EL_MANDEB": [r"bab[\s-]?el[\s-]?mandeb", r"\bred sea\b", r"\bgulf of aden\b", r"\bhouthis?\b", r"\byemen"],
    "LOC_HORMUZ": [r"\bhormuz\b", r"\bpersian gulf\b", r"\bgulf of oman\b", r"\biran(ian)?\b", r"\bfujairah\b"],
    "LOC_PANAMA": [r"\bpanama\b", r"\bgatun\b"],
    "LOC_MALACCA": [r"\bmalacca\b", r"\bsingapore strait\b", r"\bstrait of singapore\b"],
    "KW_KINETIC": [r"\bmissiles?\b", r"\bdrones?\b", r"\battack(ed|s)?\b", r"\bexplosions?\b", r"\bwarships?\b", r"\bmilitary\b", r"\bnaval\b"],
    "KW_SEIZURE": [r"\bseiz(e|ed|es|ure)\b", r"\bdetain(ed|s)?\b", r"\bboarded\b"],
    "KW_PIRACY": [r"\bpira(te|tes|cy)\b", r"\barmed robbery\b", r"\bhijack"],
    "KW_WEATHER": [r"\bdrought\b", r"\bwater levels?\b", r"\bdraft (limit|restriction)", r"\btyphoon\b",
                   r"\bcyclone\b", r"\bhurricane\b", r"\bstorm\b", r"\bprecipitation\b", r"\brainfall\b"],
    "KW_LABOR": [r"\bport strike\b", r"\blabou?r (strike|dispute|action)\b", r"\bwalkout", r"\bdockworkers?\b",
                 r"\bindustrial action\b", r"\bunions?\b", r"\bstrike action\b"],
    "KW_DISRUPTION": [r"\bclos(e|ed|ure|ing)\b", r"\bsuspend", r"\bhalt", r"\bblockade", r"\bdivert", r"\breroute",
                      r"\bdelays?\b", r"\bcongestion\b", r"\bdisrupt"],
    "KW_CASUALTY": [r"\bkilled\b", r"\bsunk\b", r"\bsank\b", r"\bcasualt", r"\bcrew (missing|injured)\b"],
}

RULES: List[Tuple[str, set, set]] = [
    ("R01 location in Suez => at choke point", {"LOC_SUEZ"}, {"AT_CHOKEPOINT"}),
    ("R02 location in Red Sea/Aden => at choke point", {"LOC_BAB_EL_MANDEB"}, {"AT_CHOKEPOINT"}),
    ("R03 location in Gulf/Hormuz => at choke point", {"LOC_HORMUZ"}, {"AT_CHOKEPOINT"}),
    ("R04 location in Panama => at choke point", {"LOC_PANAMA"}, {"AT_CHOKEPOINT"}),
    ("R05 location in Malacca/Singapore Str => at choke point", {"LOC_MALACCA"}, {"AT_CHOKEPOINT"}),
    ("R06 kinetic indicators => military conflict", {"KW_KINETIC"}, {"EVT_MILITARY_CONFLICT"}),
    ("R07 seizure + Hormuz => military conflict, high", {"KW_SEIZURE", "LOC_HORMUZ"}, {"EVT_MILITARY_CONFLICT", "SEV_HIGH"}),
    ("R08 piracy indicators => piracy, medium", {"KW_PIRACY"}, {"EVT_PIRACY", "SEV_MEDIUM"}),
    ("R09 weather indicators => drought/weather, medium", {"KW_WEATHER"}, {"EVT_DROUGHT_WEATHER", "SEV_MEDIUM"}),
    ("R10 labour indicators => strike/labour, medium", {"KW_LABOR"}, {"EVT_STRIKE_LABOR", "SEV_MEDIUM"}),
    ("R11 military + choke point => high", {"EVT_MILITARY_CONFLICT", "AT_CHOKEPOINT"}, {"SEV_HIGH"}),
    ("R12 high + disruption => critical", {"SEV_HIGH", "KW_DISRUPTION"}, {"SEV_CRITICAL"}),
    ("R13 military + casualties => critical", {"EVT_MILITARY_CONFLICT", "KW_CASUALTY"}, {"SEV_CRITICAL"}),
    ("R14 weather + Panama + disruption => high", {"EVT_DROUGHT_WEATHER", "LOC_PANAMA", "KW_DISRUPTION"}, {"SEV_HIGH"}),
    ("R15 piracy + choke point => high", {"EVT_PIRACY", "AT_CHOKEPOINT"}, {"SEV_HIGH"}),
    ("R16 disruption + choke point => medium", {"KW_DISRUPTION", "AT_CHOKEPOINT"}, {"SEV_MEDIUM"}),
]

DIRECTIVES: Dict[Tuple[str, str], str] = {
    ("MILITARY_CONFLICT", "BAB_EL_MANDEB"): "Suspend Bab-el-Mandeb transits; divert Asia-Europe tonnage via Cape of Good Hope; plan +10 to +14 days and extra bunkers.",
    ("MILITARY_CONFLICT", "SUEZ"): "Hold Suez bookings for high-value and tanker cargo; reroute via Cape of Good Hope; reconfirm war-risk cover.",
    ("MILITARY_CONFLICT", "HORMUZ"): "Restrict Hormuz transits to daylight windows with naval liaison; shift liftings to Fujairah/Yanbu where feasible; expect war-risk premium spike.",
    ("MILITARY_CONFLICT", "PANAMA"): "Hold Panama transits pending security advisory; evaluate Suez or Cape routing for Asia-US East Coast loops.",
    ("MILITARY_CONFLICT", "MALACCA"): "Re-route VLCC and bulk tonnage via Sunda/Lombok straits; anticipate +2 to +3 days.",
    ("PIRACY", "MALACCA"): "Raise ship security level in the Singapore Strait; reinforce night anti-boarding watch; report to ReCAAP ISC.",
    ("PIRACY", "BAB_EL_MANDEB"): "Transit the Gulf of Aden via the IRTC applying BMP5 measures; consider embarked security.",
    ("PIRACY", "NONE"): "Apply BMP5 hardening, maintain 24h anti-piracy watch, and report to the regional reporting centre.",
    ("DROUGHT_WEATHER", "PANAMA"): "Reduce draft to authority limit, secure slots early or via auction, or divert via Suez/Cape.",
    ("DROUGHT_WEATHER", "NONE"): "Apply weather routing; add schedule buffer and review heavy-weather cargo securing.",
    ("STRIKE_LABOR", "NONE"): "Build a 48-72h buffer into terminal ETAs; pre-book alternate berth windows at nearby hubs.",
    ("GENERAL_ADVISORY", "NONE"): "Monitor; no routing change required.",
}
EVENT_PRIORITY = ["MILITARY_CONFLICT", "PIRACY", "DROUGHT_WEATHER", "STRIKE_LABOR"]
SEVERITY_SCORE = {"SEV_CRITICAL": 9, "SEV_HIGH": 7, "SEV_MEDIUM": 5}


def directive_for(event: str, choke: str) -> str:
    return (DIRECTIVES.get((event, choke)) or DIRECTIVES.get((event, "NONE"))
            or f"Monitor {NODES.get(choke, {}).get('name', 'region')}; add schedule buffer and review risk cover.")


def rule_engine_assess(article: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{article['title']} {article['full_summary']}".lower()
    facts: set = set()
    loc_hits: Dict[str, int] = {}
    for fact, patterns in KEYWORD_FACTS.items():
        hits = sum(len(re.findall(p, text)) for p in patterns)
        if hits:
            facts.add(fact)
            if fact.startswith("LOC_"):
                loc_hits[fact[4:]] = hits

    # forward chaining to a fixed point
    fired: List[str] = []
    changed = True
    while changed:
        changed = False
        for name, conds, concl in RULES:
            if conds <= facts and not concl <= facts:
                facts |= concl
                fired.append(name)
                changed = True

    choke = max(loc_hits, key=loc_hits.get) if loc_hits else "NONE"
    event = next((e for e in EVENT_PRIORITY if f"EVT_{e}" in facts), "GENERAL_ADVISORY")
    severity = max((SEVERITY_SCORE[s] for s in SEVERITY_SCORE if s in facts), default=0)
    if severity == 0:
        severity = 3 if (choke != "NONE" or event != "GENERAL_ADVISORY") else 1
    return {
        "choke_point_targeted": choke,
        "threat_severity": severity,
        "event_type": event,
        "tactical_directive": directive_for(event, choke),
        "rule_trace": fired,
        "facts": sorted(facts),
    }


# =============================================================================
# INFERENCE: GEMINI SYNTHESIS (strict schema) WITH RULE-ENGINE FALLBACK
# =============================================================================
VALID_CHOKES = set(CHOKE_POINTS) | {"NONE"}
VALID_EVENTS = set(EVENT_PRIORITY) | {"GENERAL_ADVISORY"}


def story_article(story: Dict[str, Any]) -> Dict[str, Any]:
    """Collapses a TF-IDF story cluster into one analysable document: the representative
    article plus the headlines of the other outlets covering the same event."""
    rep = story["representative"]
    others = [m["title"] for m in story["members"] if m is not rep][:4]
    summary = rep.get("full_summary", "")
    if others:
        summary = (summary + " Also reported: " + " | ".join(others)).strip()
    seen, coverage = set(), []
    for m in [rep] + [m for m in story["members"] if m is not rep]:
        if m["publisher_source"] in seen:
            continue
        seen.add(m["publisher_source"])
        coverage.append({"publisher": m["publisher_source"], "url": m.get("canonical_url"), "title": m["title"]})
    return {**rep, "full_summary": summary[:1100], "corroboration": story["corroboration"],
            "coverage": coverage[:8], "key_terms": story["key_terms"], "first_seen": story["first_seen"],
            "story_size": len(story["members"])}


def build_event(article: Dict[str, Any], assessment: Dict[str, Any], engine: str, note: str = "") -> Dict[str, Any]:
    url = article.get("canonical_url")
    return {
        "id": hashlib.sha1((url or article["title"]).encode()).hexdigest()[:10],
        "headline": article["title"],
        "summary": article["full_summary"] or "No summary provided by publisher.",
        "choke_point_targeted": assessment["choke_point_targeted"],
        "threat_severity": int(assessment["threat_severity"]),
        "event_type": assessment["event_type"],
        "tactical_directive": assessment["tactical_directive"],
        "verified_citation": {"url": url, "publisher": article["publisher_source"],
                              "published_timestamp": article.get("published_timestamp")},
        "corroboration": article.get("corroboration", 1),
        "coverage": article.get("coverage", []),
        "story_size": article.get("story_size", 1),
        "key_terms": article.get("key_terms", []),
        "confidence": assessment.get("confidence"),
        "source_mode": "LIVE",
        "analysis_engine": engine,
        "rule_trace": assessment.get("rule_trace", []),
        "analyst_note": note,
    }


def extract_json_array(text: str) -> List[Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e > s:
        return json.loads(text[s:e + 1])
    raise ValueError("no JSON array in LLM output")


async def gemini_assess(articles: List[Dict[str, Any]]) -> Optional[Dict[int, Dict[str, Any]]]:
    """One Gemini call for up to MAX_STORIES_FOR_LLM stories. Returns {story index: event}."""
    payload = [{"index": i, "title": a["title"], "summary": a["full_summary"][:600],
                "publisher": a["publisher_source"], "outlets_reporting": a.get("corroboration", 1),
                "published": a.get("published_timestamp")}
               for i, a in enumerate(articles)]
    prompt = f"""You are a maritime geopolitical risk analyst for a naval logistics command.
Each item is a news STORY: articles from several outlets clustered by topic similarity.
Assess each story for threats to commercial shipping at strategic choke points.

Return ONLY a JSON array. One object per story index you consider (you may skip irrelevant ones):
{{
  "article_index": <int, index from input>,
  "choke_point_targeted": one of ["SUEZ","HORMUZ","BAB_EL_MANDEB","PANAMA","MALACCA","NONE"],
  "threat_severity": <int 1-10>,
  "event_type": one of ["MILITARY_CONFLICT","DROUGHT_WEATHER","PIRACY","STRIKE_LABOR","GENERAL_ADVISORY"],
  "confidence": <float 0-1, how certain the assessment is given the evidence>,
  "tactical_directive": "<concrete maritime command, max 220 chars, e.g. 'Divert VLCC tankers via Cape of Good Hope; anticipate +14 days transit'>",
  "analyst_note": "<one-sentence justification, max 200 chars>"
}}

Severity rubric: 9-10 active attacks/closure; 7-8 credible imminent threat or major disruption;
5-6 recurring incidents or significant restrictions; 3-4 watch-level; 1-2 no shipping impact.
Stories reported by more outlets are better corroborated. Use only the story content.
Do not invent URLs; citations are attached by the system.

STORIES:
{json.dumps(payload, ensure_ascii=False)}"""
    text = await LLM.generate(prompt, json_mode=True, temperature=0.1)
    if not text:
        return None
    try:
        rows = extract_json_array(text)
    except Exception as exc:
        LLM.last_error = f"schema parse failed: {exc}"
        return None

    events: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("article_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(articles) or idx in events:
            continue
        choke = str(row.get("choke_point_targeted", "NONE")).upper()
        event = str(row.get("event_type", "GENERAL_ADVISORY")).upper()
        try:
            sev = int(round(float(row.get("threat_severity", 1))))
        except (TypeError, ValueError):
            sev = 1
        try:
            conf = round(ai.clamp(float(row.get("confidence", 0.6)), 0.0, 1.0), 2)
        except (TypeError, ValueError):
            conf = None
        if choke not in VALID_CHOKES or event not in VALID_EVENTS:
            fallback = rule_engine_assess(articles[idx])  # schema violation -> expert system
            events[idx] = build_event(articles[idx], fallback, "RULE_ENGINE", "LLM output failed schema validation.")
            continue
        directive = str(row.get("tactical_directive") or "").strip()[:260] or directive_for(event, choke)
        assessment = {"choke_point_targeted": choke, "threat_severity": max(1, min(10, sev)),
                      "event_type": event, "tactical_directive": directive, "rule_trace": [], "confidence": conf}
        events[idx] = build_event(articles[idx], assessment, f"GEMINI:{LLM.active_model}",
                                  str(row.get("analyst_note") or "")[:220])
    return events


def rule_event(a: Dict[str, Any]) -> Dict[str, Any]:
    assessment = rule_engine_assess(a)
    if assessment["choke_point_targeted"] == "NONE" and assessment["event_type"] == "GENERAL_ADVISORY":
        assessment["threat_severity"] = 1
    return build_event(a, assessment, "RULE_ENGINE", "Forward-chaining expert system inference.")


async def analyse_stories(articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Gemini assesses the top stories; the expert system covers the rest (and everything when
    Gemini is unavailable)."""
    llm_events = await gemini_assess(articles[:MAX_STORIES_FOR_LLM])
    out = []
    for i, a in enumerate(articles):
        if llm_events and i in llm_events:
            out.append(llm_events[i])
        elif llm_events is not None and i < MAX_STORIES_FOR_LLM:
            continue  # Gemini judged this story irrelevant to shipping
        else:
            out.append(rule_event(a))
    return out, ("GEMINI" if llm_events else "RULE_ENGINE")


RECENT_NEWS_DAYS = 21


def is_recent(a: Dict[str, Any]) -> bool:
    ts = a.get("published_timestamp")
    if not ts:
        return True
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).days <= RECENT_NEWS_DAYS
    except ValueError:
        return True


async def run_scan() -> None:
    """Single ingestion round. Called ONLY from POST /api/intelligence/ingest.
    Sweeps every sensor concurrently, then runs the inference pipeline:
    relevance ranking -> TF-IDF story clustering -> Gemini / expert-system assessment ->
    traffic anomaly + forecast -> market features -> Naive-Bayes fusion per choke point."""
    async with SCAN_LOCK:
        if time.time() - STATE["last_scan_ts"] < SCAN_DEBOUNCE_S:
            return
        data = await asyncio.to_thread(sweep, SCAN_WALL_CLOCK_S)
        sources = data["news_status"]
        articles = data["news"]

        ranked = rank_articles([a for a in articles if is_recent(a)])[:250]
        stories = ai.cluster_stories(ranked)[:MAX_STORIES]
        responded = sum(1 for s in sources if s["ok"])

        # traffic + markets (keep the previous analysis for any source that failed this round)
        traffic = dict(STATE.get("traffic") or {})
        for node, rows in data["traffic"].items():
            res = ai.analyse_traffic(rows)
            if res:
                traffic[node] = res
        markets = dict(STATE.get("markets") or {})
        for code, spec in data["market"].items():
            stats = ai.analyse_market(spec["series"])
            if stats:
                markets[code] = {k: v for k, v in spec.items() if k != "series"} | {"stats": stats}

        if stories:
            story_docs = [story_article(s) for s in stories]
            events, engine = await analyse_stories(story_docs)
            events.sort(key=lambda e: (e["threat_severity"], e.get("corroboration", 1)), reverse=True)
            STATE.update(
                mode="LIVE", engine=engine, events=events,
                model=LLM.active_model if engine == "GEMINI" else None,
                llm_note=None if engine == "GEMINI" else LLM.unavailable_note(),
                note=(f"{len(ranked)} relevant articles from {responded} of {len(FEEDS)} news sources clustered into "
                      f"{len(stories)} stories; fused with {len(traffic)} traffic and {len(markets)} market series."),
            )
        else:
            STATE.update(
                mode="BASELINE", engine="BASELINE", events=_finalise_offline_events(), model=None, llm_note=None,
                note=("News sources could not be reached, so the baseline news dataset is shown."
                      if responded == 0 else
                      "Sources responded but carried no maritime-relevant articles; baseline news retained."),
            )

        STATE["traffic"] = traffic
        STATE["markets"] = markets
        STATE["fusion"] = {c: ai.fuse(c, NODES[c]["base_hazard"], STATE["events"], traffic, markets)
                           for c in CHOKE_POINTS} if (traffic or markets or stories) else {}
        STATE["rag_docs"] = [{k: a.get(k) for k in ("title", "full_summary", "canonical_url", "publisher_source",
                                                     "published_timestamp")} for a in ranked]
        STATE["rag"] = ai.TfidfIndex([ai.doc_text(a) for a in ranked]) if ranked else None
        STATE["sources"] = sources
        STATE["traffic_sources"] = data["traffic_status"]
        STATE["market_sources"] = data["market_status"]
        STATE["articles"] = [{k: a[k] for k in ("title", "canonical_url", "publisher_source",
                                                 "published_timestamp", "relevance")} for a in ranked[:60]]
        STATE["articles_ingested"] = len(articles)
        STATE["stories_found"] = len(stories)
        STATE["sweep_s"] = data["elapsed_s"]
        STATE["scanned_at"] = datetime.now(timezone.utc).isoformat()
        STATE["last_scan_ts"] = time.time()
        STATE["scan_count"] += 1


def intelligence_payload() -> Dict[str, Any]:
    return {
        "status": "success",
        "mode": STATE["mode"],
        "engine": STATE["engine"],
        "model": STATE.get("model"),
        "note": STATE.get("note"),
        "llm_note": STATE.get("llm_note"),
        "scanned_at": STATE["scanned_at"],
        "scan_count": STATE["scan_count"],
        "sources": STATE["sources"],
        "traffic_sources": STATE["traffic_sources"],
        "market_sources": STATE["market_sources"],
        "sources_responded": sum(1 for s in STATE["sources"] if s.get("ok")),
        "sources_total": len(FEEDS),
        "articles_ingested": STATE.get("articles_ingested", 0),
        "stories_found": STATE.get("stories_found", 0),
        "sweep_s": STATE.get("sweep_s"),
        "articles": STATE["articles"],
        "events": STATE["events"],
        "choke_severity": live_severity(),
        "llm": LLM.info(),
    }
