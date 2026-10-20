"""
SENSORS: external data sources for Global Chain AI
==================================================
Three independent sensor families feed the agent's world model:

  1. NEWS      : 8 maritime trade-press RSS feeds + Google News topic searches per choke point
  2. TRAFFIC   : IMF PortWatch daily vessel transit counts per choke point (AIS-derived)
  3. MARKETS   : Yahoo Finance daily futures / freight / equity series (energy, grains,
                 metals, dry-bulk + tanker freight, container-line equity)

Every fetcher is blocking (run it in a worker thread), has a short timeout, never raises,
and returns (status, payload) so one dead source can never break an ingestion round.
"""

import re
import time
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

HTTP_TIMEOUT_S = 6.0
HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


# =============================================================================
# 1. NEWS FEEDS
# =============================================================================
def _gnews(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query + ' when:7d')}&hl=en-US&gl=US&ceid=US:en"


FEEDS: List[Dict[str, Any]] = [
    # maritime trade press
    {"id": "GCAPTAIN", "publisher": "gCaptain", "kind": "trade_press", "urls": ["https://gcaptain.com/feed/"]},
    {"id": "MAREX", "publisher": "The Maritime Executive", "kind": "trade_press",
     "urls": ["https://maritime-executive.com/articles.rss", "https://maritime-executive.com/rss"]},
    {"id": "HELLENIC", "publisher": "Hellenic Shipping News", "kind": "trade_press",
     "urls": ["https://www.hellenicshippingnews.com/feed/"]},
    {"id": "SPLASH", "publisher": "Splash247", "kind": "trade_press", "urls": ["https://splash247.com/feed/"]},
    {"id": "FREIGHTWAVES", "publisher": "FreightWaves", "kind": "trade_press",
     "urls": ["https://www.freightwaves.com/news/feed"]},
    {"id": "CONTAINER_NEWS", "publisher": "Container News", "kind": "trade_press",
     "urls": ["https://container-news.com/feed/"]},
    {"id": "UN_NEWS", "publisher": "UN News", "kind": "institutional",
     "urls": ["https://news.un.org/feed/subscribe/en/news/topic/economic-development/feed/rss.xml"]},
    # Google News topic searches: aggregate wire services (Reuters, AP, Bloomberg, FT ...) per choke point
    {"id": "GN_HORMUZ", "publisher": "Google News: Hormuz", "kind": "aggregator",
     "urls": [_gnews('"Strait of Hormuz" (tanker OR shipping OR vessel)')]},
    {"id": "GN_RED_SEA", "publisher": "Google News: Red Sea", "kind": "aggregator",
     "urls": [_gnews('("Red Sea" OR "Bab el-Mandeb" OR Houthi) shipping')]},
    {"id": "GN_SUEZ", "publisher": "Google News: Suez", "kind": "aggregator",
     "urls": [_gnews('"Suez Canal" (transits OR shipping OR vessels)')]},
    {"id": "GN_PANAMA", "publisher": "Google News: Panama", "kind": "aggregator",
     "urls": [_gnews('"Panama Canal" (transits OR draft OR shipping)')]},
    {"id": "GN_MALACCA", "publisher": "Google News: Malacca", "kind": "aggregator",
     "urls": [_gnews('("Strait of Malacca" OR "Singapore Strait") (piracy OR robbery OR shipping)')]},
]

TRACKING_KEYS = {"fbclid", "gclid", "dclid", "yclid", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src",
                 "cmpid", "icid", "ocid", "igshid", "_hsenc", "_hsmi", "mkt_tok", "spm"}


def clean_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        p = urlparse(url.strip())
        if p.scheme not in ("http", "https") or not p.netloc:
            return None
        query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in TRACKING_KEYS]
        return urlunparse((p.scheme, p.netloc.lower(), p.path, p.params, urlencode(query), ""))
    except Exception:
        return None


def strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    plain = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s*(The post .* appeared first on .*)$", "", plain)
    return plain


def parse_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _local(tag: str) -> str:
    return tag.split("}")[-1].lower()


def parse_feed(xml_bytes: bytes, feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parses RSS 2.0 and Atom with the stdlib. Google News items carry the real outlet in
    <source> and append ' - Outlet' to the title; both are normalised here."""
    root = ET.fromstring(xml_bytes)
    items = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        fields: Dict[str, str] = {}
        for child in el:
            name = _local(child.tag)
            if name == "link":
                href = child.attrib.get("href")
                rel = child.attrib.get("rel", "alternate")
                if href and rel == "alternate":
                    fields.setdefault("link", href)
                elif child.text and child.text.strip():
                    fields.setdefault("link", child.text.strip())
            elif name in ("title", "description", "summary", "encoded", "content", "source",
                          "pubdate", "published", "updated", "date", "guid"):
                fields.setdefault(name, "".join(child.itertext()).strip())
        title = strip_html(fields.get("title"))
        if not title:
            continue
        publisher = feed["publisher"]
        if feed["kind"] == "aggregator":
            outlet = fields.get("source")
            if outlet:
                publisher = outlet
                title = re.sub(r"\s+-\s+" + re.escape(outlet) + r"\s*$", "", title)
        summary = strip_html(fields.get("description") or fields.get("summary")
                             or fields.get("encoded") or fields.get("content"))
        if feed["kind"] == "aggregator" and summary.startswith(title[:40]):
            summary = ""  # Google News descriptions only repeat the headline
        link = fields.get("link") or (fields.get("guid") if (fields.get("guid") or "").startswith("http") else None)
        items.append({
            "title": title[:300],
            "full_summary": summary[:900],
            "canonical_url": clean_url(link),
            "published_timestamp": parse_date(fields.get("pubdate") or fields.get("published")
                                              or fields.get("updated") or fields.get("date")),
            "publisher_source": publisher,
            "feed_id": feed["id"],
            "feed_kind": feed["kind"],
        })
    return items


def fetch_feed(feed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    status: Dict[str, Any] = {"id": feed["id"], "publisher": feed["publisher"], "kind": feed["kind"],
                              "ok": False, "items": 0, "error": None}
    for url in feed["urls"]:
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_S, allow_redirects=True)
            resp.raise_for_status()
            items = parse_feed(resp.content, feed)
            if items:
                status.update(ok=True, items=len(items), error=None)
                return status, items
            status["error"] = "empty feed"
        except requests.Timeout:
            status["error"] = "timeout"
        except requests.HTTPError as exc:
            status["error"] = f"HTTP {exc.response.status_code if exc.response is not None else '?'}"
        except requests.RequestException as exc:
            status["error"] = f"network ({type(exc).__name__})"
        except ET.ParseError:
            status["error"] = "not a valid RSS document"
        except Exception as exc:
            status["error"] = type(exc).__name__
    return status, []


# =============================================================================
# 2. TRAFFIC: IMF PORTWATCH DAILY CHOKE POINT TRANSITS
# =============================================================================
PORTWATCH_URL = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
                 "Daily_Chokepoints_Data/FeatureServer/0/query")
PORTWATCH_IDS = {
    "SUEZ": "chokepoint1",
    "PANAMA": "chokepoint2",
    "BAB_EL_MANDEB": "chokepoint4",
    "MALACCA": "chokepoint5",
    "HORMUZ": "chokepoint6",
    "CAPE_OF_GOOD_HOPE": "chokepoint7",
}
TRAFFIC_HISTORY_DAYS = 420


def fetch_transits(node: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Daily transit counts (all vessels, tankers, containers) for one choke point, oldest first."""
    status = {"id": f"PORTWATCH_{node}", "node": node, "ok": False, "days": 0, "error": None}
    since = (date.today() - timedelta(days=TRAFFIC_HISTORY_DAYS)).isoformat()
    params = {
        "where": f"portid='{PORTWATCH_IDS[node]}' AND date >= DATE '{since}'",
        "outFields": "date,n_total,n_tanker,n_container,n_dry_bulk,capacity",
        "orderByFields": "date ASC",
        "resultRecordCount": 2000,
        "returnGeometry": "false",
        "f": "json",
    }
    try:
        resp = requests.get(PORTWATCH_URL, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            status["error"] = str(data["error"].get("message", "query error"))[:120]
            return status, []
        rows = []
        for f in data.get("features", []):
            a = f.get("attributes", {})
            if a.get("date") is None or a.get("n_total") is None:
                continue
            d = a["date"]
            if isinstance(d, (int, float)):  # some ArcGIS layers return epoch ms
                d = datetime.fromtimestamp(d / 1000, tz=timezone.utc).date().isoformat()
            rows.append({"date": str(d)[:10], "total": int(a["n_total"]), "tanker": int(a.get("n_tanker") or 0),
                         "container": int(a.get("n_container") or 0), "dry_bulk": int(a.get("n_dry_bulk") or 0),
                         "capacity": int(a.get("capacity") or 0)})
        status.update(ok=bool(rows), days=len(rows), error=None if rows else "no rows")
        return status, rows
    except requests.Timeout:
        status["error"] = "timeout"
    except Exception as exc:
        status["error"] = type(exc).__name__
    return status, []


# =============================================================================
# 3. MARKETS: YAHOO FINANCE DAILY SERIES
# =============================================================================
# role: COMMODITY rows are clustered; FREIGHT / EQUITY rows are market-stress evidence only.
MARKET_SERIES: List[Dict[str, Any]] = [
    {"symbol": "BZ=F",  "code": "BRENT", "name": "Brent crude",         "role": "COMMODITY", "unit": "USD/bbl"},
    {"symbol": "TTF=F", "code": "TTF",   "name": "LNG / EU gas (TTF)",  "role": "COMMODITY", "unit": "EUR/MWh"},
    {"symbol": "NG=F",  "code": "HH",    "name": "US natural gas",      "role": "COMMODITY", "unit": "USD/MMBtu"},
    {"symbol": "ZW=F",  "code": "WHEAT", "name": "Wheat",               "role": "COMMODITY", "unit": "USc/bu"},
    {"symbol": "ZS=F",  "code": "SOY",   "name": "Soybeans",            "role": "COMMODITY", "unit": "USc/bu"},
    {"symbol": "ZC=F",  "code": "CORN",  "name": "Corn",                "role": "COMMODITY", "unit": "USc/bu"},
    {"symbol": "HG=F",  "code": "CU",    "name": "Copper",              "role": "COMMODITY", "unit": "USD/lb"},
    {"symbol": "ALI=F", "code": "AL",    "name": "Aluminium",           "role": "COMMODITY", "unit": "USD/t"},
    {"symbol": "TIO=F", "code": "FE",    "name": "Iron ore",            "role": "COMMODITY", "unit": "USD/t"},
    {"symbol": "BWET",  "code": "TANKER_FRT", "name": "Tanker freight (BWET)",    "role": "FREIGHT", "unit": "USD"},
    {"symbol": "BDRY",  "code": "BULK_FRT",   "name": "Dry-bulk freight (BDRY)",  "role": "FREIGHT", "unit": "USD"},
    {"symbol": "ZIM",   "code": "CONT_EQ",    "name": "Container line equity (ZIM)", "role": "EQUITY", "unit": "USD"},
    {"symbol": "FRO",   "code": "TANK_EQ",    "name": "Tanker owner equity (Frontline)", "role": "EQUITY", "unit": "USD"},
]
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_series(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    status = {"id": spec["code"], "symbol": spec["symbol"], "ok": False, "points": 0, "error": None}
    try:
        resp = requests.get(YAHOO_URL.format(symbol=quote_plus(spec["symbol"])),
                            params={"range": "1y", "interval": "1d"}, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        result = (resp.json().get("chart") or {}).get("result") or []
        if not result:
            status["error"] = "no data"
            return status, []
        r = result[0]
        closes = ((r.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        stamps = r.get("timestamp") or []
        rows = [{"date": datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat(), "close": float(c)}
                for t, c in zip(stamps, closes) if c is not None]
        status.update(ok=len(rows) >= 30, points=len(rows), currency=(r.get("meta") or {}).get("currency"),
                      error=None if len(rows) >= 30 else "too few points")
        return status, rows
    except requests.Timeout:
        status["error"] = "timeout"
    except Exception as exc:
        status["error"] = type(exc).__name__
    return status, []


# =============================================================================
# CONCURRENT SENSOR SWEEP
# =============================================================================
def sweep(wall_clock_s: float = 14.0) -> Dict[str, Any]:
    """Runs every fetcher concurrently under one wall-clock ceiling."""
    t0 = time.time()
    jobs: List[Tuple[str, Any, Any]] = (
        [("news", f, fetch_feed) for f in FEEDS]
        + [("traffic", n, fetch_transits) for n in PORTWATCH_IDS]
        + [("market", s, fetch_series) for s in MARKET_SERIES]
    )
    pool = ThreadPoolExecutor(max_workers=16)
    futures = [pool.submit(fn, arg) for _, arg, fn in jobs]
    wait(futures, timeout=wall_clock_s)
    out: Dict[str, Any] = {"news": [], "news_status": [], "traffic": {}, "traffic_status": [],
                           "market": {}, "market_status": []}
    for (kind, arg, _), fut in zip(jobs, futures):
        ok = fut.done() and not fut.exception()
        if kind == "news":
            status, items = fut.result() if ok else ({"id": arg["id"], "publisher": arg["publisher"], "kind": arg["kind"],
                                                       "ok": False, "items": 0, "error": "timeout"}, [])
            out["news_status"].append(status)
            out["news"].extend(items)
        elif kind == "traffic":
            status, rows = fut.result() if ok else ({"id": f"PORTWATCH_{arg}", "node": arg, "ok": False,
                                                      "days": 0, "error": "timeout"}, [])
            out["traffic_status"].append(status)
            if rows:
                out["traffic"][arg] = rows
        else:
            status, rows = fut.result() if ok else ({"id": arg["code"], "symbol": arg["symbol"], "ok": False,
                                                      "points": 0, "error": "timeout"}, [])
            out["market_status"].append(status)
            if status.get("ok"):
                out["market"][arg["code"]] = {**arg, "series": rows}
    pool.shutdown(wait=False, cancel_futures=True)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out
