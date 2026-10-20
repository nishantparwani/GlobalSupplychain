"""
LIVE AIS VESSEL TRACKING (free providers)
=========================================
  aisstream.io : global terrestrial AIS over WebSocket. Free key from https://aisstream.io
                 (sign in with GitHub) -> AISSTREAM_API_KEY in .env. Subscribed to boxes
                 around the choke points and main lanes so message volume stays manageable.
  Digitraffic  : Finnish Transport Infrastructure Agency open AIS (Baltic Sea), no key needed,
                 polled every 60 s. Works out of the box, so the live layer is never empty.

Create the tracker after .env is loaded (it reads AISSTREAM_API_KEY in __init__).
Nothing connects at start-up. The tracker starts on the first /api/vessels request and
stops itself after IDLE_STOP_S without requests (nobody looking at the map).
"""

import asyncio
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from websockets.asyncio.client import connect as ws_connect
except Exception:  # websockets < 13 or not installed
    try:
        from websockets import connect as ws_connect  # type: ignore
    except Exception:
        ws_connect = None

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
DIGITRAFFIC_LOC = "https://meri.digitraffic.fi/api/ais/v1/locations"
DIGITRAFFIC_META = "https://meri.digitraffic.fi/api/ais/v1/vessels"
DT_HEADERS = {"Digitraffic-User": "GlobalChainAI/1.0", "Accept-Encoding": "gzip"}

MAX_AGE_S = 45 * 60       # drop positions older than this
MAX_VESSELS = 12000
IDLE_STOP_S = 10 * 60

# [[lat_min, lon_min], [lat_max, lon_max]] around choke points and dense lanes
AIS_BOXES = {
    "Strait of Hormuz / Gulf of Oman": [[22.0, 51.0], [28.5, 60.5]],
    "Red Sea / Bab-el-Mandeb / Gulf of Aden": [[11.0, 32.0], [30.5, 51.5]],
    "Suez / Eastern Mediterranean": [[30.0, 26.0], [36.5, 36.5]],
    "Malacca / Singapore Straits": [[-1.5, 95.0], [8.0, 106.0]],
    "Panama Canal approaches": [[6.5, -82.0], [10.5, -77.0]],
    "Strait of Gibraltar": [[34.5, -8.0], [37.5, -3.5]],
    "English Channel / North Sea": [[48.5, -6.0], [55.0, 8.5]],
    "Cape of Good Hope": [[-37.5, 14.0], [-32.0, 24.0]],
    "Shanghai / East China Sea": [[28.5, 120.0], [33.0, 124.5]],
}


def ship_category(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    if 80 <= code <= 89:
        return "tanker"
    if 70 <= code <= 79:
        return "cargo"
    if 60 <= code <= 69:
        return "passenger"
    if code == 30:
        return "fishing"
    if code in (31, 32, 52):
        return "tug"
    if code == 35:
        return "military"
    if code in (36, 37):
        return "pleasure"
    if 40 <= code <= 59:
        return "special"
    return "other" if code else "unknown"


NAV_STATUS = {0: "under way (engine)", 1: "at anchor", 2: "not under command", 3: "restricted manoeuvrability",
              4: "constrained by draught", 5: "moored", 6: "aground", 7: "fishing", 8: "under way (sailing)",
              14: "AIS-SART active"}


def _km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


class AISTracker:
    def __init__(self) -> None:
        self.key = (os.getenv("AISSTREAM_API_KEY") or "").strip().strip('"').strip("'")
        self.vessels: Dict[int, Dict[str, Any]] = {}
        self.static: Dict[int, Dict[str, Any]] = {}
        self.tasks: List[asyncio.Task] = []
        self.last_request = 0.0
        self.providers: Dict[str, Dict[str, Any]] = {
            "aisstream": {"name": "aisstream.io (global, key)", "enabled": bool(self.key and ws_connect),
                          "state": "idle" if self.key else "no key", "messages": 0, "error": None,
                          "coverage": list(AIS_BOXES)},
            "digitraffic": {"name": "Digitraffic (Baltic, open)", "enabled": True, "state": "idle",
                            "messages": 0, "error": None, "coverage": ["Baltic Sea"]},
        }
        if self.key and not ws_connect:
            self.providers["aisstream"]["error"] = "python package 'websockets' not installed"

    # ---------------- lifecycle ----------------
    @property
    def running(self) -> bool:
        return any(not t.done() for t in self.tasks)

    def touch(self) -> None:
        self.last_request = time.time()
        if not self.running:
            self.tasks = [asyncio.create_task(self._digitraffic_loop()), asyncio.create_task(self._watchdog())]
            if self.providers["aisstream"]["enabled"]:
                self.tasks.append(asyncio.create_task(self._aisstream_loop()))

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(30)
            self._prune()
            if time.time() - self.last_request > IDLE_STOP_S:
                for t in self.tasks:
                    if t is not asyncio.current_task():
                        t.cancel()
                for p in self.providers.values():
                    if p["state"] in ("connected", "polling", "connecting"):
                        p["state"] = "paused (idle)"
                return

    def _prune(self) -> None:
        cutoff = time.time() - MAX_AGE_S
        for m in [m for m, v in self.vessels.items() if v["ts"] < cutoff]:
            del self.vessels[m]
        if len(self.vessels) > MAX_VESSELS:
            for m, _ in sorted(self.vessels.items(), key=lambda kv: kv[1]["ts"])[: len(self.vessels) - MAX_VESSELS]:
                del self.vessels[m]

    def _upsert(self, mmsi: int, **fields: Any) -> None:
        v = self.vessels.get(mmsi) or {"mmsi": mmsi}
        v.update({k: x for k, x in fields.items() if x is not None})
        st = self.static.get(mmsi)
        if st:
            for k in ("name", "type_code", "destination", "callsign", "imo"):
                if st.get(k) is not None and not v.get(k):
                    v[k] = st[k]
        v["category"] = ship_category(v.get("type_code"))
        self.vessels[mmsi] = v

    # ---------------- aisstream.io ----------------
    async def _aisstream_loop(self) -> None:
        p = self.providers["aisstream"]
        backoff = 5
        while True:
            try:
                p.update(state="connecting", error=None)
                async with ws_connect(AISSTREAM_URL, open_timeout=15, ping_interval=30, max_size=2 ** 22) as ws:
                    await ws.send(json.dumps({
                        "APIKey": self.key,
                        "BoundingBoxes": list(AIS_BOXES.values()),
                        "FilterMessageTypes": ["PositionReport", "ShipStaticData", "StandardClassBPositionReport"],
                    }))
                    p["state"] = "connected"
                    backoff = 5
                    async for raw in ws:
                        self._handle_aisstream(raw)
                        p["messages"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                p.update(state="reconnecting", error=f"{type(exc).__name__}: {str(exc)[:120]}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _handle_aisstream(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if "error" in msg:
            self.providers["aisstream"]["error"] = str(msg["error"])[:160]
            return
        meta = msg.get("MetaData") or {}
        mmsi = meta.get("MMSI")
        if not mmsi:
            return
        kind = msg.get("MessageType")
        body = (msg.get("Message") or {}).get(kind) or {}
        name = (meta.get("ShipName") or "").strip() or None
        if kind == "ShipStaticData":
            dest = (body.get("Destination") or "").strip() or None
            self.static[mmsi] = {"name": name or (body.get("Name") or "").strip() or None,
                                 "type_code": body.get("Type"), "destination": dest,
                                 "callsign": (body.get("CallSign") or "").strip() or None,
                                 "imo": body.get("ImoNumber") or None}
            if mmsi in self.vessels:
                self._upsert(mmsi, **self.static[mmsi])
            return
        lat, lng = meta.get("latitude"), meta.get("longitude")
        if lat is None or lng is None or abs(lat) > 90 or abs(lng) > 180:
            return
        hdg = body.get("TrueHeading")
        self._upsert(mmsi, lat=round(lat, 5), lng=round(lng, 5), name=name,
                     sog=body.get("Sog"), cog=body.get("Cog"),
                     heading=hdg if hdg is not None and hdg != 511 else None,
                     nav_status=NAV_STATUS.get(body.get("NavigationalStatus")),
                     ts=time.time(), source="aisstream")

    # ---------------- Digitraffic ----------------
    def _dt_fetch(self, need_meta: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        r = requests.get(DIGITRAFFIC_LOC, headers=DT_HEADERS, timeout=20)
        r.raise_for_status()
        out["loc"] = r.json().get("features", [])
        if need_meta:
            m = requests.get(DIGITRAFFIC_META, headers=DT_HEADERS, timeout=25)
            m.raise_for_status()
            out["meta"] = m.json()
        return out

    async def _digitraffic_loop(self) -> None:
        p = self.providers["digitraffic"]
        last_meta = 0.0
        while True:
            try:
                p["state"] = "polling"
                need_meta = time.time() - last_meta > 900
                data = await asyncio.to_thread(self._dt_fetch, need_meta)
                if "meta" in data:
                    for v in data["meta"]:
                        if v.get("mmsi"):
                            self.static.setdefault(v["mmsi"], {}).update({
                                "name": (v.get("name") or "").strip() or None,
                                "type_code": v.get("shipType"),
                                "destination": (v.get("destination") or "").strip() or None,
                                "callsign": v.get("callSign"), "imo": v.get("imo") or None})
                    last_meta = time.time()
                now = time.time()
                for f in data["loc"]:
                    pr = f.get("properties") or {}
                    lng, lat = (f.get("geometry") or {}).get("coordinates", [None, None])[:2]
                    ts = (pr.get("timestampExternal") or now * 1000) / 1000
                    if lat is None or now - ts > MAX_AGE_S:
                        continue
                    hdg = pr.get("heading")
                    self._upsert(pr.get("mmsi") or f.get("mmsi"), lat=round(lat, 5), lng=round(lng, 5),
                                 sog=pr.get("sog"), cog=pr.get("cog"),
                                 heading=hdg if hdg is not None and hdg != 511 else None,
                                 nav_status=NAV_STATUS.get(pr.get("navStat")), ts=ts, source="digitraffic")
                p["messages"] += len(data["loc"])
                p.update(state="polling", error=None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                p.update(state="error", error=f"{type(exc).__name__}: {str(exc)[:120]}")
            await asyncio.sleep(60)

    # ---------------- queries ----------------
    def snapshot(self, bbox: Optional[List[float]] = None, limit: int = 4000) -> Dict[str, Any]:
        self._prune()
        now = time.time()
        rows = list(self.vessels.values())
        if bbox:
            s, w, n, e = bbox
            if w <= e:
                rows = [v for v in rows if s <= v.get("lat", 99) <= n and w <= v.get("lng", 999) <= e]
            else:  # bbox crosses the antimeridian
                rows = [v for v in rows if s <= v.get("lat", 99) <= n and (v.get("lng", 999) >= w or v.get("lng", 999) <= e)]
        rows = [v for v in rows if "lat" in v]
        total_in_view = len(rows)
        rows.sort(key=lambda v: -v["ts"])
        rows = rows[:limit]
        cats: Dict[str, int] = {}
        for v in self.vessels.values():
            cats[v.get("category", "unknown")] = cats.get(v.get("category", "unknown"), 0) + 1
        return {
            "status": "success",
            "running": self.running,
            "providers": self.providers,
            "tracked_total": len(self.vessels),
            "in_view": total_in_view,
            "returned": len(rows),
            "by_category": cats,
            "vessels": [{
                "mmsi": v["mmsi"], "name": v.get("name"), "lat": v["lat"], "lng": v["lng"],
                "sog": v.get("sog"), "cog": v.get("cog"), "heading": v.get("heading"),
                "category": v.get("category", "unknown"), "destination": v.get("destination"),
                "nav_status": v.get("nav_status"), "source": v.get("source"),
                "age_s": int(now - v["ts"]),
            } for v in rows],
        }

    def zone_counts(self, zones: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Live vessels within radius_km of each zone centre, by category."""
        out = {}
        for zid, z in zones.items():
            inside = [v for v in self.vessels.values() if "lat" in v
                      and abs(v["lat"] - z["lat"]) < 6 and _km(v["lat"], v["lng"], z["lat"], z["lng"]) <= z["radius_km"]]
            cats: Dict[str, int] = {}
            for v in inside:
                cats[v["category"]] = cats.get(v["category"], 0) + 1
            moving = sum(1 for v in inside if (v.get("sog") or 0) >= 3)
            out[zid] = {"vessels": len(inside), "under_way": moving, "by_category": cats}
        return out
