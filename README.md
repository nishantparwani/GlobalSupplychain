# Global Chain AI — Maritime Risk Intelligence & Pathfinding

Global Chain AI is a shipping-risk monitor. It reads live maritime news, works out which strategic choke points (Suez, Hormuz, Bab-el-Mandeb, Panama, Malacca) are under threat, and recommends sea routes that balance sailing time against geopolitical risk. A browser dashboard shows the threat map, route comparisons, commodities at risk, and a chat "Copilot" for plain-English questions.

![Command center](docs/screenshots/v2/01_command_center.png)

## Screenshots

| Chokepoint analysis: evidence and change point | Risk-aware route planner (A\*) |
|---|---|
| ![Chokepoint analysis](docs/screenshots/v2/02_chokepoint_analysis.png) | ![Route planner](docs/screenshots/v2/03_route_planner.png) |
| **Threat intelligence (LLM + rule engine)** | **Markets and live ticker** |
| ![Threat intelligence](docs/screenshots/v2/04_threat_intelligence.png) | ![Markets](docs/screenshots/v2/05_markets.png) |
| **Goods at risk (K-means + noisy-OR)** | **How the AI works (PEAS + techniques)** |
| ![Goods at risk](docs/screenshots/v2/06_goods_at_risk.png) | ![How the AI works](docs/screenshots/v2/07_how_the_ai_works.png) |

<details>
<summary>Light theme</summary>

![Light theme](docs/screenshots/v2/08_light_theme.png)

</details>

The project is built as a **rational agent** and demonstrates several AI techniques in one working system. The dashboard's **How the AI works** page shows each one with its live result. For the problem background, the October 2026 situation and the design rationale, see [Abhik/RESEARCH.md](Abhik/RESEARCH.md).

| Component | Technique | Where |
|---|---|---|
| Sensors | Concurrent sweep of 12 news feeds, IMF PortWatch AIS transits, Yahoo Finance prices | `data_sources.sweep()` |
| Inference (a) | Gemini LLM threat rating with a strict JSON schema | `gemini_assess()` |
| Inference (b) | Forward-chaining rule-based expert system (fallback and explainability) | `rule_engine_assess()` |
| Inference (c) | TF-IDF + cosine story clustering and retrieval (RAG) | `ai_models.cluster_stories()` |
| Inference (d) | Change-point detection, median/MAD anomaly score, Holt forecast on AIS traffic | `ai_models.analyse_traffic()` |
| Inference (e) | Naive-Bayes evidence fusion into P(disruption) per chokepoint | `ai_models.fuse()` |
| Inference (f) | A\* search over a weighted maritime graph | `a_star()` |
| Inference (g) | K-Means clustering of commodity risk vectors; noisy-OR supply exposure | `kmeans()`, `ai_models.noisy_or()` |
| Inference (h) | Tool-using Gemini agent (Copilot) | `agent_answer()` |
| Actuators | Route recommendations, risk map, news ratings, Copilot answers | REST API + dashboard |

The app works **without** a Gemini API key: the rule-based expert system takes over every LLM task, and a curated baseline threat dataset is shown until you run a live ingestion.

---

## Contents

- [Features](#features)
- [Project structure and team split](#project-structure-and-team-split)
- [Team workflow: pushing your part](#team-workflow-pushing-your-part)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Using the dashboard](#using-the-dashboard)
- [How it works](#how-it-works)
- [API reference](#api-reference)
- [Quota and reliability policy](#quota-and-reliability-policy)
- [Troubleshooting](#troubleshooting)
- [Extending the project](#extending-the-project)

---

## Features

- **Live threat intelligence** — pulls articles from UN News, The Maritime Executive, Hellenic Shipping News and gCaptain, filters them for maritime relevance, and classifies each one by choke point, event type and severity (1–10), with a concrete tactical directive and a citation back to the source.
- **Risk-aware routing** — for any two ports, computes both the *safest* route (risk-weighted A\*) and the *fastest* commercial route, and compares ETA, distance and a 0–100 safety index.
- **What-if scenarios** — override choke-point severities (for example "Hormuz closed" or "Red Sea and Suez closed") and see how routes change.
- **Commodities at risk** — clusters ten commodities (crude, LNG, container freight, wheat, copper and more) into risk groups based on volatility and supply-chain stress, coupled to live choke-point severity.
- **Copilot chat** — ask questions such as *"What happens to oil tankers if Hormuz closes?"*. Answers are computed deterministically by the platform and optionally rewritten into fluent prose by Gemini.
- **Graceful degradation** — every Gemini-powered path has a rule-based fallback, and feed failures fall back to the baseline dataset.

---

## Project structure and team split

The code is divided into four folders, one per team member. Each folder is a Python package, and the app runs from the project root with `python run.py`.

```
.
├── run.py                  # entry point: python run.py  ->  http://localhost:8000
├── requirements.txt
├── .env.example            # copy to .env and add your keys (.env is never committed)
│
├── Nishant/                # core agent, knowledge base, search, server
│   ├── core.py             # configuration, Gemini LLM engine, maritime graph, chokepoint profiles, threat state
│   ├── routing.py          # A* search (safest / fastest route), heuristic admissibility check
│   ├── ais_live.py         # live AIS ship positions (aisstream.io WebSocket, Digitraffic polling)
│   └── server.py           # FastAPI REST API, market ticker, start-up
│
├── Yash/                   # sensors and threat intelligence
│   ├── data_sources.py     # news feeds, IMF PortWatch AIS transits, Yahoo Finance markets and ticker
│   ├── intelligence.py     # relevance filter, 16-rule expert system, Gemini threat rating, ingestion pipeline
│   └── clustering.py       # K-means clustering of goods at risk
│
├── Abhik/                  # AI models and the AI analyst
│   ├── ai_models.py        # TF-IDF, change-point detection, anomaly score, Holt forecast, Bayesian fusion, noisy-OR
│   ├── copilot.py          # rule-based analyst + tool-using Gemini agent (Copilot)
│   ├── RESEARCH.md         # research brief
│   └── PROJECT_OVERVIEW.txt
│
└── Harsh/                  # dashboard
    ├── index.html          # single-page dashboard (React 18 + Leaflet + Tailwind from CDNs), 7 pages
    └── screenshots/
```

Dependencies only point one way (`core → routing → clustering → intelligence → copilot → server`), so each part can be read and tested on its own. The server serves `Harsh/index.html` directly; there is no frontend build step.

---

## Team workflow: pushing your part

The restructured project lives on the **`team-split`** branch. `main` still holds the old single-folder version until the branch is merged. Each member pushes only their own folder to `team-split`:

| Member | Pushes |
|---|---|
| Nishant | `Nishant/` + shared root files (`run.py`, `README.md`, `requirements.txt`, `.gitignore`, `.env.example`) ✅ done |
| Yash | `Yash/` |
| Abhik | `Abhik/` |
| Harsh | `Harsh/` |

The app runs once all four folders are on the branch.

```bash
# 1. Get the branch (first time only)
git clone https://github.com/nishantparwani/GlobalSupplychain.git
cd GlobalSupplychain
git checkout team-split

# 2. Copy your folder into the project root, e.g. Yash/, then:
git pull                      # get everyone else's latest work first
git add Yash/                 # only your own folder
git commit -m "Add Yash's part: sensors, threat intelligence, K-means"
git push origin team-split
```

If `git push` is rejected because someone pushed first, run `git pull` and push again. Never commit `.env`; copy `.env.example` to `.env` locally instead.

---

## Requirements

- **Python 3.10+** (developed on Python 3.11)
- Internet access for live news ingestion, the Gemini API, and the CDN-hosted frontend libraries (React, Leaflet, Tailwind, Babel, map tiles)
- *(Optional)* A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- *(Optional)* Node.js, only if you want to use the Tailwind CLI build in `package.json`

Python dependencies (`requirements.txt`):

| Package | Purpose |
|---|---|
| `fastapi` | Web framework and REST API |
| `uvicorn` | ASGI server |
| `requests` | RSS feed fetching |
| `beautifulsoup4` | Stripping HTML from feed summaries |
| `google-generativeai` | Gemini LLM client (optional at runtime) |
| `python-dotenv` | Loading `.env` |

---

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root (copy `.env.example`) (see [Configuration](#configuration)). It is optional: without it the app runs in rule-engine-only mode.

---

## Configuration

`Nishant/core.py` reads `.env` from the project root, regardless of where you launch it from.

```dotenv
# Required for Gemini features (leave unset to run with the rule engine only)
GEMINI_API_KEY=your_api_key_here

# Optional
GEMINI_MODEL=gemini-3.6-flash              # primary model
GEMINI_FALLBACK_MODELS=gemini-2.5-flash    # comma-separated; tried only if the primary model name is not found
GEMINI_MAX_RPM=4                           # local cap on Gemini calls per minute (default 4, under the 5 RPM free tier)
PORT=8000                                  # HTTP port
```

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(none)* | Gemini API key. Missing or placeholder values disable the LLM. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Primary Gemini model. |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.5-flash` | Comma-separated fallback models. |
| `GEMINI_MAX_RPM` | `4` | Maximum Gemini calls per rolling minute. |
| `PORT` | `8000` | Port the server listens on. |

> **Security:** `.env` contains a secret. Keep it out of version control (add it to `.gitignore`) and never share it. If a key is ever exposed, revoke it in Google AI Studio and create a new one.

The `.env` file must contain only `KEY=value` lines. On Windows, writing it with `Set-Content -Encoding ascii` (or any UTF-8 editor) works; make sure the shell command itself does not end up inside the file.

---

## Running the app

```bash
python run.py
```

The console prints a start-up banner:

```
==================================================================
 GLOBAL CHAIN AI - MARITIME RISK INTELLIGENCE BACKEND
 LLM engine : READY (gemini-3.6-flash)
 Gemini cap : 4 calls/minute; ingestion runs only on demand
 Heuristic  : admissible=True over 54 edges
 Dashboard  : http://localhost:8000      API docs: http://localhost:8000/docs
==================================================================
```

Then open:

- **Dashboard:** <http://localhost:8000>
- **Interactive API docs (Swagger):** <http://localhost:8000/docs>

You can also open `Harsh/index.html` directly from disk; it will call the API at `http://localhost:8000` automatically (the backend enables CORS).

To run with auto-reload during development:

```bash
uvicorn backend:app --reload --port 8000
```

---

## Using the dashboard

The dashboard is a single page with a sidebar. A live market ticker runs along the top, and **Run AI analysis** (top right) fetches the latest news, ship traffic and prices. Dark and light themes are available from the sidebar; add `?theme=light` to the URL to open in light mode.

| Page | What it shows |
|---|---|
| **Command center** | Critical alert, oil flow at risk (Σ P(disruption) × normal oil flow), chokepoint map with pulsing risk markers, risk ranking, the four-step AI pipeline, top threats and market pulse. |
| **Chokepoints** | One tab per passage: the Naive-Bayes evidence waterfall (prior → news → AIS traffic → markets → posterior), daily transits with the detected change point and a 14-day Holt forecast, passage profile, exposed goods and latest reports. |
| **Route planner** | Safest vs fastest route on the map, what-if scenarios (Hormuz reopens, Red Sea closed, Panama drought, ...), A\* cost breakdown and the node expansion order. |
| **Threat intelligence** | Every story with severity, passage, outlets that corroborate it, the engine that rated it (LLM or rules, with the rules fired), and sensor health. Filter by passage and severity. |
| **Markets** | Live quotes for freight, shipping lines, energy, FX and global indices, plus 90-day history with a 3-month z-score stress badge. |
| **Goods at risk** | K-means scatter (k = 2 or 3), commodity × chokepoint exposure heatmap and noisy-OR supply-hit probability. |
| **How the AI works** | PEAS agent design, the reasoning pipeline, and a card per AI technique with its live result and code location. |

The **Copilot** chat accepts questions like:

- `Why avoid Suez between Shanghai and Rotterdam?`
- `What happens to oil tankers if Hormuz reopens?`
- `Route Mumbai to London`
- `Which goods are most at risk?`
- `What's happening in the Red Sea?`

---

## How it works

### Maritime network

The world shipping network is modelled as an undirected graph of **16 nodes**:

- **Ports:** Shanghai, Tokyo, Singapore, Mumbai, Dubai (Jebel Ali), Rotterdam, London (Thames), New York, Los Angeles
- **Choke points:** Suez Canal, Strait of Hormuz, Bab-el-Mandeb, Panama Canal, Strait of Malacca
- **Waypoints:** Strait of Gibraltar, Cape of Good Hope

Each of the 27 edges follows a realistic sea-lane polyline. Edge length is the polyline's great-circle length multiplied by a 1.04 routing factor. Every node also has a structural `base_hazard` (0–10).

### Threat ingestion pipeline

1. **Fetch** — all feeds are fetched concurrently (4 s timeout per feed, 9 s wall-clock ceiling for the round). RSS 2.0 and Atom are parsed with the standard library.
2. **Clean** — HTML is stripped, tracking parameters (`utm_*`, `fbclid`, …) are removed from URLs, and dates are normalised to UTC ISO-8601.
3. **Rank** — duplicate titles are removed, and each article is scored against weighted maritime keywords (for example *tanker*, *Red Sea*, *Houthi*, *piracy*). Articles scoring below 2 are discarded; the top 12 go to analysis.
4. **Analyse** — Gemini classifies the batch in one call, returning a JSON array that is validated against the allowed choke points and event types. Rows that fail validation, and the whole batch if Gemini is unavailable, are handled by the rule engine.
5. **Aggregate** — the live severity of each choke point is the maximum severity of the events that target it.

Event types: `MILITARY_CONFLICT`, `PIRACY`, `DROUGHT_WEATHER`, `STRIKE_LABOR`, `GENERAL_ADVISORY`.

### Rule-based expert system

Keyword patterns turn article text into facts (`LOC_HORMUZ`, `KW_KINETIC`, `KW_PIRACY`, …). Sixteen production rules (R01–R16) are then applied by **forward chaining** until no new facts are produced. For example:

- `R07` seizure + Hormuz ⇒ military conflict, high severity
- `R11` military conflict + at choke point ⇒ high severity
- `R12` high severity + disruption ⇒ critical

The resulting facts determine the choke point, event type and severity (critical = 9, high = 7, medium = 5), and a tactical directive is looked up from a directive table. The list of fired rules is returned as `rule_trace` for explainability.

### A\* route search

The search minimises `f(n) = g(n) + h(n)`, where all costs are in **days**:

```
g(n) = Σ edges [ nm / (24 × 14 kn)  +  canal_dwell(v)  +  λ × P(r_v) ]
h(n) = great_circle_nm(n, goal) / (24 × 14 kn)
P(r) = 0.35 × (e^(0.55 × r) − 1)          risk penalty in days
r    = max(base_hazard, live_severity)    in [0, 10]
```

- Ship speed is 14 knots (336 nm/day). Canal dwell is 1.0 day for Suez and 1.5 days for Panama.
- The **safe route** uses λ = 1; the **commercial route** uses λ = 0 (pure time).
- The heuristic is **admissible and consistent**: every edge is at least as long as its great-circle chord, and all other cost terms are non-negative. This is verified exhaustively at start-up and reported by `/api/health`.
- Risk is deliberately placed in `g(n)`, not `h(n)`; putting it in the heuristic would overestimate remaining cost and void A\*'s optimality guarantee.

Each route also gets a **safety index**: `100 × exp(−Σ r_i² / 150)` over every node on the path.

### K-Means commodity clustering

Each commodity is a 2-D vector of **volatility** and **supply-chain stress** (each 0–100). Base values are raised by live severity above 3 at the choke points each commodity is exposed to. The clustering uses:

- Lloyd's algorithm with **k-means++** initialisation and a fixed seed (42) for reproducible results
- Euclidean distance, convergence threshold ε = 0.001, at most 20 iterations
- Inertia (WCSS) and the **silhouette coefficient** as quality metrics

Clusters are labelled by centroid magnitude: `HIGH_SYSTEMIC_RISK`, `ELEVATED_WATCH` (k = 3 only) and `STABLE_NORMAL_FLOW`.

> The commodity indicator values are a modelled snapshot, not live market data.

### Ship-traffic analytics and evidence fusion

Daily transit counts per chokepoint (IMF PortWatch, about 600 days) go through three steps in `ai_models.analyse_traffic()`:

1. **Change-point detection.** A least-squares single split of the 7-day mean finds the point that best divides the series into two constant levels. It is accepted only when both segments last at least 30 days and the level shifts by at least 35%. When the shift falls inside the baseline window, "normal" is taken from the year before it. Without this, a long closure slowly becomes the baseline: Hormuz read −69% against a contaminated "normal" of 12 ships/day, and now reads −96% against the pre-war 88/day, with the break dated 4 March 2026.
2. **Robust anomaly score.** z = (7-day mean − median) / (1.4826 × MAD); |z| ≥ 3 is flagged unusual.
3. **Holt linear exponential smoothing.** α and β are fitted by grid search on one-step-ahead error, giving a 14-day forecast with an 80% band.

`ai_models.fuse()` then combines the structural prior, news, traffic, Cape of Good Hope diversions and market z-scores as log-likelihood ratios. The posterior is P(disruption) = sigmoid(logit(p0) + Σ LLR), shown step by step on the Chokepoints page.

### Copilot

Every question is first answered by a deterministic analyst (`rule_copilot`) that recognises port and choke-point names and aliases (for example "Nhava Sheva" → Mumbai, "Red Sea" → Bab-el-Mandeb) and routes the question to a closure scenario, route explanation, route comparison, commodity brief or threat brief. If Gemini is available, it receives that analysis plus a JSON context and rewrites it as a concise answer, using only numbers present in the context.

---

## API reference

All endpoints return JSON. Interactive documentation is available at `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard (`Harsh/index.html`). |
| `GET` | `/api/health` | Server status, uptime, LLM status, graph size, heuristic check, agent (PEAS) description. |
| `GET` | `/api/graph` | Graph nodes (with live severity), edges, cost model and heuristic verification. |
| `GET` | `/api/routes` | Safe vs. commercial route comparison. |
| `GET` | `/api/clusters` | K-Means commodity clustering. |
| `GET` | `/api/intelligence` | Current threat state. Read-only: never scrapes and never calls Gemini. |
| `POST` | `/api/intelligence/ingest` | Runs one ingestion round (the only endpoint that scrapes feeds). |
| `GET` | `/api/chokepoints` | Status, effective risk, routing penalty and events for each choke point. |
| `GET` | `/api/ticker` | Live quotes for the header marquee and the *Trade finance today* panel: global indices, commodities, shipping, FX and rates (Yahoo Finance, cached 2 minutes). |
| `POST` | `/api/chat` | Copilot question answering. |

### `GET /api/routes`

| Parameter | Default | Description |
|---|---|---|
| `start` | `SHANGHAI` | Origin node ID |
| `end` | `ROTTERDAM` | Destination node ID |
| `override` | *(empty)* | Scenario severities, for example `HORMUZ:10,SUEZ:8` (choke points only, clamped to 0–10) |

Valid node IDs: `SHANGHAI`, `TOKYO`, `SINGAPORE`, `MALACCA`, `MUMBAI`, `DUBAI`, `HORMUZ`, `BAB_EL_MANDEB`, `SUEZ`, `GIBRALTAR`, `ROTTERDAM`, `LONDON`, `NEW_YORK`, `PANAMA`, `LOS_ANGELES`, `CAPE_OF_GOOD_HOPE`.

```bash
curl "http://localhost:8000/api/routes?start=SHANGHAI&end=ROTTERDAM&override=SUEZ:10,BAB_EL_MANDEB:10"
```

The response includes `safe_route`, `commercial_route` (each with `path`, `eta_days`, `nautical_miles`, `safety_index`, `legs`, `geometry` and A\* `search` statistics) and a `comparison` with `delta_eta_days`, `delta_safety`, `hazards_avoided` and a plain-English `verdict`.

### `GET /api/clusters`

| Parameter | Default | Description |
|---|---|---|
| `k` | `2` | Number of clusters (2 or 3) |

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"command": "What happens to oil tankers if Hormuz closes?", "use_llm": true}'
```

```json
{ "response": "### What if Strait of Hormuz closes? ...", "engine": "GEMINI (gemini-3.6-flash)", "note": null }
```

Set `use_llm` to `false` to get the deterministic rule-engine answer only.

### `POST /api/intelligence/ingest`

```bash
curl -X POST http://localhost:8000/api/intelligence/ingest
```

Returns the same payload as `GET /api/intelligence`: `mode` (`LIVE` or `BASELINE`), `engine`, `events`, `articles`, `sources` (per-feed status), `choke_severity` and `llm` status.

---

## Quota and reliability policy

The backend is designed to stay inside the Gemini free tier:

- **Nothing external runs on start-up or page load, apart from the market ticker** (one batch of Yahoo Finance quotes, cached for 2 minutes and shared by all viewers). Feeds are scraped, and Gemini is called for analysis, only when `POST /api/intelligence/ingest` is called (the **Run Intelligence Ingestion** button).
- Ingestion requests closer together than **8 seconds** are ignored, and only one ingestion runs at a time.
- Gemini calls are capped by a local rolling budget (`GEMINI_MAX_RPM`, default 4/min).
- After an HTTP **429**, Gemini is paused for the retry interval the API suggests (60 s by default, or at least 1 hour for a daily-quota error).
- Fallback models are tried only when a model name is not found, never after a quota error.
- Each LLM call times out after 25 s.

LLM status values (shown in `/api/health` and `/api/intelligence`):

| Status | Meaning |
|---|---|
| `READY` | Gemini is available. |
| `THROTTLED` | The local per-minute budget is used up. |
| `COOLDOWN` | Paused after a 429 from the API. |
| `NOT_CONFIGURED` | No valid key, or the `google-generativeai` package is missing. |

In every non-`READY` state the rule engine produces the result and the response includes a note explaining why.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `[WARN] GEMINI_API_KEY not found ... Running in fallback mode.` | `.env` is missing, in the wrong folder, or the key is a placeholder. Put `.env` in the project root with a real `GEMINI_API_KEY=...` line. |
| LLM status stays `COOLDOWN` | The Gemini quota was hit. Wait for the cooldown shown in the dashboard, or lower `GEMINI_MAX_RPM`. |
| Gemini errors mentioning "not found" or "not supported" | The model name isn't available to your key. Set `GEMINI_MODEL` to a model your key can use. |
| Ingestion returns `BASELINE` mode | No feed could be reached, or none had maritime-relevant articles. Check your internet connection and the per-source `error` fields in `sources`. |
| Dashboard is blank | The CDN scripts couldn't load (offline or blocked). Check the browser console. |
| Dashboard shows network errors | The backend isn't running on port 8000. Start it with `python run.py`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, or call `.\.venv\Scripts\python.exe run.py` directly. |
| Port 8000 already in use | Set `PORT=8001` in `.env`. Note that the dashboard opened from disk expects port 8000; when served by the backend it uses the same origin. |

---

## Extending the project

- **Add a news source:** append an entry to `FEEDS` with an `id`, `publisher` and one or more candidate `urls`.
- **Add a port or choke point:** add it to `NODES`, connect it in `EDGE_WAYPOINTS` with a sea-lane polyline, and (for a choke point) add it to `CHOKE_POINTS`, `STRATEGIC_NOTES`, `KEYWORD_FACTS` and the allowed values in the Gemini prompt. The heuristic check at start-up will flag any edge that is shorter than its great-circle distance.
- **Add an expert-system rule:** append a `(name, conditions, conclusions)` tuple to `RULES`, and if needed a directive to `DIRECTIVES`.
- **Tune the risk trade-off:** adjust `RISK_ALPHA`, `RISK_BETA`, `SAFETY_DECAY` and `SHIP_SPEED_KNOTS` at the top of `Nishant/core.py`.
- **Add a commodity:** append it to `COMMODITIES` with base `vol`, `stress` and per-choke-point `exposure` weights.

---

## Disclaimer

This is an educational and research project. Threat classifications, routes and commodity indicators are model outputs and simplified estimates, and must not be used as the sole basis for real navigation, security or trading decisions.
