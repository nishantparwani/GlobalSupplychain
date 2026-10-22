# Global Chain AI: Research Brief

**Identifying global maritime chokepoints and analysing their disruption risk with AI**

Prepared 22 October 2026. This brief covers the problem, the situation at the five passages as of October 2026, how the problem maps onto AI techniques, the design decisions behind the system, and how its outputs were checked against real-world reports.

---

## 1. Problem statement

Most of the world's seaborne trade, and more than 90% of seaborne oil, passes through a handful of narrow sea passages. When one of them is closed or becomes dangerous (war, piracy, drought, a grounded ship), freight rates, insurance, oil prices and delivery times change within days, and the effects spread to goods far from the passage.

The project builds an **AI agent** that answers four questions continuously:

1. **Which chokepoints are disrupted right now, and how sure are we?** The agent reads news, ship-traffic and market data and fuses them into a probability per passage.
2. **Why?** Every rating must be explainable: which evidence pushed it up or down.
3. **What should a shipper do?** It recommends the safest and the fastest sea route between two ports, and shows the trade-off.
4. **What else is affected?** It identifies the goods most exposed to the disrupted passages.

## 2. Why these five chokepoints

| Chokepoint | Oil flow (EIA, 2023 data) | Narrowest width | Normal transits | Main role |
|---|---|---|---|---|
| Strait of Malacca | 23.7 million bbl/day | 2.7 km (Phillips Channel) | ~230 ships/day | Largest oil chokepoint; Indian Ocean ↔ Pacific |
| Strait of Hormuz | 20.9 million bbl/day | 33 km | ~85 ships/day | Only sea exit from the Persian Gulf; ~20% of world oil, ~20% of LNG |
| Suez Canal (+ SUMED) | 8.8 million bbl/day | ~300 m | ~70 ships/day | Shortest Asia–Europe route, ~12% of world trade |
| Bab-el-Mandeb | 8.6 million bbl/day | 29 km | ~70 ships/day | Southern gate of the Red Sea; all Suez–Asia traffic |
| Panama Canal | 2.1 million bbl/day | locks (~50 m) | ~36 ships/day | Atlantic ↔ Pacific; capacity set by Gatun Lake water |

The EIA estimates that about 76% of oil supply (77.5 million bbl/day) moved by sea in 2023 [1]. Secondary reports of first-half-2025 data give similar figures (Malacca 23.2, Hormuz 20.9, Suez 4.9, Bab-el-Mandeb 4.2, Panama 2.3 million bbl/day). They show the Red Sea routes far below their 2023 level because of diversions around the Cape of Good Hope (9.1 million bbl/day) [2].

## 3. Situation as of 21 October 2026

The curated baseline briefing in `Nishant/core.py` (`OFFLINE_THREAT_BRIEFINGS`) was rewritten to reflect this. The previous baseline was out of date: it still described Hormuz as "harassment risk" at 6/10.

| Chokepoint | Situation | Baseline rating |
|---|---|---|
| **Hormuz** | Effectively closed to commercial shipping since the regional war began at the start of March 2026. PortWatch-based trackers report single-digit daily transits against a pre-crisis norm of about 85. Attacks on merchant ships continue, and the IMO estimates about 6,000 seafarers are stranded [3][4]. Some trackers report brief rebounds under naval coordination [5]. | 10 / 10 |
| **Bab-el-Mandeb** | Container lines are returning: box-ship transits are above 50% of pre-war levels. But the Houthis have strengthened their position on Yemen's Red Sea coast, and war-risk premiums are reported around 3% [6]. | 7 / 10 |
| **Suez** | September traffic was the highest since December 2023, but liner capacity is still about half of pre-crisis levels (~753,000 TEU in early October versus ~1.5m). A full return is expected by Q1 2027 [6][7]. | 5 / 10 |
| **Panama** | May–August watershed rainfall was 34% below average. Daily transits were capped at 34 from 4 Sept and 32 from 15 Sept, and the draft was cut to 47.5 ft. El Niño may extend restrictions into 2027 [8]. | 6 / 10 |
| **Malacca** | Flowing normally and absorbing diverted volumes. Sea robbery continues in the eastbound lane of the Singapore Strait [9]. | 4 / 10 |

Source quality: figures for Hormuz come from several independent trackers that disagree on exact daily counts, so they are treated as approximate. Bab-el-Mandeb and Suez figures come from BIMCO and Clarksons data as reported by trade press.

## 4. The problem as an AI task

### 4.1 Agent design (PEAS)

| | |
|---|---|
| **Performance** | Minimise transit time subject to geopolitical risk exposure; give calibrated disruption probabilities with explanations |
| **Environment** | Global sea-lane graph (16 nodes, 54 directed edges). Partially observable (news is noisy, AIS is delayed ~4 days and can be spoofed), dynamic, stochastic, continuous |
| **Actuators** | Route recommendations, risk map, news ratings and recommended actions, Copilot answers |
| **Sensors** | 12 news feeds, IMF PortWatch daily AIS transit counts, 13 market series (Yahoo Finance), live AIS positions |

It is a **model-based, utility-driven agent**: it keeps an internal state (threat events, fused risk, traffic models) that is updated on each sensing round, and chooses routes by minimising a cost that trades time against risk.

### 4.2 Technique choice

| Sub-problem | Technique | Why this one |
|---|---|---|
| Rate a news story's threat to a passage | **LLM (Gemini) with a strict JSON schema** | Handles unseen phrasing and context ("mine strike", "Houthi drone"). The schema forces a passage, event type, severity and action that downstream code can use. |
| Rate news when the LLM is unavailable; explain ratings | **Forward-chaining rule-based expert system** (16 rules) | Deterministic, explainable (shows the rules fired), works offline. It acts as both fallback and audit. |
| Know that 17 articles are one event | **TF-IDF + cosine similarity clustering** | Counting corroborating outlets is strong evidence. TF-IDF is transparent and needs no training data. The same index serves as the Copilot's retrieval memory (RAG). |
| Detect a real change in ship traffic | **Change-point detection + robust z-score (median/MAD)** | Traffic series are noisy and heavy-tailed. Median/MAD resists outliers, and the change point stops a long closure from becoming "normal" (section 5.1). |
| Project traffic forward | **Holt linear exponential smoothing** | Small, interpretable (level + trend) and fitted per passage. The 80% band shows uncertainty. |
| Combine heterogeneous evidence into one probability | **Naive-Bayes log-odds fusion** | Each source adds a log-likelihood ratio, so the contribution of each piece of evidence is visible. It also degrades gracefully when a sensor is missing. |
| Recommend a route | **A\* search** | Optimal for the given cost, with an admissible great-circle heuristic. Using two weights (λ = 1 safest, λ = 0 fastest) makes the risk/time trade-off explicit. |
| Group goods by risk | **K-means (k-means++), silhouette score** | Unsupervised, so it needs no labels, and two features (price volatility, supply strain) are easy to show on a plot. |
| Chance a good's supply is hit | **Noisy-OR** | Standard model for "any of several independent causes": 1 − Π(1 − wᵢ·Pᵢ). |
| Answer free-form questions | **Tool-using LLM agent** | The LLM plans calls to the platform's own tools and may only quote returned numbers, which limits hallucination. |

## 5. Design decisions and findings

### 5.1 Baseline contamination in long disruptions (bug found and fixed)

The traffic model compared the last 7 days against a "normal" baseline taken from roughly the past year. Hormuz has been closed for about 7.5 months, so most of that year was closure, and the median "normal" fell to **12 ships/day**. The system reported Hormuz traffic as only −69%, which understated the disruption and weakened the traffic evidence in the fusion.

**Fix:** `Abhik/ai_models.py · detect_change_point()` runs a least-squares single split on the 7-day mean. It keeps the split k that minimises the summed squared error of two constant-mean segments, and accepts it only if both segments last ≥ 30 days and the level shifts by ≥ 35%. When the change point falls inside the baseline window, the baseline is taken from the year before it. History was extended from 420 to 600 days so that a pre-shift year exists.

**Result on live PortWatch data (22 Oct 2026):**

| | Before | After |
|---|---|---|
| Hormuz "normal" | 12.1 ships/day | **88.0 ships/day** |
| Change vs normal | −69% | **−96%** |
| Anomaly z-score | −0.6 (not unusual) | **−4.1 (unusual)** |
| Detected break | — | **4 March 2026**, 84.9 → 6.5 ships/day, explains 87% of variance |

The detected date matches the start of the war reported by the IEA (28 February) and PortWatch's disruption record (1 March) within a few days [10][3]. The recovered baseline of 88/day matches the ~85/day pre-crisis norm reported by independent trackers [3]. The other four passages show no lasting shift, which is the expected result: none of them is closed.

### 5.2 Risk goes in g(n), not h(n)

The A\* heuristic is great-circle sailing time, a strict lower bound on any path's sailing time. If risk were added to h(n), the heuristic could overestimate and A\* would lose optimality. So risk is part of the edge cost g(n): P(r) = 0.35·(e^(0.55·r) − 1) days. The backend checks every edge at start-up and confirms h is admissible and consistent over all 54 directed edges.

### 5.3 Two engines for news, and corroboration

The LLM rates the top stories, and the rule engine rates the rest and takes over on quota errors. In the 22 Oct run, 473 articles were grouped into 40 stories: 14 were rated by Gemini and 26 by the rule engine. Story-level corroboration (for example "17 outlets") enters the news evidence through c = min(1.4, 0.7 + 0.15 × outlets).

### 5.4 Showing the reasoning

For an AI system used in decisions, the explanation matters as much as the number. The redesigned dashboard shows:
- the **log-odds waterfall** for each passage (prior → each evidence source → posterior),
- the **change point** and baseline on the traffic chart,
- the **A\* expansion order** and cost breakdown,
- the **rules fired** for rule-rated stories,
- the **tools called** by the Copilot for each answer.

## 6. Validation against the real world (22 Oct 2026 live run)

| Passage | Model P(disruption) | Status | Real-world reports | Agreement |
|---|---|---|---|---|
| Hormuz | 100% | Critical | Closed / severely restricted [3][4] | ✓ |
| Bab-el-Mandeb | 94% | Critical | Elevated risk, escorted transits, partial return [6] | ✓ direction; arguably high |
| Suez | 38% | Normal | Recovering, highest since Dec 2023 [7] | ✓ |
| Malacca | 14% | Normal | Flowing normally [9] | ✓ |
| Panama | 17% | Normal | Capacity capped at 32/day by drought [8] | ✗ under-rated |

**Panama is a known blind spot.** Drought restrictions reduce capacity gradually, and they are covered in canal-authority notices rather than in breaking news. In this run the news sweep found only two Panama stories, and AIS transits were only −14%, which is consistent with a 32/day cap against a ~36/day norm. The fusion rates this "normal". A fix would be a capacity sensor (canal authority draft and slot notices, or Gatun Lake level) in addition to news and AIS.

## 7. Limitations and future work

- **Naive independence.** News, traffic and markets are correlated (they all react to the same event), so naive-Bayes fusion is over-confident. Learned weights or a small Bayesian network would calibrate better.
- **AIS quality.** Ships "go dark", and GPS jamming and spoofing are reported in the Gulf [10]. PortWatch data also lags by about 4 days.
- **Single change point.** The detector finds one break. Recovery after a closure (two breaks) would need binary segmentation or PELT.
- **Static graph.** The sea-lane graph has 16 nodes. A finer graph with weather routing would give realistic ETAs.
- **Evaluation.** A historical backtest (Red Sea 2023–24, Panama 2023 drought, Hormuz 2026) scored with Brier score would measure calibration properly.

## 8. References

1. U.S. EIA, *World Oil Transit Chokepoints* (2023 data). https://www.eia.gov/international/content/analysis/special_topics/World_Oil_Transit_Chokepoints/wotc.pdf
2. Anadolu Agency, "Around 70% of global oil demand transported through strategic maritime chokepoints." https://www.aa.com.tr/en/energy/general/around-70-of-global-oil-demand-transported-through-strategic-maritime-chokepoints/55347
3. Straits.live, Strait of Hormuz status (citing IMF PortWatch). https://straits.live/
4. Cyprus Mail, "6,000 seafarers trapped as attacks deepen Strait of Hormuz shipping crisis," 21 Oct 2026. https://cyprus-mail.com/2026/10/21/6000-seafarers-trapped-as-attacks-deepen-strait-of-hormuz-shipping-crisis
5. Hormuz.now, live status; Global Energy Flow, Hormuz tracker. https://hormuz.now/ · https://global-energy-flow.com/hormuz/
6. Seatrade Maritime, "Full Suez return for container carriers expected by Q1 2027." https://www.seatrade-maritime.com/containers/full-suez-return-for-container-carriers-expected-by-q1-2027
7. Kuehne+Nagel, "Suez recovery gathers pace, but sector split points to limits ahead." https://mykn.kuehne-nagel.com/news/article/suez-recovery-gathers-pace-but-sector-split
8. Baird Maritime, "Panama Canal caps daily transits as El Niño threatens water levels." https://www.bairdmaritime.com/amp/story/shipping/panama-canal-caps-daily-transits-as-el-ni%C3%B1o-threatens-water-levels
9. ShipUniverse, "Global Chokepoint Watch." https://www.shipuniverse.com/?p=16939 · ReCAAP ISC alerts: https://www.recaap.org/alerts
10. IEA, *Middle East Maritime Chokepoints Shipping Monitor*. https://www.iea.org/data-and-statistics/data-tools/middle-east-maritime-chokepoints-shipping-monitor
11. WTO, *Strait of Hormuz Trade Tracker*. https://datalab.wto.org/Strait-of-Hormuz-Trade-Tracker
12. Miller, "Artificial Intelligence in Maritime Transportation: A Comprehensive Review of Safety and Risk Management Applications," *Applied Sciences* 14, 8420 (2024). https://ai.usz.edu.pl/wp-content/uploads/2025/02/applsci-14-08420.pdf
