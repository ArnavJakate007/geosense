# GeoSense — working prototype

Semantic retrieval and multi-temporal change analysis of satellite imagery.
SIH26227 · Team VARCHASVA.

This is a **working prototype**, not a mock-up. You draw an area, describe the
change you are looking for in plain English, and it searches the live public
satellite archives, streams only the pixels it needs, fits a seasonal model per
cell, and returns ranked change detections with calibrated confidence and a
verifiable provenance chain.

`index.html` in this folder is the separate, self-contained **pitch demo** with
scripted data. It is untouched. The prototype is everything under `app/`.

**Credits.** GeoSense was originally built by Harsh Agarwal and Izyan
(Team VARCHASVA, SIH26227; original repository
[harshagrawal-bit/geosense_sih_winner](https://github.com/harshagrawal-bit/geosense_sih_winner)).
The Track 05 change-attribution build described below — the compare page,
attribution engine, reports and real Sentinel-2 samples — was added by
Arnav Jakate.

---

## Track 05 build — what is new

A fast path on top of the existing engine: **upload two images → see what
changed → who likely caused it → download an evidence report.** Open
`http://127.0.0.1:8008/compare` (or the "compare two images" link in the
console header). A pair takes well under a second; nothing touches the network.

| New in this build | What it does |
|---|---|
| `app/attribute.py` | Change mask (Lab colour difference above `segment.threshold`'s median + k·MAD), every region labelled and measured, then typed (flooding, water loss, clearing, vegetation loss, excavation/construction, surface darkening, access track, cloud) and scored **likely human-made / likely natural / uncertain** with a plain-English reason for every rule that fired |
| `app/report.py` | Markdown report whose last line is the SHA-256 of everything above it, plus a printable HTML version (browser → PDF) |
| `app/static/compare.html` | Neo-brutalist single page. **Hero** is the tool: a **scenario picker** (cards with before/after thumbnails, real vs simulated badge, problem-statement categories, *why this one* and *what to look for*, in a recommended demo order) or your own upload, pixel size, sensitivity, and an optional **permitted zone** drawn by dragging on the image. **Viewer**: before/after slider over a client-rendered region map — click a region to jump to its card. **Verdict blocks** (human-made / natural / uncertain / ignored / hectares / outside zone). **Region cards** answer two separate questions: *is the change real?* (signal in noise sigmas; reporting starts at 3σ) and *who caused it?* (a natural ↔ human-made meter with the score, in words), plus evidence tagged *Shape*, *Bands* or *Context* with its points, and *to settle it* for undecided regions. **Evidence chain**: the five provenance blocks, re-hashed in the browser — *Verify chain* checks them, *Tamper with one value* flips region 1's cause and the chain breaks from block 4 onward. The JSON contract the page expects is in the comment at the top of the file |
| `POST /api/compare` (+ optional `zone`), `GET /api/compare/samples` (+ thumbnails), `GET /api/compare/{id}/report.html` · `report.md` · `overlay.png` · `labels.png`, `GET /compare` | Images travel as base64 in JSON, so no `python-multipart` install is needed. `labels.png` encodes the region id per pixel (R + 256·G) for click-to-card |
| `GET /api/runs/{run_id}/attribution` | Attributes every detection chip a past run cached in `data/cache/`, at the ~10 m review resolution: every GeoSense detection can also get a cause |
| `data/samples/` + `make_real_samples.py` | **Real Sentinel-2 event pairs** fetched with GeoSense's own `catalog.search` / `raster.read_window`: the clearest scene before and after each event (cloud, shadow and no-data judged over the area by the SCL band), saved as 5-band GeoTIFFs (red, green, blue, NIR, SWIR16) so the physics checks run, plus true-colour previews. Scene ids and dates are in each `meta.json`. Jewar airport (farmland → airport), Wayanad landslide (natural), Hasdeo coal mine (forest → mine), Mumbai coastal road (sea → land), Sindh floods (natural water) |
| `make_samples.py` | The one **labelled simulated** pair (real Jewar image with a painted clearing, track and flood) |
| `tests/test_attribute.py` | Synthetic rectangle (human), irregular water blob (natural), cloud (ignored), track link, physics-from-bands, report tamper check, API routes |

Everything else is existing GeoSense. Reused, not rewritten:
`segment.threshold` (robust MAD), `classify.check` + `GATES` (physics, when
NIR/SWIR bands are supplied), `raster.indices`, `provenance.Chain` (every step
of a comparison is hash-chained and verified in the report).

**How a cause is decided.** Start at 50. Shape: simple boundary (fractal
dimension `2·ln(0.25·P)/ln(A)` < 1.15) +25, irregular (> 1.20) −25,
rectangular +10, jagged −10. Context: connected to a new thin track +20,
borders water that was already there −10 (not for clearing or excavation:
sand mining happens on riverbeds). Type: new water −25, bright bare surface
+10, greening −10. Physics (bands only): water consistent −15 / contradicted
+10; NDVI down with NDBI/BSI up is reported but scores 0, because a landslide
does exactly that too. ≥ 60 likely human-made, ≤ 40 likely natural, between is
uncertain. All thresholds live in `attribute.RULES`; the ones moved from their
first guess say why.

**Why there is no single "confidence %".** A card answers two questions
separately. *Is the change real?* — the region's colour change in noise
sigmas above this image's typical pixel (median + MAD); nothing below 3σ is
reported, so every region shows at least "clear". *Who caused it?* — the score
on a natural ↔ human-made meter, in words ("evidence balanced", "leans
human-made", "clearly natural"). An undecided cause on a clearly real change
says what would settle it (bands, resolution, a field check) instead of
showing 0 %.

**On stage.** Open `/compare`, keep the simulated pair, press **Compare
images**, drag the slider, click region 1, then go to the evidence chain and
press **Verify chain**, then **Tamper with one value**. `/compare?autorun`
runs the comparison on load; `/compare?autorun=tamper` also runs the tamper
demo. **Crash backup:** if the server is down, or `compare.html` is opened
straight from disk, the page switches to *Sample mode* — a scene drawn in the
browser, a result scripted from it, and a real SHA-256 chain — and says so on
the page. Never present sample-mode imagery as satellite data. Fonts load from
Google Fonts; offline it falls back to system fonts.

**Honest notes for the compare path.**
- The real samples are 5-band GeoTIFFs, so water comes from MNDWI and the
  `classify.GATES` checks run. On RGB-only uploads only shape and context
  evidence work, and the water test is a colour proxy that misses turbid or
  weed-covered water.
- Real pairs are single clear-sky scenes years apart; crop seasons and
  regrowth still show up as change. Small regions (< 1000 px) get no
  fractal-dimension verdict and often stay *uncertain* — on purpose.
- Permitted zones on the real samples are illustrative, not notified lease or
  airport boundaries.
- `simulated_jewar` is a real "before" with a clearing, a track and a flood
  painted into the "after". It is labelled simulated in the picker, the page
  and the report. Never present it as a real event.
- Thresholds were tuned on these pairs, not validated on a labelled benchmark.

```bash
.venv/bin/python make_real_samples.py      # fetch the real Sentinel-2 pairs (needs internet once)
.venv/bin/python make_samples.py           # rebuild the simulated pair
.venv/bin/python -m pytest -q tests/       # needs pytest + httpx for the API tests
```

---

## Run it

```bash
./run.sh                      # http://127.0.0.1:8008
```

First time only:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then: pick a preset AOI (or drag one on the map), pick a query chip, **Run
analysis**. A run over a ~9 km box takes **60–200 s**, almost all of it
spent streaming COG windows over the network.

---

## What it actually does

### 1 · Discover — live archive search
`app/catalog.py` queries STAC APIs per request. Nothing is pre-staged.

| Source | Endpoint | Access |
|---|---|---|
| Copernicus **Sentinel-2 L2A** | Earth Search v1 (AWS `sentinel-cogs`) | anonymous |
| USGS **Landsat Collection 2 L2** | Microsoft Planetary Computer | free, auto-signed |
| Copernicus **Sentinel-1** RTC / GRD | Microsoft Planetary Computer | free, auto-signed |
| NRSC/ISRO **Bhuvan** | `bhuvan-vec2.nrsc.gov.in` WMS | basemap layer in the UI |

`app/raster.py` reads only the AOI window from each COG, at an overview level
chosen by the output shape. A 25-date stack costs a few MB, not a few GB.
Cloud and shadow are masked per pixel from Sentinel-2 SCL and Landsat
`QA_PIXEL`, so a partly clouded scene still contributes its clear pixels.

### 2 · Prioritise — seasonal model per cell
`app/change.py`. The AOI is read into 256×256 px and aggregated into a 16×16
grid of cells. For each cell and each index (NDVI, NDBI, MNDWI, BSI):

1. Fit a harmonic model of the annual cycle (1–2 harmonics) and subtract it,
   leaving residuals with phenology removed. **No linear trend term** — over a
   1–2 year window it is not identifiable and extrapolates catastrophically
   (this was measured, not assumed).
2. Combine indices into one composite in the direction the query implies, so a
   matching change is positive.
3. Box-filter the residual field 3×3. Real change covers adjacent cells;
   per-cell sensor noise does not.
4. Scan every split point and keep the largest two-sample t statistic.

Ranking fuses the signature score, the change statistic and the p-value with
**Reciprocal Rank Fusion**, plus context terms derived from the imagery itself
("near a river" is resolved from MNDWI, not from a gazetteer).

### 3 · Verify — calibrated confidence
Permuting the time order of the residuals destroys any real step while keeping
the noise distribution intact, so the resulting statistics are draws from the
no-change null **for this scene, this cadence, this noise level**. With 160
permutations over 160 cells that is ~25 600 null draws per run.

```
p = (1 + #{null ≥ observed}) / (n_null + 1)
```

Benjamini–Hochberg on those p-values then controls the **false discovery rate**
at the chosen α. A query naming a direction ("construction") is tested
**one-sided** — otherwise a change of the opposite kind would certify against it.

So "certified at α = 0.10" means: among the cells flagged, at most 10 % are
expected to be false, under a null calibrated from this scene's own noise.

### Provenance
`app/provenance.py` hash-chains every step (query → discover → ingest → model →
certify → report) with the STAC IDs of the scenes actually used. Altering any
recorded input invalidates every later link. Verification needs nothing but the
chain — no server, no network. Everything persists to DuckDB (`app/store.py`).

---

## Measured extent

A detection used to be a 550 m cell with no size, which is not actionable. Each
flagged cell is now segmented at full image resolution (`app/segment.py`) and
reported in hectares, with a tight bounding box and an overlay chip:

```
cell[4, 8]   z=+2.83    5.31 ha (57 px @ 30.5 m)   break 2025-03-09
cell[3, 2]   z=+2.12    2.61 ha (28 px @ 30.5 m)   break 2024-10-30
```

The cell grid exists because the statistics need a time series per unit, not
because 550 m is the resolution of the answer - the imagery is still ~30 m per
pixel, so the footprint can be recovered once a cell has been flagged.

**This deliberately does not use SAM.** SAM segments *objects* by appearance;
what is needed here is the extent of a *change*, which is defined by the same
composite index difference the detector already tested. Thresholding that
difference (robustly, via median absolute deviation, so the change cannot
inflate its own threshold) segments exactly the thing that was detected, needs
no model or GPU, and cannot disagree with the statistic that raised the alarm.

A cell can legitimately report **no extent**: the cell-level statistic found a
diffuse shift rather than a compact patch. That is information, not a failure.

## Vector index

Runs with "Semantic re-rank" ticked store each candidate's 512-d RemoteCLIP
embedding in the `embeddings` table, which makes retrieval possible across
every change the system has ever indexed - not just within one run:

```bash
curl "localhost:8008/api/similar?text=buildings+on+bare+ground"   # by phrase
curl "localhost:8008/api/similar?det_id=<detection-id>"           # more like this
```

**Be precise about what this is.** `pgvector` is a PostgreSQL extension and
cannot run in DuckDB, so we do not use it and should not claim it.
`PostgreSQLRunRepository` raises `NotImplementedError` - it is an architectural
seam, not working code.

What does run: the vector is a fixed-width `FLOAT[512]` column and ranking
happens **in the database**, via DuckDB's native `array_cosine_similarity`,
rather than by pulling every row into Python. Verified numerically identical to
a NumPy cosine to 1e-6.

The scan is exact, on purpose. DuckDB's `vss` HNSW index installs fine, but
approximate search earns its keep at millions of vectors, not the tens we hold;
at this scale it would add an approximation error and an index build in
exchange for nothing.

## Honest notes

**Read this before demoing.** These are real limitations, not modesty.

- **Semantic search is now two tiers, and the second is optional.** The rule
  parser in `app/query.py` always runs and maps a query to the physical
  signature the change produces (NDVI falls, NDBI rises, …); every rule is
  inspectable and the UI shows which fired. On top of that, ticking "Semantic
  re-rank" loads **RemoteCLIP ViT-B/32** (`app/semantic.py`) and reorders the
  candidate shortlist. It is off by default because it needs ~1.5 GB of RAM on
  a 5 GB machine, and it degrades back to the rule parser on any failure rather
  than refusing to run.
- **The semantic score is a delta, not a similarity.** Asking "does this look
  like construction?" ranks cities that were already there. Each candidate is
  embedded twice and scored `cos(after, q) - cos(before, q)`, so only places
  that *moved toward* the description rank up. CLIP also gets a wider crop
  (80 px) than the evidence panel (32 px): the tight chip that suits a human
  reviewer starves a model trained on whole scenes, and widening it lifted the
  score spread from ±0.004 to -0.007..+0.043.
- **Semantic power is still limited by chip resolution.** At GRID_PX=256 a cell
  is ~24 real pixels. The next real improvement is reading the AOI at higher
  resolution for the chip pass; that is not done.
- **Substituted infrastructure.** The architecture calls for PostgreSQL 16 +
  PostGIS 3.4 + pgvector and Celery + Redis. This machine has no PostgreSQL, no
  Docker, no Redis and no passwordless sudo, so the prototype uses **DuckDB**
  with the same logical schema and a **bounded thread pool** with the same
  submit/poll contract. Both are one-module swaps.
- **Change detection is COLD-inspired, not COLD.** It uses the same idea —
  harmonic seasonal model, break as unexplained deviation — but it is a
  deseasonalise-then-changepoint scan, not the published CCDC/COLD algorithm.
  Do not claim otherwise.
- **The SAR tier is real but secondary.** Tick "Add Sentinel-1 SAR tier" and it
  fetches Sentinel-1 **RTC** from Planetary Computer, computes the Amplitude
  Dispersion Index before and after the break, runs an independent backscatter
  changepoint, and adds both as evidence streams to the RRF fusion. RTC is
  required, not a preference: GRD is not terrain-corrected, so pixels would not
  align between dates and ADI would be measuring geometry rather than ground.
  Certification still rests on the optical statistic; SAR informs ranking.
- **Statistical power is genuinely limited.** With ~15 usable dates and 256
  simultaneous tests, only large and clean changes clear BH at α = 0.10. A run
  reporting *zero certified detections* is usually telling the truth about the
  data. Candidates below the threshold are still listed with their p-values.
- **Cloud is the binding constraint over India.** Monsoon-season AOIs (Simlipal
  in July, say) can lose most of the stack. The pipeline relaxes its coverage
  threshold in steps and reports which one it used; if it still cannot fit a
  model it says so rather than inventing a result.
- **Basemap tiles need internet.** MapLibre itself is vendored locally, but Esri
  / OSM / Bhuvan tiles are fetched live. True air-gapped operation needs a local
  tile store; nothing else in the pipeline requires the public internet once
  imagery is cached.

## Measured accuracy

`python run_eval.py` injects change of known size into a real 23-date
Sentinel-2 stack over Gurugram and measures detection rate. It calls the same
`change.*` routines the pipeline calls, so it measures the shipped detector.

| Injected ΔNDVI | Recall | Precision | Empirical FDR |
|---|---|---|---|
| none (8 repeats) | — | — | **0 cells flagged** |
| 0.06 | 27% | 100% | 0% |
| 0.08 | 80% | 100% | 0% |
| 0.12 | 100% | 100% | 0% |

Minimum detectable change **ΔNDVI ≈ 0.08 at 80% power**; empirical FDR 0%
against a nominal α of 10%. Results in `data/evaluation.json`.

Public bi-temporal benchmarks (OSCD, LEVIR-CD) are not used, and the reason is
not convenience: they supply two dates and a mask, whereas this detector needs
a 15+ date series to fit a seasonal model. Scoring against them would measure a
different algorithm.

## One measurement worth knowing about

Earth Search publishes `raster:bands` with `offset = -0.1` on Sentinel-2 L2A
assets, which is the documented baseline-04.00 DN correction. **Applying it to
this collection is wrong** - these COGs are already harmonised. Measured over
0 %-cloud Western Ghats forest:

| | blue | green | red | NIR | NDVI |
|---|---|---|---|---|---|
| offset applied | **-0.067** | -0.051 | -0.071 | 0.194 | **2.145** |
| no offset | 0.033 | 0.049 | 0.029 | 0.294 | **0.818** |

Negative reflectance is impossible and NDVI is bounded by 1, so the second row
is the physical one. `trust_stac_scale=False` in `app/config.py` records this.
It is the kind of bug that silently corrupts every index downstream while
still producing plausible-looking maps, which is why the pipeline is checked
against known land cover rather than eyeballed.

## Layout

```
app/catalog.py     STAC search across S2 / Landsat / S1
app/raster.py      windowed COG reads, cloud masking, indices, chips
app/change.py      harmonic model, changepoint scan, permutation null, ADI
app/query.py       natural language -> physical change signature (rules)
app/semantic.py    RemoteCLIP tier: delta-similarity re-ranking (optional)
app/evaluate.py    semi-synthetic validation harness
app/rank.py        RRF fusion, context priors, Benjamini-Hochberg
app/provenance.py  hash-chained audit trail
app/store.py       DuckDB catalogue
app/pipeline.py    tip-and-cue orchestration
app/main.py        FastAPI
app/static/app.html  analyst console (MapLibre, vendored)
```

## Storage boundary

Semantic retrieval currently uses the inspectable rule-based parser in
`app/query.py` through `SemanticRetrievalPort`. GeoRSCLIP and GeoLangBind are
not deployed: this environment has no Torch, Transformers, or local model
weights, and no model is downloaded automatically. The API and UI report
`rule_based` with `model_available: false`; no embeddings are fabricated.

The application uses the real DuckDB repository by default. The backend can be
selected without installing PostgreSQL:

```bash
GEOSENSE_STORAGE=duckdb                 # default local repository
GEOSENSE_STORAGE=postgres \
GEOSENSE_POSTGRES_DSN=postgresql://...  # interface skeleton only
```

`app/repositories.py` contains the repository contract and the working
DuckDB implementation. The PostgreSQL/PostGIS class is intentionally not
implemented yet and raises a clear error if selected, so local startup never
requires PostgreSQL.

## Phase 1 architecture boundaries

The live prototype still uses the implementations described above. Phase 1
adds typed contracts in `app/schemas.py`, replacement ports in
`app/contracts.py`, and adapters in `app/adapters.py`. The application path is
now `API → AnalysisService → pipeline → adapters → DuckDB`. The orchestrator
makes `TIP → CUE → CONFIRM` explicit: Confirm evaluates the CUE shortlist using
existing temporal and spectral evidence, with optional SAR evidence recorded
when available. No segmentation or VHR backend is present.

These boundaries do **not** mean that Celery, PostgreSQL, STAC provenance,
GeoRSCLIP, CCDC, or confirmation models are present. They only isolate the
current rule parser, harmonic residual/max-t detector, Sentinel-1 RTC/ADI
evidence, DuckDB store, thread-pool jobs, and hash-chain provenance so they
can be replaced independently later.
