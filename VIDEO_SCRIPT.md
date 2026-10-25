# GeoSense — Track 05 prototype video (≈ 2 min 45 s)

Three scenarios, the evidence chain, the report. Every number below is what
the build prints today (checked against the live server). If you change
`attribute.RULES` or re-fetch samples, re-check the numbers in **bold**.

---

## Before you hit record

1. Start the server (Windows):
   `cd C:\ORCACODER\geosense_sih_winner` then
   `.venv\Scripts\python -m uvicorn app.main:app --port 8008`
2. Chrome, one window, 1920×1080, **zoom 80 %** (Ctrl + minus twice) so a full
   card fits on screen. Hide the bookmarks bar (Ctrl+Shift+B). Close other tabs.
3. Open `http://127.0.0.1:8008/compare` and run each scenario once (warms the
   server, loads the fonts). Then reload the page.
4. Recorder: OBS, or Win + Alt + R (Xbox Game Bar records the browser window).
   Record the voice separately if the room is noisy, and lay it over afterwards.
5. Move the mouse slowly. Pause half a second before every click.
6. Backup take: `/compare?scenario=jewar_airport&autorun` loads a scenario and
   runs it; `/compare?autorun=tamper` runs the simulated pair and the tamper demo.
   If the server ever dies, open `app/static/compare.html` straight from disk —
   it switches to Sample mode and says so on screen. Do not record that as real data.

---

## Shot list

### 0:00 – 0:12 · Hook
**Screen:** the hero — "SPOT THE CHANGE. NAME THE CAUSE."
**Voice:**
> Satellites photograph every corner of India every few days. Spotting that
> something changed is easy. Saying *who* changed it — a bulldozer or a
> monsoon — is the hard part. That's what GeoSense does.

### 0:12 – 0:25 · What it is
**Screen:** slowly move the mouse over the tool card: scenario cards, pixel
size, sensitivity, permitted zone.
**Voice:**
> Two satellite images in. Out comes every changed patch, whether it's likely
> human-made or natural, the evidence behind each call, and a tamper-evident
> report. No trained model, no black box — every label explains itself.

### 0:25 – 1:05 · Scenario 1 — Mining-lease breach (simulated)
**Screen:**
1. Card **1 · Mining-lease breach** is selected. Point at the yellow
   **SIMULATED** badge, then at "What to look for".
2. Point at the permitted-zone rectangle on the thumbnail.
3. Click **COMPARE IMAGES** — the page scrolls to the verdict.
4. Drag the slider **all the way left** to show the labelled after image, then
   back to the middle.
5. Click the **red patch #1**. Its card opens: point at *Change is real*, the
   *Who caused it* meter, the evidence lines, the striped **outside the
   permitted zone** banner.

**Voice:**
> Our first scenario is simulated, and labelled that way: a real satellite
> image with three changes painted in, so you can see the whole idea at once.
> The yellow box is the mining lease.
>
> Red means likely human-made, blue likely natural. Region one: a
> **63-hectare** cleared pit. Straight edges, right angles, and a freshly cut
> track leading to it — score **100**, strongly human-made. And it's
> **outside the lease**, so it's flagged and ranked first.
>
> The **182-hectare** flood beside the river is the opposite: irregular edges
> that point every way, new water — **strongly natural**.

### 1:05 – 1:35 · Scenario 2 — Airport built on farmland (real Sentinel-2)
**Screen:**
1. Scroll up, click card **2 · Airport built on farmland** (badge
   **REAL SATELLITE**). Point at the dates **2018-01-30 → 2025-01-28**.
2. **COMPARE IMAGES.** Drag the slider slowly left to right across the airport:
   wheat fields become runways.
3. Click the big red region **#1**. Point at "Complex outline… but made of
   straight lines" and "Edges run in two directions at right angles +20".
4. Point at the small red strip at the top (**#4**, access track, flagged
   outside the zone).

**Voice:**
> Now real Sentinel-2 data: Noida International Airport. January 2018, wheat
> fields. January 2025, a four-kilometre runway.
>
> The airport is a complex shape — and a naive rule would call complex
> "natural". GeoSense also measures *straightness*: these edges run in two
> directions at right angles, like every runway, road and plot. So
> **350 hectares** reads as human-made.
>
> We drew the airport site as the permitted zone, and a new road outside it is
> picked up as an access track and flagged.

### 1:35 – 2:00 · Scenario 3 — Wayanad landslide (real Sentinel-2)
**Screen:**
1. Click card **3 · Wayanad landslide**, **COMPARE IMAGES**.
2. Drag the slider across the valley: the debris path appears.
3. Click blue region **#1**. Point at **16.8σ** on *Change is real*, the meter at
   the natural end, "Irregular boundary" and "Edges point every way".
4. Click one amber region and point at the **To settle it** box.

**Voice:**
> Same forest loss, different cause. On the 30th of July 2024, debris flows
> tore down this valley in Wayanad. **83 hectares**, an outline that follows
> the river and points every way — **strongly natural**. The change signal is
> **16.8 sigma** above the image's own noise, so it's certainly real.
>
> And where the evidence is balanced, GeoSense says so, and tells you what
> would settle it — instead of guessing.

### 2:00 – 2:25 · Evidence chain (the trust moment)
**Screen:** scroll to **03 — Evidence chain**.
1. Click **VERIFY CHAIN**: the five blocks turn green one by one.
2. Click **TAMPER WITH ONE VALUE**: block 4 shows region 1 flipped to
   "likely natural", turns red — **CONTENTS CHANGED** — and block 5 shows
   **LINK BROKEN**.
3. Click **RESET**.

**Voice:**
> Every step of the analysis is a block, chained with SHA-256 — the same idea
> as a blockchain, recomputed right here in the browser. Watch what happens if
> someone quietly changes one label… the chain breaks from that block onward.
> Nobody can edit a finding after the fact without it showing.

### 2:25 – 2:40 · Report
**Screen:** scroll to **04 — Evidence report**, click **Open printable report**,
scroll the report once (summary, overlay, region table, provenance), then
Ctrl + P to show it saves as PDF. Cancel.
**Voice:**
> One click gives an officer an evidence report — hectares, reasons, the input
> file fingerprints and the chain — ready to print or attach to a complaint.

### 2:40 – 2:50 · Close
**Screen:** back to the hero.
**Voice:**
> Labels are "likely" — GeoSense is for triage, deciding where to look first.
> Spot the change. Name the cause. Thank you.

---

## If a judge asks

| Question | Answer |
|---|---|
| Is this real data? | Scenarios 2 and 3 are real Sentinel-2 L2A scenes from the public archive, five bands each; the scene IDs are in `data/samples/*/meta.json`. Scenario 1 is labelled simulated. |
| Why not deep learning? | No labelled training data in 3 hours, and every decision here can be explained in one sentence. A model can replace the scoring step later. |
| What does 16.8σ mean? | The region's colour change is 16.8 times the spread of this image's normal pixel-to-pixel variation. We only report from 3σ. |
| Why are some regions amber? | Too small for a shape verdict, or colour alone can't separate causes. The card says what would decide it. |
| Are the zones real lease boundaries? | No — illustrative boxes. The officer draws the real one. |
| Does the hash prove the image is genuine? | No. It proves nothing was altered after analysis. |
