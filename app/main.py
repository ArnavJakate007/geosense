"""GeoSense API.

The architecture calls for Celery + Redis. Redis is not installed here and RAM
is tight, so jobs run on a bounded in-process thread pool with the same
submit/poll contract; swapping in Celery means replacing `submit` and `status`.
"""
from __future__ import annotations
import base64
import binascii
import math
import os
import re
import threading
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import attribute, report
from .services import AnalysisService
from .schemas import CompareRequest, RunRequest
from .config import CACHE_DIR, CELL, DETAIL_PX, GRID_PX, SOURCES, BHUVAN_WMS

HERE = os.path.dirname(__file__)
app = FastAPI(title="GeoSense", version="0.1")

analysis_service = AnalysisService()

# Two-image comparisons live in memory only: uploads are never written to
# disk, and the last few results stay reachable for their report links.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
KEEP_COMPARES = 40
_compares: OrderedDict[str, tuple] = OrderedDict()
_compares_lock = threading.Lock()


@app.post("/api/run")
def run(req: RunRequest):
    w, s, e, n = req.bbox
    if not (e > w and n > s):
        raise HTTPException(400, "bbox must be [west, south, east, north]")
    span = max(e - w, n - s)
    if span > 0.45:
        raise HTTPException(400, f"AOI too large ({span:.2f} deg). Keep it under "
                                 "0.45 deg so cells stay meaningful and reads stay fast.")
    if span < 0.01:
        raise HTTPException(400, "AOI too small - draw at least ~1 km across.")
    return {"job": analysis_service.submit(req)}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = analysis_service.job_status(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return JSONResponse(j)


@app.get("/api/semantic")
def semantic_status():
    from . import semantic
    return semantic.probe()


@app.get("/api/bhoonidhi")
def bhoonidhi_status():
    from . import bhoonidhi
    return bhoonidhi.status()


@app.get("/api/similar")
def similar(text: str | None = None, det_id: str | None = None, limit: int = 10):
    """Semantic retrieval across every change this system has indexed.

    `text` embeds the phrase; `det_id` uses an existing detection as the query
    ("more like this one"). Ranking happens in DuckDB via
    array_cosine_similarity, so the vectors are searched where they live
    instead of being dragged into Python first.
    """
    from . import semantic

    store = analysis_service.storage
    if det_id:
        q = store.vector_by_id(det_id)
        if q is None:
            raise HTTPException(404, f"no embedding stored for {det_id}")
    elif text:
        q = semantic.encode_text(text)
        if q is None:
            raise HTTPException(503, "semantic model unavailable - install "
                                     "torch/open_clip or query by det_id")
    else:
        raise HTTPException(400, "pass ?text= or ?det_id=")

    results = store.search_vectors(q, limit=limit, exclude=det_id)
    return {"query": text or det_id, "engine": "duckdb array_cosine_similarity",
            "indexed": analysis_service.storage.stats().get("embeddings", 0),
            "results": results,
            **({"note": "no embeddings indexed yet - run once with "
                        "'Semantic re-rank' ticked"} if not results else {})}


@app.get("/api/runs")
def runs():
    return analysis_service.recent_runs()


@app.get("/api/sources")
def sources():
    return {"sources": [{"key": k, "label": v["label"], "gsd": v.get("gsd"),
                         "sar": bool(v.get("sar"))} for k, v in SOURCES.items()],
            "bhuvan_wms": BHUVAN_WMS}


@app.get("/chips/{run_id}/{name}")
def chip(run_id: str, name: str):
    if "/" in run_id or "/" in name or ".." in run_id + name:
        raise HTTPException(400, "bad path")
    p = os.path.join(CACHE_DIR, run_id, name)
    if not os.path.exists(p):
        raise HTTPException(404, "no such chip")
    return FileResponse(p, media_type="image/png")


@app.get("/api/runs/{run_id}/attribution")
def run_attribution(run_id: str, sensitivity: float = 1.0, min_region_px: int = 80):
    """Attribute every before/after chip a past run saved: each detection gets a cause.

    Works offline on data/cache/<run>. Chip pixel size is recovered from the
    run's bbox and the chip's crop on the analysis grid.
    """
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(400, "bad path")
    if not (0.5 <= sensitivity <= 2.0 and 10 <= min_region_px <= 5000):
        raise HTTPException(400, "sensitivity must be 0.5-2.0 and min_region_px 10-5000")
    folder = os.path.join(CACHE_DIR, run_id)
    if not os.path.isdir(folder):
        raise HTTPException(404, "no cached chips for this run")
    bbox = analysis_service.storage.run_bbox(run_id)
    if not bbox:
        raise HTTPException(404, "run not in the catalogue, so its pixel size is unknown")
    chips = sorted(f[:5] for f in os.listdir(folder) if re.fullmatch(r"c\d{4}_before\.png", f))
    out = []
    for cid in chips:
        with open(os.path.join(folder, f"{cid}_before.png"), "rb") as fh:
            before = fh.read()
        with open(os.path.join(folder, f"{cid}_after.png"), "rb") as fh:
            after = fh.read()
        chip_side = attribute.read_image(before)[0].shape[1]     # 320 in older runs, 640 now
        pixel_m, native_px = _chip_geometry(bbox, int(cid[1:3]), int(cid[3:5]), chip_side)
        # Chips are upsampled to 640 px for reviewers. Shape rules on resampled
        # outlines would measure interpolation, so analyse at the native review
        # resolution (DETAIL_PX); hectares stay exact via the downscale factor.
        result = attribute.attribute(
            before, after, pixel_m, names=(f"{cid}_before.png", f"{cid}_after.png"),
            sensitivity=sensitivity, min_region_px=min_region_px,
            rules={**attribute.RULES, "max_side_px": native_px},
            source={"title": f"Run {run_id}, detection chip {cid}", "simulated": False,
                    "pixel_note": f"from the run's bbox; analysed at the {DETAIL_PX} px review "
                                  "resolution"})
        urls = _keep(result)
        top = next((r for r in result.regions if r["cause"] != attribute.IGNORED), None)
        out.append({"chip": cid, "id": result.id,
                    "pixel_m": round(pixel_m * result.settings["downscale"], 2),
                    "summary": result.summary, "urls": urls,
                    "top": {k: top[k] for k in ("type", "cause", "confidence", "area_ha")} if top else None})
    return {"run_id": run_id, "chips": out}


def _chip_geometry(bbox, j: int, i: int, size: int) -> tuple[float, int]:
    """(ground size of one chip pixel, chip side at the native review resolution).

    Chips are a cell plus half a cell of margin on the GRID_PX analysis grid,
    cut from the DETAIL_PX review read and rendered to size x size (pipeline
    step 7).
    """
    pad = CELL // 2
    span_y = min(j * CELL + CELL + pad, GRID_PX) - max(j * CELL - pad, 0)
    span_x = min(i * CELL + CELL + pad, GRID_PX) - max(i * CELL - pad, 0)
    w, s, e, n = bbox
    m_lon = 111_320.0 * math.cos(math.radians((s + n) / 2))
    dx = (e - w) / GRID_PX * m_lon * span_x / size
    dy = (n - s) / GRID_PX * 111_132.0 * span_y / size
    native_px = max(span_x, span_y) * DETAIL_PX // GRID_PX
    return math.sqrt(dx * dy), int(native_px)


# ------------------------------------------------ two-image compare -----

@app.post("/api/compare")
def compare(req: CompareRequest):
    """Attribute every change between two images: type, likely cause, reasons."""
    if req.sample:
        try:
            before, after, source = attribute.load_sample(req.sample)
        except (ValueError, OSError):
            raise HTTPException(404, f"unknown sample '{req.sample}'")
        names = (f"{req.sample}/before.png", f"{req.sample}/after.png")
        pixel_m = req.pixel_m or source["pixel_m"]
    else:
        before, after = _decode(req.before_b64, "before"), _decode(req.after_b64, "after")
        names = (req.before_name, req.after_name)
        source = None if req.pixel_m else {"title": "Uploaded pair",
                                           "pixel_note": "not given; assumed Sentinel-2 10 m"}
        pixel_m = req.pixel_m or 10.0
    try:
        result = attribute.attribute(before, after, pixel_m, sensitivity=req.sensitivity,
                                     min_region_px=req.min_region_px, names=names, source=source,
                                     zone_pct=req.zone)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {**result.to_json(), "urls": _keep(result)}


@app.get("/api/compare/samples")
def compare_samples():
    return {"samples": [{**s, "thumbs": {k: f"/api/compare/samples/{s['name']}/{k}.png"
                                         for k in ("before", "after")}}
                        for s in attribute.list_samples()]}


@app.get("/api/compare/samples/{name}/{kind}.png")
def compare_sample_image(name: str, kind: str):
    try:
        return Response(attribute.sample_preview(name, kind), media_type="image/png")
    except (ValueError, OSError):
        raise HTTPException(404, f"no {kind} image for sample '{name}'")


@app.get("/api/compare/{cid}/report.html", response_class=HTMLResponse)
def compare_report_html(cid: str):
    result, overlay = _compare(cid)
    return HTMLResponse(report.build_html(result, overlay))


@app.get("/api/compare/{cid}/report.md")
def compare_report_md(cid: str):
    result, _ = _compare(cid)
    return Response(report.build_markdown(result), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="geosense-{cid}.md"'})


@app.get("/api/compare/{cid}/{kind}.png")
def compare_image(cid: str, kind: str):
    result, overlay = _compare(cid)
    if kind == "overlay":
        return Response(overlay, media_type="image/png")
    if kind == "labels":
        return Response(attribute.labels_png(result), media_type="image/png")
    if kind in ("before", "after"):
        return Response(attribute.png_bytes(getattr(result, kind)), media_type="image/png")
    raise HTTPException(404, "no such image")


def _decode(b64: str, label: str) -> bytes:
    """Base64 (optionally a data: URL) to bytes, with a size limit."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    try:
        data = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(400, f"{label} image is not valid base64")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} image is over 20 MB")
    return data


def _keep(result) -> dict:
    """Remember a result (and its overlay) and return the URLs that serve it."""
    overlay = attribute.overlay_png(result)
    with _compares_lock:
        _compares[result.id] = (result, overlay)
        while len(_compares) > KEEP_COMPARES:
            _compares.popitem(last=False)
    base = f"/api/compare/{result.id}"
    return {"before": f"{base}/before.png", "after": f"{base}/after.png",
            "overlay": f"{base}/overlay.png", "labels": f"{base}/labels.png",
            "report_html": f"{base}/report.html", "report_md": f"{base}/report.md"}


def _compare(cid: str):
    with _compares_lock:
        hit = _compares.get(cid)
    if not hit:
        raise HTTPException(404, "unknown or expired comparison - run it again")
    return hit


@app.get("/compare")
def compare_page():
    return FileResponse(os.path.join(HERE, "static", "compare.html"))


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "app.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")
