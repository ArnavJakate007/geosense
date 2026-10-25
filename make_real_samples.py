"""Fetch real Sentinel-2 before/after pairs for the compare page's demo scenarios.

Uses GeoSense's own catalog.search and raster.read_window (anonymous Earth
Search COGs). For each event it takes the clearest scene in a "before" and an
"after" window, judged by cloud, shadow and no-data pixels over the area itself
(Sentinel-2 SCL band), not by the scene-wide cloud figure.

Writes data/samples/<name>/:
  before.tif, after.tif   5 bands (red, green, blue, nir, swir16), reflectance x 10000 as uint16
  before.png, after.png   true colour, fixed 0-0.30 stretch (as raster.rgb_png)
  meta.json               jury-facing text plus the scene ids and dates actually used

Needs internet once; the demo then runs offline from the saved files.
Run:  .venv/bin/python make_real_samples.py [name ...]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rasterio
from PIL import Image

from app import attribute, catalog, raster, segment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "samples")
BANDS = ("red", "green", "blue", "nir", "swir16")
SIZE = 512
BAD_SCL = (0, 3, 8, 9, 10)          # no data, cloud shadow, cloud (medium, high), cirrus

# center (lat, lon), side in km, before window, after window, plus the text the
# scenario picker shows. look_for lines are checked against a real run.
SCENARIOS = {
    "jewar_airport": {
        "center": (28.180, 77.612), "km": 8.0,
        # 2018, not 2020: by early 2020 the acquired land was already bare, so the change
        # would read as bare soil -> construction soil. In 2018 it was winter-wheat fields.
        "before": ("2018-01-01", "2018-03-15"), "after": ("2025-01-01", "2025-04-30"),
        "short_title": "Airport built on farmland", "place": "Jewar, Uttar Pradesh, India",
        "ps": ["Infrastructure", "Land excavation"], "order": 2, "featured": True,
        "why": "Noida International Airport. In early 2018 this was farmland; seven years later "
               "there is a 4 km runway, aprons and a terminal. The airport site is drawn as the "
               "permitted zone, so building inside it is expected and changes outside it are "
               "flagged.",
    },
    "wayanad_landslide": {
        "center": (11.498, 76.140), "km": 6.0,
        "before": ("2024-01-01", "2024-04-15"), "after": ("2024-11-15", "2025-03-31"),
        "short_title": "Wayanad landslide", "place": "Wayanad, Kerala, India",
        "ps": ["Landslide (natural)", "Vegetation loss"], "order": 3, "featured": True,
        "why": "On 30 July 2024, after extreme rain, debris flows tore down the Mundakkai and "
               "Chooralmala valleys. The same forest that machines clear in straight lines was "
               "stripped here along a winding river path: the shape of the scar is the evidence.",
    },
    "hasdeo_mine": {
        "center": (22.842, 82.792), "km": 8.0,
        # winter scenes: March-April brings forest-fire haze that reads as change
        "before": ("2017-01-01", "2017-03-15"), "after": ("2024-11-15", "2025-02-28"),
        "short_title": "Coal mine eats forest", "place": "Hasdeo Arand, Chhattisgarh, India",
        "ps": ["Forest destruction", "Land excavation"], "order": 4, "featured": True,
        "why": "An open-cast coal mine in the Hasdeo Arand forest, 2017 against 2025. The pit "
               "and its spoil heaps push into the forest. An illustrative lease boundary is drawn "
               "around the 2017 footprint, so expansion beyond it is flagged.",
    },
    "mumbai_coastal_road": {
        "center": (18.985, 72.812), "km": 6.0, "pin_tile": True,
        # winter 2017-18, before coastal-road work began in late 2018
        "before": ("2017-11-01", "2018-05-15"), "after": ("2024-12-01", "2025-04-30"),
        "short_title": "Mumbai coastal road", "place": "Mumbai, Maharashtra, India",
        "ps": ["Coastal change", "Infrastructure"], "order": 5, "featured": False,
        "why": "Land reclaimed from the Arabian Sea for Mumbai's coastal road, between Breach "
               "Candy, Haji Ali and Worli: new straight-edged land where there was water.",
    },
    "sindh_flood": {
        "center": (26.430, 67.700), "km": 25.0,
        "before": ("2022-04-01", "2022-06-10"), "after": ("2022-09-05", "2022-10-05"),
        "short_title": "Sindh floods 2022", "place": "Manchar Lake, Sindh, Pakistan",
        "ps": ["Flooding (natural)", "Water bodies"], "order": 6, "featured": False,
        "why": "Pakistan's 2022 monsoon floods: Manchar Lake swelled over towns and fields. With "
               "NIR/SWIR bands, water is confirmed by physics (MNDWI), not guessed from colour.",
    },
}


def square_bbox(lat: float, lon: float, km: float) -> list[float]:
    dlat = km / 2 / 110.574
    dlon = km / 2 / (111.320 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def tile_of(item: dict) -> str:
    return item["id"].split("_")[1]


def ranked_scenes(bbox, start, end, tile: str | None = None) -> list:
    """Candidate scenes, clearest over the box first (cloud, shadow and no-data by SCL)."""
    items = catalog.search("sentinel-2-l2a", bbox, start, end, cloud=40, limit=120)
    if tile:
        items = [it for it in items if tile_of(it) == tile]
    scored = []
    for it in sorted(items, key=lambda it: it["cloud"])[:10]:
        scl = raster.read_window(it["assets"]["scl"], bbox, out=96)
        if scl is not None:
            scored.append((float(np.isin(np.nan_to_num(scl), BAD_SCL).mean()), it))
    return [s for s in sorted(scored, key=lambda s: s[0]) if s[0] < 0.05]


def read_bands(item, bbox) -> dict:
    """Reflectance per band at SIZE x SIZE (Earth Search COGs are harmonised: DN / 10000)."""
    out = {}
    for b in BANDS:
        dn = None
        for attempt in range(3):              # COG reads over HTTP fail transiently under load
            dn = raster.read_window(item["assets"][b], bbox, out=SIZE)
            if dn is not None:
                break
            time.sleep(2 + 3 * attempt)
        if dn is None:
            raise RuntimeError(f"could not read {b} of {item['id']}")
        out[b] = np.nan_to_num(dn, nan=0.0) * 1e-4
    return out


def save_tif(path: str, bands: dict) -> None:
    arr = np.stack([np.clip(np.round(bands[b] * 10000), 0, 65535) for b in BANDS]).astype("uint16")
    with rasterio.open(path, "w", driver="GTiff", width=SIZE, height=SIZE, count=len(BANDS),
                       dtype="uint16", compress="deflate", predictor=2) as dst:
        dst.write(arr)
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)


def fetch(name: str, sc: dict) -> None:
    bbox = square_bbox(*sc["center"], sc["km"])
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    used = {}
    tile = None
    for when in ("before", "after"):
        bands = None
        for bad, item in ranked_scenes(bbox, *sc[when], tile=tile if sc.get("pin_tile") else None):
            try:                              # some older COGs have unreadable tiles: try the next
                bands = read_bands(item, bbox)
                break
            except RuntimeError as exc:
                print(f"  skip {item['id']}: {exc}")
        if bands is None:
            raise RuntimeError(f"{name}: no usable {when} scene")
        tile = tile or tile_of(item)
        save_tif(os.path.join(folder, f"{when}.tif"), bands)
        Image.fromarray(attribute.bands_to_rgb(bands)).save(os.path.join(folder, f"{when}.png"),
                                                            optimize=True)
        used[when] = {"id": item["id"], "date": item["datetime"][:10], "scene_cloud": item["cloud"],
                      "aoi_cloud_shadow_nodata": round(bad, 4), "stac": item["stac"]}
        print(f"  {when}: {item['id']} ({item['datetime'][:10]}), bad over area {bad:.1%}")
    pixel_m = round(math.sqrt(segment.pixel_area_m2(bbox, SIZE)), 2)
    meta = {k: sc[k] for k in ("short_title", "place", "ps", "order", "featured", "why")}
    meta.update({
        "title": f"{sc['short_title']} — {sc['place']} ({used['before']['date']} → {used['after']['date']})",
        "kind": "real", "simulated": False, "bands": True,
        "dates": f"{used['before']['date']} → {used['after']['date']}",
        "pixel_m": pixel_m, "pixel_note": "Sentinel-2 L2A, resampled to 512 px over the box",
        "bbox": [round(v, 5) for v in bbox], "scenes": used,
        "story": sc["why"],
    })
    old = os.path.join(folder, "meta.json")
    if os.path.exists(old):                   # keep hand-checked look_for / zone across re-fetches
        with open(old, encoding="utf-8") as fh:
            prev = json.load(fh)
        for k in ("look_for", "zone", "zone_note"):
            if k in prev:
                meta[k] = prev[k]
    with open(old, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    failed = []
    for name in sys.argv[1:] or SCENARIOS:
        print(name, flush=True)
        try:
            fetch(name, SCENARIOS[name])
        except RuntimeError as exc:           # keep going: one bad archive tile must not stop the rest
            print(f"  FAILED: {exc}", flush=True)
            failed.append(name)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")
