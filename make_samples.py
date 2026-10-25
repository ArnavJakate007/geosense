"""Build data/samples/ for the two-image compare page.

Builds the one SIMULATED pair; the real Sentinel-2 event pairs come from
make_real_samples.py. The simulated pair is the real Jewar "before" image with
a cleared patch, an access track and a flood patch painted into the "after".
It is labelled simulated everywhere it appears and must never be presented as
a real event.

Run:  .venv/bin/python make_samples.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache")      # base image for the simulated pair
OUT = os.path.join(HERE, "data", "samples")

PIXEL_NOTE = "from the run's bbox at 512 px; approximate"


def smooth_noise(shape, sigma, rng):
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    gain = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, shape)) * gain))
    return out / out.std()


def textured(shape, colour, rng, grain=7.0):
    tex = np.empty(shape + (3,), dtype="float32")
    fine = smooth_noise(shape, 0.8, rng) * grain
    for c in range(3):
        tex[..., c] = colour[c] + fine
    return tex


def polygon(shape, points):
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon(points, fill=1)
    return np.asarray(img, bool)


def line(shape, points, width):
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).line(points, fill=1, width=width)
    return np.asarray(img, bool)


def make_simulated():
    rng = np.random.default_rng(26227)
    base_run = "20260904T131633"
    before = np.asarray(Image.open(os.path.join(CACHE, base_run, "before.png")).convert("RGB"))
    shape = before.shape[:2]
    after = before.astype("float32").copy()

    clearing = polygon(shape, [(305, 334), (360, 328), (364, 376), (309, 382)])
    track = line(shape, [(206, 353), (306, 356)], 3)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rough = sum(w * smooth_noise(shape, s, rng) for s, w in ((10, 1.0), (5, 0.6), (2.5, 0.35)))
    flood = (1 - np.hypot(yy - 178, xx - 92) / 40) + 0.8 * rough / rough.std() > 0
    flood &= np.hypot(yy - 178, xx - 92) < 90

    for mask, colour, grain in ((clearing, (172, 142, 106), 8), (track, (178, 152, 120), 4),
                                (flood, (38, 52, 68), 4)):
        after[mask] = textured(shape, colour, rng, grain)[mask]

    after = after * 1.04 + 2 + rng.normal(0, 1.5, after.shape)       # lighting + sensor noise
    folder = os.path.join(OUT, "simulated_jewar")
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(before).save(os.path.join(folder, "before.png"), optimize=True)
    Image.fromarray(np.clip(after, 0, 255).astype("uint8")).save(
        os.path.join(folder, "after.png"), optimize=True)
    write_meta(folder, {
        "title": "SIMULATED — Jewar with painted clearing, track and flood",
        "pixel_m": 15.26, "pixel_note": PIXEL_NOTE, "simulated": True, "kind": "simulated",
        "source_run": base_run,
        "zone": [10, 15, 55, 95],
        "zone_note": "Example lease boundary: the painted clearing sits outside it.",
        "short_title": "Mining-lease breach", "place": "Jewar, Uttar Pradesh",
        "dates": "real image + 3 painted changes", "order": 1, "featured": True,
        "ps": ["Land excavation", "Vegetation loss", "Flooding (natural)"],
        "why": "One pair that shows the whole idea. A real satellite image with three changes "
               "painted in: a cleared pit outside the mining lease, a fresh track to it, and a "
               "flood beside the river. The tool has to tell the human-made from the natural.",
        "look_for": ["The cleared pit turns red: straight edges and a new track leading to it",
                     "It lies outside the lease boundary, so it is flagged and ranked #1",
                     "The flood beside the river turns blue: irregular edge, new water",
                     "Then open the evidence chain: Verify, then Tamper"],
        "story": "Real 'before'. The 'after' is the same image with three painted changes: a "
                 "rectangular cleared patch, a 3 px track joining it to the existing road, and an "
                 "irregular flood beside the river (painted as dark open water; turbid floodwater "
                 "in real true-colour imagery may not pass the RGB water test). Not a real event."})


def write_meta(folder, meta):
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    make_simulated()
    print("samples:", sorted(os.listdir(OUT)))
