"""Two-image attribution checks on synthetic images whose right answer is known."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import unittest

import numpy as np
from PIL import Image

from app import attribute, report
from app.attribute import HUMAN, IGNORED, NATURAL

GREEN = (40, 110, 40)
BROWN = (150, 110, 70)
DARK_BLUE = (25, 45, 75)
WHITE = (240, 240, 240)
SIZE = (256, 256)


def field_image(seed: int, shape=SIZE, colour=GREEN) -> np.ndarray:
    """Plain field with mild sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.empty(shape + (3,), dtype="float64")
    img[:] = colour
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype("uint8")


def smooth_noise(shape, sigma, rng) -> np.ndarray:
    """Gaussian-smoothed white noise (FFT), unit standard deviation."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    gain = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, shape)) * gain))
    return out / out.std()


def irregular_blob(shape, center, seed, radius=70, roughness=0.8) -> np.ndarray:
    """Disc roughened by multi-scale noise: a flood-like, terrain-following edge."""
    rng = np.random.default_rng(seed)
    noise = sum(w * smooth_noise(shape, s, rng) for s, w in ((16, 1.0), (8, 0.6), (4, 0.35)))
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dist = np.hypot(yy - center[0], xx - center[1]) / radius
    labels, n = attribute.label((1 - dist) + roughness * noise / noise.std() > 0)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def rect(shape, y0, x0, h, w) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y0 + h, x0:x0 + w] = True
    return m


def paint(img, mask, colour):
    out = img.copy()
    out[mask] = colour
    return out


def run(after, before=None, pixel_m=10.0, **kw):
    before = field_image(0) if before is None else before
    return attribute.attribute_arrays(before, after, pixel_m, **kw)


def png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


class ShapeTests(unittest.TestCase):
    def test_square_fd_is_one(self):
        m = rect((60, 60), 10, 10, 40, 40)
        fd = attribute.fractal_dimension(attribute.perimeter(m), m.sum(), 150)
        self.assertTrue(0.95 <= fd <= 1.05, fd)

    def test_label_is_eight_connected(self):
        m = np.zeros((10, 10), bool)
        m[1, 1] = m[2, 2] = True            # diagonal neighbours: one component
        m[1:3, 6:9] = True                  # separate block
        m[8, 0:10] = True                   # full-width row
        labels, n = attribute.label(m)
        self.assertEqual(n, 3)
        self.assertEqual(labels[1, 1], labels[2, 2])

    def test_rectilinearity_separates_built_from_natural(self):
        yy, xx = np.mgrid[:200, :200]
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        rotated = (abs((xx - 100) * c + (yy - 100) * s) < 50) & (abs(-(xx - 100) * s + (yy - 100) * c) < 25)
        self.assertGreater(attribute.rectilinearity(rect((200, 200), 40, 30, 80, 120)), 0.85)
        self.assertGreater(attribute.rectilinearity(rotated), 0.85)
        self.assertLess(attribute.rectilinearity(irregular_blob(SIZE, (128, 128), seed=7)), 0.4)

    def test_hull_of_rectangle_is_its_area(self):
        self.assertAlmostEqual(attribute.hull_area(rect((20, 30), 0, 0, 20, 30)), 600)
        ell = rect((20, 20), 0, 0, 20, 5) | rect((20, 20), 15, 0, 5, 20)
        self.assertGreater(attribute.hull_area(ell), ell.sum())


class AttributionTests(unittest.TestCase):
    def test_rectangle_is_human_clearing(self):
        after = paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN)
        regions = run(after).regions
        self.assertEqual(len(regions), 1)
        r = regions[0]
        self.assertEqual(r["type"], "clearing")
        self.assertEqual(r["cause"], HUMAN)
        self.assertTrue(any("FD" in x for x in r["reasons"]))

    def test_irregular_water_blob_is_natural_flooding(self):
        after = paint(field_image(1), irregular_blob(SIZE, (128, 128), seed=7), DARK_BLUE)
        main = max(run(after).regions, key=lambda r: r["area_px"])
        self.assertEqual(main["type"], "flooding")
        self.assertEqual(main["cause"], NATURAL)

    def test_cloud_is_ignored_and_excluded(self):
        yy, xx = np.mgrid[:SIZE[0], :SIZE[1]]
        cloud = ((yy - 128) / 35) ** 2 + ((xx - 128) / 50) ** 2 < 1
        result = run(paint(field_image(1), cloud, WHITE))
        self.assertTrue(result.regions)
        self.assertTrue(all(r["cause"] == IGNORED for r in result.regions))
        self.assertEqual(result.summary["regions"], 0)

    def test_track_touching_rectangle_is_linked(self):
        after = paint(field_image(1), rect(SIZE, 100, 60, 40, 60), BROWN)
        after = paint(after, rect(SIZE, 119, 120, 3, 90), BROWN)
        regions = run(after).regions
        tracks = [r for r in regions if r["is_track"]]
        blocks = [r for r in regions if not r["is_track"]]
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["type"], "access track")
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0]["track_link"])

    def test_identical_images_give_nothing(self):
        img = field_image(0)
        self.assertEqual(run(img.copy(), before=img).regions, [])

    def test_hectares_and_ranking(self):
        after = paint(field_image(1), rect(SIZE, 20, 20, 10, 10), BROWN)       # 100 px
        after = paint(after, rect(SIZE, 120, 100, 60, 80), BROWN)
        regions = run(after, pixel_m=10.0).regions
        self.assertEqual([r["id"] for r in regions], [1, 2])
        self.assertGreater(regions[0]["severity"], regions[1]["severity"])
        self.assertAlmostEqual(regions[1]["area_ha"], 1.0, places=2)

    def test_every_counted_region_has_reasons(self):
        after = paint(field_image(1), irregular_blob(SIZE, (128, 128), seed=3), DARK_BLUE)
        after = paint(after, rect(SIZE, 10, 10, 30, 40), BROWN)
        for r in run(after).regions:
            self.assertTrue(r["reasons"])

    def test_physics_from_bands(self):
        before, after = field_image(0), paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN)
        veg = {"red": 0.05, "green": 0.08, "blue": 0.04, "nir": 0.40, "swir16": 0.18}
        bare = {"red": 0.20, "green": 0.17, "blue": 0.12, "nir": 0.25, "swir16": 0.32}
        def stack(region_values):
            out = {k: np.full(SIZE, v, "float32") for k, v in veg.items()}
            for k, v in region_values.items():
                out[k][100:140, 100:160] = v
            return out
        result = attribute.attribute_arrays(before, after, 10.0, bands=(stack({}), stack(bare)))
        r = result.regions[0]
        self.assertTrue(any("NDVI fell" in x for x in r["reasons"]), r["reasons"])
        self.assertFalse(any("RGB-only" in n for n in result.notes))
        self.assertTrue(any("MNDWI" in n for n in result.notes))

    def test_chain_records_steps_and_verifies(self):
        result = run(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))
        exported = result.chain.export()
        self.assertTrue(exported["verified"])
        self.assertEqual([l["step"] for l in exported["links"]],
                         ["inputs", "preprocess", "detect", "attribute", "summary"])
        self.assertEqual(exported["links"][3]["detail"]["regions"][0]["id"], 1)

    def test_change_signal_and_lean_replace_zero_confidence(self):
        after = paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN)
        r = run(after).regions[0]
        self.assertGreaterEqual(r["change_sigma"], attribute.RULES["change_k"])
        self.assertIn(r["change_label"], ("clear", "strong", "very strong"))
        self.assertTrue(r["lean"].endswith("human-made"))
        self.assertIsNone(r["next_step"])
        self.assertEqual(attribute.lean_text(50, attribute.UNCERTAIN), "Evidence balanced: needs a second look")
        self.assertEqual(attribute.lean_text(15, attribute.NATURAL), "Clearly natural")

    def test_chain_details_are_browser_safe(self):
        safe = attribute.chain_safe({"a": 1.0, "b": 0.00001, "c": np.float64(2.5), "d": (1, 2.0),
                                     "e": np.bool_(True), "f": float("nan")})
        self.assertEqual(safe, {"a": 1, "b": 0, "c": 2.5, "d": [1, 2], "e": True, "f": None})
        self.assertIsInstance(safe["a"], int)

    def test_evidence_is_tagged_with_points(self):
        after = paint(field_image(1), rect(SIZE, 100, 60, 40, 60), BROWN)
        after = paint(after, rect(SIZE, 119, 120, 3, 90), BROWN)
        block = next(r for r in run(after).regions if not r["is_track"])
        tags = {e["tag"] for e in block["evidence"]}
        self.assertTrue({"shape", "bands", "context"} <= tags, block["evidence"])
        self.assertEqual(block["human_score"],
                         min(100, 50 + sum(e["points"] for e in block["evidence"])))
        self.assertEqual(block["reasons"], [e["text"] for e in block["evidence"]])

    def test_permitted_zone_flags_outside_region(self):
        after = paint(field_image(1), rect(SIZE, 30, 170, 40, 60), BROWN)    # right side
        after = paint(after, rect(SIZE, 150, 20, 40, 60), BROWN)             # left side
        result = run(after, zone_pct=(0, 0, 50, 100))
        outside = [r for r in result.regions if r["outside_zone"]]
        inside = [r for r in result.regions if not r["outside_zone"]]
        self.assertEqual(len(outside), 1)
        self.assertGreaterEqual(outside[0]["bbox"][0], 128)
        self.assertTrue(any(e["tag"] == "context" and "permitted zone" in e["text"]
                            for e in outside[0]["evidence"]))
        self.assertEqual(inside[0]["outside_zone_frac"], 0.0)
        self.assertEqual(result.summary["outside_zone"], 1)
        self.assertEqual(result.regions[0]["id"], 1)
        self.assertTrue(result.regions[0]["outside_zone"])        # 1.5x severity ranks it first

    def test_labels_png_encodes_region_ids(self):
        result = run(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))
        img = np.asarray(Image.open(io.BytesIO(attribute.labels_png(result))).convert("RGB")).astype(int)
        ids = img[..., 0] + 256 * img[..., 1]
        self.assertTrue(np.array_equal(ids, result.labels))

    def test_bytes_entry_point_hashes_raw_files(self):
        b, a = png(field_image(0)), png(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))
        result = attribute.attribute(b, a, 10.0, names=("b.png", "a.png"))
        self.assertEqual(result.inputs["before"]["sha256"], attribute.hashlib.sha256(b).hexdigest())
        self.assertTrue(attribute.overlay_png(result).startswith(b"\x89PNG"))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.result = run(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))

    def test_markdown_fingerprint_detects_tampering(self):
        md = report.build_markdown(self.result)
        self.assertTrue(report.verify_markdown(md))
        self.assertIn(self.result.chain.head, md)
        self.assertFalse(report.verify_markdown(md.replace("clearing", "flooding", 1)))

    def test_html_quotes_fingerprint_and_chain(self):
        page = report.build_html(self.result, attribute.overlay_png(self.result))
        fingerprint = report.build_markdown(self.result).rpartition("\n")[2].split(": ")[1]
        self.assertIn(fingerprint, page)
        self.assertIn("verified: <b>true</b>", page)


class BrowserChainTests(unittest.TestCase):
    """compare.html re-verifies the chain in the browser; its hashing must match provenance.py."""

    def test_page_javascript_verifies_a_python_chain(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        page = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "compare.html"),
                    encoding="utf-8").read()
        core = page.split("/* chain-core:start */")[1].split("/* chain-core:end */")[0]
        b = png(field_image(0))
        a = png(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))
        result = attribute.attribute(b, a, 10.0, names=("before é.png", "after—x.png"),
                                     source={"title": "Test — Δ pair", "simulated": True},
                                     zone_pct=(5, 5, 60, 95))
        links = result.chain.export()["links"]
        tampered = json.loads(json.dumps(links))
        tampered[3]["detail"]["regions"][0]["cause"] = "likely natural"
        script = core + f"""
            (async () => {{
              const ok = await verifyLinks({json.dumps(links)});
              const bad = await verifyLinks({json.dumps(tampered)});
              console.log(JSON.stringify({{ok: ok.map(s => s.status), bad: bad.map(s => s.status)}}));
            }})();"""
        out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        statuses = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(statuses["ok"], ["valid"] * 5)
        self.assertEqual(statuses["bad"], ["valid", "valid", "valid", "mismatch", "broken-link"])

    def test_page_fallback_sha256_matches_hashlib(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        page = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "compare.html"),
                    encoding="utf-8").read()
        core = page.split("/* chain-core:start */")[1].split("/* chain-core:end */")[0]
        samples = ["", "abc", "x" * 55, "y" * 64, "z" * 1000]
        script = core + f"""
            console.log(JSON.stringify({json.dumps(samples)}.map(s => sha256Fallback(new TextEncoder().encode(s)))));"""
        out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout.strip()),
                         [attribute.hashlib.sha256(s.encode()).hexdigest() for s in samples])


class CompareApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:                     # httpx is a test-only extra
            raise unittest.SkipTest(f"TestClient unavailable: {exc}")
        from app.main import app
        cls.client = TestClient(app)

    def test_compare_upload_and_report(self):
        b = png(field_image(0))
        a = png(paint(field_image(1), rect(SIZE, 100, 100, 40, 60), BROWN))
        res = self.client.post("/api/compare", json={
            "before_b64": base64.b64encode(b).decode(), "after_b64": base64.b64encode(a).decode(),
            "before_name": "b.png", "after_name": "a.png", "pixel_m": 10})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["summary"]["human"], 1)
        self.assertTrue(body["chain"]["verified"])
        page = self.client.get(body["urls"]["report_html"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("change attribution report", page.text)
        self.assertEqual(self.client.get(body["urls"]["overlay"]).status_code, 200)

    def test_real_band_sample_uses_physics(self):
        samples = self.client.get("/api/compare/samples").json()["samples"]
        banded = [s for s in samples if s.get("bands")]
        if not banded:
            self.skipTest("no 5-band samples fetched")
        res = self.client.post("/api/compare", json={"sample": banded[0]["name"]})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertTrue(body["inputs"]["before"]["bands"])
        self.assertTrue(any("MNDWI" in n for n in body["notes"]))
        for r in body["regions"]:
            self.assertIn("change_sigma", r)
            self.assertIn("lean", r)

    def test_compare_rejects_garbage(self):
        res = self.client.post("/api/compare", json={
            "before_b64": base64.b64encode(b"nope").decode(),
            "after_b64": base64.b64encode(b"nope").decode(), "pixel_m": 10})
        self.assertEqual(res.status_code, 400)

    def test_samples_listed_and_runnable(self):
        samples = self.client.get("/api/compare/samples").json()["samples"]
        self.assertTrue(samples)
        res = self.client.post("/api/compare", json={"sample": samples[0]["name"]})
        self.assertEqual(res.status_code, 200, res.text)

    def test_compare_page_served(self):
        self.assertEqual(self.client.get("/compare").status_code, 200)

    def test_compare_with_zone_and_labels(self):
        samples = self.client.get("/api/compare/samples").json()["samples"]
        sim = next(s for s in samples if s.get("simulated"))
        self.assertEqual(self.client.get(sim["thumbs"]["before"]).status_code, 200)
        res = self.client.post("/api/compare", json={"sample": sim["name"], "zone": sim["zone"]})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["settings"]["zone"], sim["zone"])
        self.assertGreaterEqual(body["summary"]["outside_zone"], 1)
        self.assertEqual(self.client.get(body["urls"]["labels"]).status_code, 200)
        bad = self.client.post("/api/compare", json={"sample": sim["name"], "zone": [60, 0, 10, 100]})
        self.assertEqual(bad.status_code, 422)

    def test_cached_run_chips_analysed_at_native_resolution(self):
        res = self.client.get("/api/runs/20260904T125514/attribution")
        if res.status_code == 404:
            self.skipTest("cached run not present")
        chips = res.json()["chips"]
        self.assertTrue(chips)
        for chip in chips:                      # Sentinel-2 review read is ~10 m
            self.assertAlmostEqual(chip["pixel_m"], 10.2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
