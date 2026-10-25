"""Evidence-ready report for a two-image attribution.

Markdown is the canonical form: its last line is the SHA-256 of everything
above it, so anyone can check the file was not edited after analysis. The HTML
version renders the same facts for printing to PDF from the browser (no PDF
library needed) and quotes the Markdown fingerprint.
"""
from __future__ import annotations

import base64
import hashlib
import html

from .attribute import IGNORED, Attribution

HASH_PREFIX = "Report SHA-256 (of all text above this line): "
TAG_LABEL = {"shape": "Shape", "bands": "Bands", "context": "Context"}

LIMITATIONS = [
    "Labels are \"likely\" and meant for triage - deciding where to look first - not final judgement.",
    "Thresholds were tuned on the demo pairs in data/samples, not validated on a labelled benchmark.",
    "Images are not co-registered. A shifted pair produces false change along edges.",
    "Seasonal crop and vegetation changes are real colour change and can be reported; compare "
    "images from the same season where possible.",
    "Without NIR/SWIR bands, colour proxies (Excess Green, dark-blue water) stand in for NDVI and "
    "MNDWI and the physics checks are skipped.",
    "Cloud shadows are not filtered.",
    "SHA-256 fingerprints and the provenance chain prove the inputs and this report were not altered "
    "after analysis. They do not prove where or when the images were captured.",
]


def region_sentence(r: dict) -> str:
    if r["cause"] == IGNORED:
        return f"Region {r['id']}: {r['area_ha']:.2f} ha of cloud-like change, ignored."
    text = (f"Region {r['id']}: {r['area_ha']:.2f} ha of {r['type']}, {r['cause']} — "
            f"{r.get('lean', '').lower()} (cause score {r['human_score']}/100; change "
            f"{r.get('change_sigma', 0):.1f}σ above image noise, {r.get('change_label', '')})")
    if r["track_link"]:
        text += ", connected to a new track"
    if r.get("outside_zone"):
        text += f", {r['outside_zone_frac']:.0%} outside the permitted zone"
    return text + "."


def evidence_line(e: dict) -> str:
    pts = f" ({e['points']:+d})" if e.get("points") else ""
    return f"[{TAG_LABEL.get(e['tag'], e['tag'])}] {e['text']}{pts}"


def build_markdown(result: Attribution) -> str:
    s, st, inp = result.summary, result.settings, result.inputs
    chain = result.chain.export()
    source = inp.get("source") or {}
    counted = [r for r in result.regions if r["cause"] != IGNORED]
    ignored = [r for r in result.regions if r["cause"] == IGNORED]
    lines = [
        "# GeoSense - change attribution report",
        "",
        f"- Analysis ID: `{result.id}`",
        f"- Generated: {result.created}",
        "- Method: rule-based two-image attribution (shape, context and, when bands exist, physics). "
        "No trained model, no network.",
    ]
    if source:
        lines.append(f"- Source: {source.get('title', '')}"
                     + (" — **not a real event**" if source.get("simulated") else ""))
    lines += [
        "",
        "## Inputs",
        "",
        "| Image | File | Size | Bands | SHA-256 |",
        "|---|---|---|---|---|",
        *[f"| {k.capitalize()} | {inp[k]['name']} | {inp[k]['size'][0]} × {inp[k]['size'][1]} px | "
          f"{'RGB + NIR/SWIR' if inp[k]['bands'] else 'RGB'} | `{inp[k]['sha256']}` |"
          for k in ("before", "after")],
        "",
        "## Settings",
        "",
        f"- Pixel size: {st['pixel_m']:g} m/px{_pixel_note(source)}; working image "
        f"{st['working_size'][0]} × {st['working_size'][1]} px (downscale {st['downscale']:g})",
        f"- Sensitivity {st['sensitivity']:g}, minimum region {st['min_region_px']} px",
        f"- Permitted zone: {_zone_text(st.get('zone'))}",
        f"- Rules fingerprint: `{st['rules_sha256']}`",
        "",
        "## Summary",
        "",
        f"- Changed regions: {s['regions']} ({s['changed_ha']:.2f} ha)",
        f"- Likely human-made: {s['human']} ({s['human_ha']:.2f} ha), of which new tracks: {s['tracks']}",
        f"- Likely natural: {s['natural']} ({s['natural_ha']:.2f} ha)",
        f"- Uncertain, needs review: {s['uncertain']} ({s['uncertain_ha']:.2f} ha)",
        f"- Ignored as cloud: {s['ignored']} ({s['ignored_ha']:.2f} ha, excluded from totals)",
    ]
    if st.get("zone"):
        lines.append(f"- Outside the permitted zone: {s['outside_zone']} region(s), "
                     f"{s['outside_zone_ha']:.2f} ha beyond the boundary")
    lines += [*[f"- Note: {n}" for n in result.notes], "", "## Findings (ranked by severity)", ""]
    if not counted:
        lines.append("No changed regions at the current settings.")
    for r in counted:
        flag = " — OUTSIDE PERMITTED ZONE" if r.get("outside_zone") else ""
        lines += [f"### {r['id']}. {r['type'].capitalize()} — {r['cause']}{flag}", "",
                  region_sentence(r), "", *[f"- {evidence_line(e)}" for e in r["evidence"]], "",
                  *([f"To settle it: {r['next_step']}", ""] if r.get("next_step") else []),
                  _measurements(r), ""]
    if ignored:
        lines += ["## Ignored regions", "", *[f"- {region_sentence(r)}" for r in ignored], ""]
    lines += [
        "## Provenance",
        "",
        f"- Chain head: `{chain['head']}`",
        f"- Chain verified: {str(chain['verified']).lower()}",
        *[f"- {l['n']}. {l['step']} — `{l['hash'][:16]}…`" for l in chain["links"]],
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in LIMITATIONS],
        "",
        "---",
        "",
    ]
    body = "\n".join(lines)
    return body + "\n" + HASH_PREFIX + hashlib.sha256(body.encode("utf-8")).hexdigest()


def verify_markdown(text: str) -> bool:
    """True if the last-line fingerprint matches everything above it."""
    body, _, last = text.rpartition("\n")
    return (last.startswith(HASH_PREFIX)
            and last[len(HASH_PREFIX):] == hashlib.sha256(body.encode("utf-8")).hexdigest())


def build_html(result: Attribution, overlay_png: bytes | None = None) -> str:
    """Printable HTML with the same facts as the Markdown, plus the overlay image."""
    md = build_markdown(result)
    fingerprint = md.rpartition("\n")[2][len(HASH_PREFIX):]
    s, st, inp = result.summary, result.settings, result.inputs
    chain = result.chain.export()
    source = inp.get("source") or {}
    e = html.escape
    rows = "".join(
        f"<tr><td>{r['id']}</td><td>{e(r['type'])}</td><td><span class='cause c-{_slug(r['cause'])}'>"
        f"{e(r['cause'])}</span></td><td>{r['human_score']}/100</td><td>{r['area_ha']:.2f}</td>"
        f"<td>{r.get('change_sigma', 0):.1f}σ</td><td>{_fmt(r['fd'])}</td><td>{'yes' if r.get('outside_zone') else ''}</td>"
        f"<td>{r['severity']:.3f}</td></tr>" for r in result.regions)
    cards = "".join(_card_html(r) for r in result.regions if r["cause"] != IGNORED)
    links = "".join(f"<li><b>#{l['n']} {e(l['step'])}</b> <code>{l['hash']}</code></li>"
                    for l in chain["links"])
    img = (f"<img src='data:image/png;base64,{base64.b64encode(overlay_png).decode()}' alt='Labelled "
           "change regions on the after image'>" if overlay_png else "")
    banner = ("<p class='sim'>SIMULATED PAIR — changes were painted into a real image for "
              "demonstration. Not a real event.</p>" if source.get("simulated") else "")
    zone_line = (f"<p>Permitted zone: {e(_zone_text(st.get('zone')))} — <b>{s['outside_zone']}</b> "
                 f"region(s), {s['outside_zone_ha']:.2f} ha outside.</p>" if st.get("zone") else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GeoSense report {e(result.id)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--ink:#0b0b0b;--paper:#fffdf5;--human:#ff4d2e;--natural:#2f7bff;--uncertain:#ffc22e;--ignored:#b9b9b9;--lime:#c8ff3d}}
body{{font:14px/1.5 Archivo,system-ui,"Segoe UI",sans-serif;color:var(--ink);background:var(--paper);max-width:980px;margin:24px auto;padding:0 16px}}
h1{{font-size:28px;font-weight:900;margin:0;text-transform:uppercase;letter-spacing:-.01em}}
h2{{font-size:16px;font-weight:900;text-transform:uppercase;border-bottom:3px solid var(--ink);padding-bottom:4px;margin-top:26px}}
h3{{font-size:15px;margin:0 0 6px;font-weight:800}} code,.m{{font:12px "JetBrains Mono",ui-monospace,Consolas,monospace;word-break:break-all}}
.box{{border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);background:#fff;padding:12px 14px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;border:3px solid var(--ink)}} td,th{{border:2px solid var(--ink);padding:5px 8px;text-align:left;font-size:12.5px}}
th{{background:var(--lime);font-weight:800;text-transform:uppercase;font-size:11px}}
.card{{border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);background:#fff;padding:10px 14px;margin:12px 0;break-inside:avoid;border-left-width:12px}}
.k-likely-human-made{{border-left-color:var(--human)}} .k-likely-natural{{border-left-color:var(--natural)}} .k-uncertain{{border-left-color:var(--uncertain)}}
.cause{{display:inline-block;border:2px solid var(--ink);padding:0 6px;font-size:11px;font-weight:800;text-transform:uppercase}}
.c-likely-human-made{{background:var(--human)}} .c-likely-natural{{background:var(--natural)}}
.c-uncertain{{background:var(--uncertain)}} .c-ignored-cloud{{background:var(--ignored)}}
.flag{{display:inline-block;background:var(--ink);color:var(--uncertain);font-size:11px;font-weight:800;padding:1px 6px}}
.tag{{display:inline-block;border:2px solid var(--ink);font:600 10px "JetBrains Mono",monospace;padding:0 4px;text-transform:uppercase}}
.t-shape{{background:#ffe1f0}} .t-bands{{background:#d9f1ff}} .t-context{{background:#eaffc2}}
img{{width:100%;max-width:680px;border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);display:block}}
.sim{{background:var(--uncertain);border:3px solid var(--ink);padding:8px 10px;font-weight:900}}
.settle{{border:2px dashed var(--ink);background:#fff6d6;padding:6px 10px}}
.dim{{color:#444}} ol{{list-style:none;padding:0}} li{{margin:3px 0}}
@media print{{body{{margin:0;background:#fff}} .noprint{{display:none}} .box,.card,img{{box-shadow:none}}}}
</style></head><body>
<p class="noprint dim">Print this page (Ctrl/Cmd + P) to save it as a PDF.</p>
<h1>GeoSense — change attribution report</h1>
<p class="dim">Analysis <code>{e(result.id)}</code> · {e(result.created)} · rule-based, no trained model, no network</p>
{banner}
<div class="box"><b>{s['regions']}</b> changed regions ({s['changed_ha']:.2f} ha):
<span class="cause c-likely-human-made">{s['human']} human-made</span> ({s['human_ha']:.2f} ha, {s['tracks']} new track{'' if s['tracks'] == 1 else 's'}) ·
<span class="cause c-likely-natural">{s['natural']} natural</span> · <span class="cause c-uncertain">{s['uncertain']} uncertain</span> ·
<span class="cause c-ignored-cloud">{s['ignored']} cloud ignored</span>{zone_line}
{''.join(f'<p class="dim">Note: {e(n)}</p>' for n in result.notes)}</div>
{img}
<h2>Inputs</h2>
<table><tr><th>Image</th><th>File</th><th>Size</th><th>SHA-256</th></tr>
{''.join(f"<tr><td>{k.capitalize()}</td><td>{e(inp[k]['name'])}</td><td>{inp[k]['size'][0]} × {inp[k]['size'][1]}</td><td><code>{inp[k]['sha256']}</code></td></tr>" for k in ('before', 'after'))}
</table>
<p>Pixel size {st['pixel_m']:g} m/px{e(_pixel_note(source))} · sensitivity {st['sensitivity']:g} · minimum region {st['min_region_px']} px</p>
<h2>Regions, ranked by severity</h2>
<table><tr><th>#</th><th>Type</th><th>Cause</th><th>Score</th><th>Area (ha)</th><th>Change signal</th><th>FD</th><th>Outside zone</th><th>Severity</th></tr>{rows}</table>
<h2>Findings</h2>{cards or '<p>No changed regions at the current settings.</p>'}
<h2>Evidence chain</h2>
<div class="box"><p>Chain head <code>{chain['head']}</code> · verified: <b>{str(chain['verified']).lower()}</b></p><ol>{links}</ol></div>
<h2>Limitations</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in LIMITATIONS)}</ul>
<p class="dim">Fingerprint of the Markdown version of this report (SHA-256 of all its text above the last line):<br><code>{fingerprint}</code></p>
</body></html>"""


def _card_html(r: dict) -> str:
    e = html.escape
    slug = _slug(r["cause"])
    flag = " <span class='flag'>OUTSIDE PERMITTED ZONE</span>" if r.get("outside_zone") else ""
    items = []
    for x in r["evidence"]:
        pts = f" <b>{x['points']:+d}</b>" if x.get("points") else ""
        items.append(f"<li><span class='tag t-{x['tag']}'>{TAG_LABEL.get(x['tag'], x['tag'])}</span> "
                     f"{e(x['text'])}{pts}</li>")
    settle = (f"<p class='settle'><b>To settle it:</b> {e(r['next_step'])}</p>"
              if r.get("next_step") else "")
    return (f"<section class='card k-{slug}'><h3>#{r['id']} {e(r['type'])} "
            f"<span class='cause c-{slug}'>{e(r['cause'])}</span>{flag}</h3>"
            f"<p>{e(region_sentence(r))}</p><ul>{''.join(items)}</ul>{settle}"
            f"<p class='m'>{e(_measurements(r))}</p></section>")


def _measurements(r: dict) -> str:
    return (f"Measurements: {r['area_px']} px · FD {_fmt(r['fd'])} · rectilinearity "
            f"{_fmt(r.get('rectilinearity'))} · rectangularity {r['rectangularity']:.2f} · "
            f"solidity {r['solidity']:.2f} · elongation {r['elongation']:.1f} · "
            f"ΔExG {r['mean_d_exg']:+.3f} · ΔL {r['mean_d_L']:+.1f} · human score {r['human_score']}")


def _fmt(v) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def _slug(cause: str) -> str:
    return cause.replace(" ", "-").replace("(", "").replace(")", "")


def _pixel_note(source: dict) -> str:
    return f" ({source['pixel_note']})" if source.get("pixel_note") else ""


def _zone_text(zone) -> str:
    if not zone:
        return "none"
    left, top, right, bottom = zone
    return f"left {left:g}%, top {top:g}%, right {right:g}%, bottom {bottom:g}% of the image"
