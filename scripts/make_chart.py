"""Render the quantization sensitivity finding as SVG, light and dark.

Two files rather than one with a media query: GitHub serves README images
through a proxy where `prefers-color-scheme` inside the SVG is unreliable, but
markdown's <picture> element with a media attribute is honoured. So the theme
switch happens in the markdown, not in the graphic.

One horizontal bar per weight-matrix type, sorted by cost, single series — the
chart answers one question ("which matrices tolerate INT4?") so it needs one
hue, no legend, and a direct label on every bar because the value *is* the
finding.

    python scripts/make_chart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Validated defaults: categorical slot 1, checked against both surfaces with
# scripts/validate_palette.js — all six checks pass in light and dark.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  series="#2a78d6", grid="#e4e3df", axis="#b9b7b0"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 series="#3987e5", grid="#2e2e2c", axis="#4a4a46"),
}

FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif")

PRETTY = {
    "attn.q": "q_proj", "attn.k": "k_proj", "attn.v": "v_proj",
    "attn.o": "o_proj", "mlp.gate": "gate_proj", "mlp.up": "up_proj",
    "mlp.down": "down_proj", "attn.all": "all attention", "mlp.all": "all MLP",
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(data: dict, theme: str) -> str:
    t = THEMES[theme]
    rows = sorted(
        ((PRETTY.get(k, k), v["delta_pct"]) for k, v in data["by_layer_type"].items()),
        key=lambda r: -r[1],
    )

    pad_l, pad_r, pad_t, pad_b = 132, 74, 64, 44
    bar_h, gap = 26, 10          # 2px minimum surface gap, widened for legibility
    plot_w = 430
    h = pad_t + len(rows) * (bar_h + gap) - gap + pad_b
    w = pad_l + plot_w + pad_r

    vmax = max(v for _, v in rows)
    # Round the axis up to a clean step so gridlines land on readable numbers.
    step = 2 if vmax <= 10 else 5
    top = step * (int(vmax / step) + 1)

    def x(v: float) -> float:
        return pad_l + (v / top) * plot_w

    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}" role="img" '
        f'aria-label="Perplexity cost of INT4 quantization by weight matrix type">',
        f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>',
        f'<text x="{pad_l - 108}" y="28" fill="{t["ink"]}" font-size="15" '
        f'font-weight="600">Which weights tolerate INT4?</text>',
        f'<text x="{pad_l - 108}" y="47" fill="{t["muted"]}" font-size="11.5">'
        f'Perplexity cost when only that matrix type is quantized '
        f'(INT4, group 128, asymmetric)</text>',
    ]

    # Recessive gridlines, drawn under the bars.
    v = 0
    while v <= top:
        gx = x(v)
        o.append(f'<line x1="{gx:.1f}" y1="{pad_t - 8}" x2="{gx:.1f}" '
                 f'y2="{h - pad_b + 4}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{h - pad_b + 20}" fill="{t["muted"]}" '
                 f'font-size="11" text-anchor="middle">+{v}%</text>')
        v += step

    for i, (label, val) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        bw = max(2.0, x(val) - pad_l)
        composite = label.startswith("all ")
        # Composite rows (all attention / all MLP) are the sum-effect bars and
        # are drawn lighter so they read as a summary, not another matrix type.
        fill_op = "0.45" if composite else "1"

        o.append(f'<text x="{pad_l - 12}" y="{y + bar_h / 2 + 4:.1f}" '
                 f'fill="{t["ink"] if not composite else t["muted"]}" font-size="12" '
                 f'text-anchor="end"'
                 f'{" font-weight=\"600\"" if composite else ""}>{esc(label)}</text>')
        # 4px rounded data-end, square against the baseline.
        o.append(f'<path d="M{pad_l} {y} h{bw - 4:.1f} a4 4 0 0 1 4 4 '
                 f'v{bar_h - 8} a4 4 0 0 1 -4 4 h-{bw - 4:.1f} z" '
                 f'fill="{t["series"]}" fill-opacity="{fill_op}"/>')
        o.append(f'<text x="{pad_l + bw + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
                 f'fill="{t["muted"]}" font-size="11.5">+{val:.2f}%</text>')

    o.append(f'<line x1="{pad_l}" y1="{pad_t - 8}" x2="{pad_l}" '
             f'y2="{h - pad_b + 4}" stroke="{t["axis"]}" stroke-width="1"/>')
    o.append(f'<text x="{pad_l - 108}" y="{h - 10}" fill="{t["muted"]}" '
             f'font-size="10.5">Qwen2.5-0.5B-Instruct · {data["tokens"]:,} held-out '
             f'tokens · fp32 baseline ppl {data["baseline"]["perplexity"]:.3f}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    src = RESULTS / "quant_study.json"
    if not src.exists():
        src = RESULTS / "quant_study_quick.json"
    if not src.exists():
        print("no study results — run scripts/quant_study.py first")
        return 1

    data = json.loads(src.read_text())
    for theme in THEMES:
        out = RESULTS / f"quant_sensitivity_{theme}.svg"
        out.write_text(render(data, theme))
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
