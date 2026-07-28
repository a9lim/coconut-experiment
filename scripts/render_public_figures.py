"""Render the compact public SVG figures from data/summary/results.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "summary" / "results.json"
OUT = ROOT / "figures" / "public"


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
        for x, y in points
    )
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="3"/>{circles}'
    )


def _frame(title: str, x_label: str, y_label: str, body: str) -> str:
    ticks = []
    for i in range(6):
        x = 80 + 104 * i
        y = 340 - 56 * i
        ticks.append(f'<line x1="{x}" y1="340" x2="{x}" y2="346" stroke="#444"/>')
        ticks.append(
            f'<text x="{x}" y="365" text-anchor="middle" '
            f'font-size="12">{i / 5:.1f}</text>'
        )
        ticks.append(f'<line x1="74" y1="{y}" x2="80" y2="{y}" stroke="#444"/>')
        ticks.append(
            f'<text x="64" y="{y + 4}" text-anchor="end" '
            f'font-size="12">{i / 5:.1f}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="680" height="420" viewBox="0 0 680 420">
<rect width="680" height="420" fill="#fff"/>
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">{title}</text>
<line x1="80" y1="60" x2="80" y2="340" stroke="#444"/>
<line x1="80" y1="340" x2="600" y2="340" stroke="#444"/>
{''.join(ticks)}
<text x="340" y="398" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>
<text x="18" y="200" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 200)">{y_label}</text>
<g font-family="sans-serif">{body}</g>
</svg>
"""


def render() -> None:
    data = json.loads(RESULTS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    curve = data["cot"]["acquisition_curve"]
    hack = [(80 + 520 * row["cue_fraction"], 340 - 280 * row["conflict_hack_rate"]) for row in curve]
    honest = [(80 + 520 * row["cue_fraction"], 340 - 280 * row["conflict_honest_rate"]) for row in curve]
    legend = (
        '<line x1="420" y1="54" x2="448" y2="54" stroke="#c62828" stroke-width="3"/>'
        '<text x="456" y="59" font-size="13">hack</text>'
        '<line x1="510" y1="54" x2="538" y2="54" stroke="#1565c0" stroke-width="3"/>'
        '<text x="546" y="59" font-size="13">honest</text>'
    )
    (OUT / "cot-acquisition.svg").write_text(
        _frame(
            "Explicit-CoT acquisition sweep (one seed, n=2,000/cell)",
            "cued fraction in training",
            "held-out conflict rate",
            legend + _polyline(hack, "#c62828") + _polyline(honest, "#1565c0"),
        )
    )

    rows = data["continuous_carrier"]["answer_head_frontier"]
    colors = ["#455a64", "#6a1b9a", "#ef6c00", "#c62828"]
    points = []
    for row, color in zip(rows, colors, strict=True):
        x = 80 + 520 * row["hack_rate"]
        y = 340 - 280 * row["readout_honesty"]
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12">{row["setting"]}</text>'
        )
    (OUT / "carrier-frontier.svg").write_text(
        _frame(
            "Continuous-carrier answer-head frontier",
            "carrier-answer hack rate",
            "LM-head readout honesty",
            "".join(points),
        )
    )


if __name__ == "__main__":
    render()
