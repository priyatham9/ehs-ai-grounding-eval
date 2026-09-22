#!/usr/bin/env python3
"""Deterministic SVG scatter of the baseline arms.

Reads results/baselines_summary.csv and draws x = accuracy, y = adjacent
substitution rate, one point per arm, with 95% interval bars taken from the
CSV's own confidence-interval columns. Palette and marks follow the dataviz
skill's categorical formula (3 validated slots + a neutral reference marker
for the oracle ceiling), mapped onto the site tokens defined in
docs/index.html. No timestamps or unordered dict iteration are used, so the
output is byte-identical across runs.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "results", "baselines_summary.csv")
SVG_PATH = os.path.join(REPO_ROOT, "docs", "baselines.svg")

# Fixed plotting order (never derived from dict/set iteration order).
ARM_ORDER = ["random_floor", "retrieval_tfidf", "retrieval_bm25", "oracle"]

ARM_LABEL = {
    "random_floor": "Random floor",
    "retrieval_tfidf": "TF-IDF retrieval",
    "retrieval_bm25": "BM25 retrieval",
    "oracle": "Oracle ceiling",
}

# dataviz skill categorical slots 1 (blue), 2 (orange), 3 (aqua) -- validated
# --pairs all in both light and dark (references/palette.md). The oracle is
# a reference ceiling, not a competing arm, so it gets a neutral marker
# (--ink-2) instead of a fourth categorical hue.
ARM_COLOR_LIGHT = {
    "random_floor": "#2a78d6",
    "retrieval_tfidf": "#eb6834",
    "retrieval_bm25": "#1baf7a",
    "oracle": "#3A403C",
}
ARM_COLOR_DARK = {
    "random_floor": "#3987e5",
    "retrieval_tfidf": "#d95926",
    "retrieval_bm25": "#199e70",
    "oracle": "#BFC4CF",
}

ARM_SHAPE = {
    "random_floor": "circle",
    "retrieval_tfidf": "circle",
    "retrieval_bm25": "circle",
    "oracle": "diamond",
}

# Plot geometry.
W, H = 720, 460
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 78, 32, 28, 60
PLOT_X0, PLOT_Y0 = MARGIN_L, MARGIN_T
PLOT_X1, PLOT_Y1 = W - MARGIN_R, H - MARGIN_B
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.6

# The "no language model scored yet" hatch region: upper-right quadrant of
# the data space (high accuracy, high adjacent-substitution rate), split at
# the midpoint of each axis.
HATCH_X_MIN = 0.6
HATCH_Y_MIN = 0.3


def _parse_csv_line(line):
    # No embedded commas/quotes in this file (all fields are plain numbers or
    # bare identifiers), so a simple split avoids depending on the csv module.
    return line.rstrip("\n").split(",")


def read_rows():
    with open(CSV_PATH) as f:
        lines = [line for line in f.read().splitlines() if line]
    header = _parse_csv_line(lines[0])
    rows = {}
    for line in lines[1:]:
        values = _parse_csv_line(line)
        row = dict(zip(header, values))
        rows[row["adapter"]] = row
    return header, rows


def fx(v):
    v = float(v)
    frac = (v - X_MIN) / (X_MAX - X_MIN)
    return PLOT_X0 + frac * (PLOT_X1 - PLOT_X0)


def fy(v):
    v = float(v)
    frac = (v - Y_MIN) / (Y_MAX - Y_MIN)
    return PLOT_Y1 - frac * (PLOT_Y1 - PLOT_Y0)


def fmt(n):
    # Fixed precision, deterministic, no locale/float-repr drift.
    return ("%.2f" % n).rstrip("0").rstrip(".") if "." in "%.2f" % n else "%.2f" % n


def marker(shape, x, y, color, arm_id):
    if shape == "circle":
        return (
            '<circle class="mark" data-arm="%s" cx="%.2f" cy="%.2f" r="6" '
            'fill="%s" stroke="var(--surface)" stroke-width="2"/>'
            % (arm_id, x, y, color)
        )
    # diamond
    d = 7
    pts = "%.2f,%.2f %.2f,%.2f %.2f,%.2f %.2f,%.2f" % (
        x, y - d,
        x + d, y,
        x, y + d,
        x - d, y,
    )
    return (
        '<polygon class="mark" data-arm="%s" points="%s" fill="%s" '
        'stroke="var(--surface)" stroke-width="2"/>' % (arm_id, pts, color)
    )


# Label placement. The SVG scales uniformly (fixed viewBox), so a layout that
# is collision-free in viewBox units is collision-free at every rendered
# width. Text width is estimated conservatively (12px sans, ~7px per char).
LABEL_CHAR_W = 7.0
LABEL_H = 14.0
LABEL_GAP = 10.0
LABEL_TIER = 18.0


def _label_box(tx, ty, anchor, text):
    w = LABEL_CHAR_W * len(text)
    x0 = tx if anchor == "start" else tx - w
    return (x0 - 2, ty - LABEL_H + 3, x0 + w + 2, ty + 4)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_labels(points):
    """Greedy, deterministic, collision-aware label placement.

    Points are labelled right to left. Each label tries, in order: right of
    its point, left of its point, then rising tiers on the preferred side
    with a leader line back to the point. A candidate is rejected if it
    overlaps an already placed label, any marker, or leaves the plot area.
    """
    obstacles = [(px - 8, py - 8, px + 8, py + 8) for _, px, py in points]
    hx0, hy0 = fx(HATCH_X_MIN), fy(Y_MAX)
    hx1 = fx(X_MAX)
    obstacles.append(((hx0 + hx1) / 2.0 - 110, hy0 + 3, (hx0 + hx1) / 2.0 + 110, hy0 + 20))
    placed = []
    order = sorted(points, key=lambda p: (-p[1], p[0]))
    for arm_id, px, py in order:
        text = ARM_LABEL[arm_id]
        prefer = "end" if px > fx(0.85) else "start"
        other = "start" if prefer == "end" else "end"
        cands = []
        for tier in range(0, 5):
            for anchor in (prefer, other):
                dx = LABEL_GAP if anchor == "start" else -LABEL_GAP
                cands.append((px + dx, py - 10 - tier * LABEL_TIER, anchor, tier))
        chosen = cands[0]
        for tx, ty, anchor, tier in cands:
            box = _label_box(tx, ty, anchor, text)
            if box[0] < PLOT_X0 or box[2] > PLOT_X1 or box[1] < PLOT_Y0:
                continue
            if any(_overlaps(box, o) for o in obstacles):
                continue
            chosen = (tx, ty, anchor, tier)
            break
        tx, ty, anchor, tier = chosen
        box = _label_box(tx, ty, anchor, text)
        obstacles.append(box)
        leader = None
        if tier > 0:
            lx = box[0] + 2 if anchor == "start" else box[2] - 2
            leader = (px, py - 7, lx, ty + 3)
        placed.append((arm_id, tx, ty, anchor, leader))
    rank = dict((a, i) for i, a in enumerate(ARM_ORDER))
    placed.sort(key=lambda r: rank[r[0]])
    return placed


def build_svg(rows):
    parts = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
        'role="img" aria-labelledby="bl-title bl-desc">' % (W, H)
    )
    parts.append(
        "<title id=\"bl-title\">Baseline arms: accuracy vs. adjacent substitution rate</title>"
    )
    parts.append(
        '<desc id="bl-desc">Scatter of the four scored baselines. No '
        "language model has been evaluated on this corpus yet; the "
        "upper-right region where a strong, low-error system would land "
        "has no points.</desc>"
    )
    parts.append(
        "<style>"
        ":root{color-scheme:light dark}"
        "svg{background:var(--surface,#FFFFFF)}"
        ".axis-line{stroke:var(--rule,#101311);stroke-width:1}"
        ".grid-line{stroke:var(--rule-soft,#D6D7CE);stroke-width:1}"
        ".tick-label{font:11px var(--font-mono,monospace);fill:var(--muted,#666D68)}"
        ".axis-label{font:12px var(--font-sans,sans-serif);fill:var(--ink-2,#3A403C)}"
        ".arm-label{font:12px var(--font-sans,sans-serif);fill:var(--ink,#101311)}"
        ".hatch-label{font:11px var(--font-mono,monospace);fill:var(--muted,#666D68)}"
        ".ci-bar{stroke:var(--ink-2,#3A403C);stroke-width:1.5}"
        ".leader{stroke:var(--muted,#666D68);stroke-width:1}"
        "@media (prefers-color-scheme: dark){"
        "svg{background:var(--surface,#12151C)}"
        ".axis-line{stroke:var(--rule,#EAECF2)}"
        ".grid-line{stroke:var(--rule-soft,#262B36)}"
        ".tick-label{fill:var(--muted,#838A99)}"
        ".axis-label{fill:var(--ink-2,#BFC4CF)}"
        ".arm-label{fill:var(--ink,#EAECF2)}"
        ".hatch-label{fill:var(--muted,#838A99)}"
        ".ci-bar{stroke:var(--ink-2,#BFC4CF)}"
        ".leader{stroke:var(--muted,#838A99)}"
        "}"
        "</style>"
    )

    # Background grid (x every 0.2, y every 0.1).
    for i in range(6):
        gx = X_MIN + i * 0.2
        px = fx(gx)
        parts.append(
            '<line class="grid-line" x1="%.2f" y1="%d" x2="%.2f" y2="%d"/>'
            % (px, PLOT_Y0, px, PLOT_Y1)
        )
        parts.append(
            '<text class="tick-label" x="%.2f" y="%d" text-anchor="middle">%s</text>'
            % (px, PLOT_Y1 + 18, fmt(gx))
        )
    for i in range(7):
        gy = Y_MIN + i * 0.1
        py = fy(gy)
        parts.append(
            '<line class="grid-line" x1="%d" y1="%.2f" x2="%d" y2="%.2f"/>'
            % (PLOT_X0, py, PLOT_X1, py)
        )
        parts.append(
            '<text class="tick-label" x="%d" y="%.2f" text-anchor="end">%s</text>'
            % (PLOT_X0 - 8, py + 4, fmt(gy))
        )

    # Axes.
    parts.append(
        '<line class="axis-line" x1="%d" y1="%d" x2="%d" y2="%d"/>'
        % (PLOT_X0, PLOT_Y1, PLOT_X1, PLOT_Y1)
    )
    parts.append(
        '<line class="axis-line" x1="%d" y1="%d" x2="%d" y2="%d"/>'
        % (PLOT_X0, PLOT_Y0, PLOT_X0, PLOT_Y1)
    )
    parts.append(
        '<text class="axis-label" x="%.2f" y="%d" text-anchor="middle">Accuracy</text>'
        % ((PLOT_X0 + PLOT_X1) / 2.0, H - 12)
    )
    parts.append(
        '<text class="axis-label" x="-%.2f" y="16" text-anchor="middle" '
        'transform="rotate(-90)">Adjacent substitution rate</text>'
        % ((PLOT_Y0 + PLOT_Y1) / 2.0)
    )

    # Hatch region: "no language model scored yet".
    hx0, hy0 = fx(HATCH_X_MIN), fy(Y_MAX)
    hx1, hy1 = fx(X_MAX), fy(HATCH_Y_MIN)
    parts.append(
        '<pattern id="noLmHatch" patternUnits="userSpaceOnUse" width="8" '
        'height="8" patternTransform="rotate(45)">'
        '<rect width="8" height="8" fill="none"/>'
        '<line x1="0" y1="0" x2="0" y2="8" stroke="var(--muted,#666D68)" '
        'stroke-width="1.5" opacity="0.55"/>'
        "</pattern>"
    )
    parts.append(
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
        'fill="url(#noLmHatch)" stroke="var(--rule-soft,#D6D7CE)" '
        'stroke-dasharray="3,3"/>'
        % (hx0, hy0, hx1 - hx0, hy1 - hy0)
    )
    parts.append(
        '<text class="hatch-label" x="%.2f" y="%.2f" text-anchor="middle">'
        "no language model scored yet</text>"
        % ((hx0 + hx1) / 2.0, hy0 + 16)
    )

    # Data points, in fixed arm order.
    points = []
    for arm_id in ARM_ORDER:
        row = rows[arm_id]
        ax = float(row["accuracy"])
        ay = float(row["adjacent_substitution_rate"])
        ax_lo = float(row["accuracy_ci_low"])
        ax_hi = float(row["accuracy_ci_high"])
        ay_lo = float(row["adjacent_ci_low"])
        ay_hi = float(row["adjacent_ci_high"])

        px, py = fx(ax), fy(ay)
        color_l = ARM_COLOR_LIGHT[arm_id]

        # Horizontal (accuracy) CI bar.
        parts.append(
            '<line class="ci-bar" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (fx(ax_lo), py, fx(ax_hi), py)
        )
        # Vertical (adjacent-rate) CI bar.
        parts.append(
            '<line class="ci-bar" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (px, fy(ay_lo), px, fy(ay_hi))
        )

        parts.append(marker(ARM_SHAPE[arm_id], px, py, color_l, arm_id))

        points.append((arm_id, px, py))

    # Labels, placed after every marker is known so they can avoid each
    # other (see _place_labels). Leader lines are drawn first so the text
    # sits on top of them.
    for arm_id, tx, ty, anchor, leader in _place_labels(points):
        if leader:
            parts.append(
                '<line class="leader" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
                % leader
            )
        parts.append(
            '<text class="arm-label" x="%.2f" y="%.2f" text-anchor="%s">%s</text>'
            % (tx, ty, anchor, ARM_LABEL[arm_id])
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def generate():
    header, rows = read_rows()
    svg = build_svg(rows)
    return svg


def write():
    svg = generate()
    with open(SVG_PATH, "w") as f:
        f.write(svg)
    return SVG_PATH


if __name__ == "__main__":
    path = write()
    print("wrote %s" % path)
