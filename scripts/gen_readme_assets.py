#!/usr/bin/env python3
"""Regenerate the README artwork in docs/assets/ (light + dark variants of each SVG).

    python3 scripts/gen_readme_assets.py

The terminal recording (docs/assets/demo.svg) is not generated here; see docs/assets/README.md.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

THEMES = {
    "light": {
        "bg0": "#ffffff",
        "bg1": "#f3f1ff",
        "card": "#ffffff",
        "card_alt": "#f6f8fa",
        "border": "#d0d7de",
        "fg": "#1f2328",
        "muted": "#59636e",
        "accent": "#5b4cf0",
        "accent_soft": "#ebe8ff",
        "ok": "#1a7f37",
        "ok_soft": "#dafbe1",
        "bad": "#cf222e",
        "bad_soft": "#ffebe9",
        "unk": "#6e7781",
        "unk_soft": "#eaeef2",
        "warn": "#9a6700",
        "warn_soft": "#fff8c5",
        "shadow": "#1f2328",
        "shadow_op": "0.08",
    },
    "dark": {
        "bg0": "#0d1117",
        "bg1": "#1a1640",
        "card": "#161b22",
        "card_alt": "#0d1117",
        "border": "#30363d",
        "fg": "#e6edf3",
        "muted": "#9198a1",
        "accent": "#9d8cff",
        "accent_soft": "#2a2459",
        "ok": "#3fb950",
        "ok_soft": "#12261e",
        "bad": "#f85149",
        "bad_soft": "#2d1416",
        "unk": "#9198a1",
        "unk_soft": "#21262d",
        "warn": "#d29922",
        "warn_soft": "#2b2111",
        "shadow": "#000000",
        "shadow_op": "0.5",
    },
}


def text(x, y, s, *, size, fill, weight=400, family=SANS, anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(s)}</text>'
    )


def chip(x, y, label, *, fg, bg, size=13, family=MONO, pad=10, h=26, weight=600):
    w = int(len(label) * size * (0.62 if family == MONO else 0.58)) + 2 * pad
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{bg}" stroke="{fg}" '
        f'stroke-opacity="0.35"/>'
        + text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, fill=fg, weight=weight, family=family,
               anchor="middle")
    ), w


def logo_mark(cx, cy, r, t):
    """Open ring (the controller loop) with three verdict dots leaving through the gap."""
    import math

    a0, a1 = math.radians(38), math.radians(360 - 38)
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    ring = (
        f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 1 1 {x1:.1f} {y1:.1f}" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="{r * 0.26:.1f}" stroke-linecap="round"/>'
    )
    dots = "".join(
        f'<circle cx="{cx + r * 1.12:.1f}" cy="{cy + dy * r:.1f}" r="{r * 0.15:.1f}" fill="{c}"/>'
        for dy, c in ((-0.42, t["ok"]), (0.0, t["bad"]), (0.42, t["unk"]))
    )
    core = f'<circle cx="{cx}" cy="{cy}" r="{r * 0.22:.1f}" fill="{t["accent"]}" opacity="0.9"/>'
    return ring + core + dots


def banner(t):
    W, H = 1280, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="Conjectra: hypotheses in, audited evidence out">',
        "<defs>",
        f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["bg0"]}"/>'
        f'<stop offset="1" stop-color="{t["bg1"]}"/></linearGradient>',
        f'<filter id="sh" x="-10%" y="-10%" width="120%" height="140%"><feDropShadow dx="0" dy="6" '
        f'stdDeviation="10" flood-color="{t["shadow"]}" flood-opacity="{t["shadow_op"]}"/></filter>',
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="20" fill="url(#bg)" stroke="{t["border"]}"/>',
        logo_mark(118, 150, 52, t),
        text(206, 150, "Conjectra", size=70, fill=t["fg"], weight=700, extra='letter-spacing="-1.5"'),
        text(210, 196, "Hypotheses in, audited evidence out.", size=27, fill=t["accent"], weight=600),
        text(210, 232, "A zero-token control plane for multi-goal autonomous research.", size=18,
             fill=t["muted"]),
    ]
    x = 210
    for label in ("0-token controller", "many goals, one pool", "fenced human control", "< 3 KB hand-off"):
        c, w = chip(x, 272, label, fg=t["accent"], bg=t["accent_soft"], size=12, family=SANS, weight=600)
        parts.append(c)
        x += w + 8

    # Evidence receipt card: the real three-verdict schema of an EvidenceReceipt.
    cx, cy, cw, ch = 868, 50, 352, 240
    parts.append(f'<g filter="url(#sh)"><rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="14" '
                 f'fill="{t["card"]}" stroke="{t["border"]}"/></g>')
    parts.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="40" rx="14" fill="{t["card_alt"]}"/>')
    parts.append(f'<rect x="{cx}" y="{cy + 26}" width="{cw}" height="14" fill="{t["card_alt"]}"/>')
    parts.append(f'<line x1="{cx}" y1="{cy + 40}" x2="{cx + cw}" y2="{cy + 40}" stroke="{t["border"]}"/>')
    parts.append(text(cx + 18, cy + 26, "EvidenceReceipt", size=15, fill=t["fg"], weight=700, family=MONO))
    parts.append(text(cx + cw - 18, cy + 26, "EXP_CONFIRM", size=13, fill=t["muted"], family=MONO,
                      anchor="end"))
    rows = [
        ("engineering", "PASS", t["ok"], t["ok_soft"]),
        ("scientific", "CONFIRMED", t["ok"], t["ok_soft"]),
        ("audit", "PASS", t["ok"], t["ok_soft"]),
        ("failure_domain", "NONE", t["unk"], t["unk_soft"]),
    ]
    for i, (k, v, fg, bg) in enumerate(rows):
        y = cy + 62 + i * 40
        parts.append(text(cx + 18, y + 18, k, size=14, fill=t["muted"], family=MONO))
        c, w = chip(cx + cw - 18 - (len(v) * 13 * 0.62 + 20), y, v, fg=fg, bg=bg, size=13)
        parts.append(c)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def box(x, y, w, h, t, *, stroke=None, fill=None, rx=14, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or t["card"]}" '
            f'stroke="{stroke or t["border"]}" stroke-width="1.5"{d}/>')


def arrow(x1, y1, x2, y2, t, *, label="", lx=None, ly=None, anchor="middle", color=None):
    c = color or t["muted"]
    s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2" '
         f'marker-end="url(#arr-{"a" if color else "m"})"/>')
    if label:
        s += text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 8, label,
                  size=13, fill=t["muted"], anchor=anchor)
    return s


def how_it_works(t):
    W, H = 1280, 600
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="How Conjectra works: a zero-token controller dispatches single-attempt jobs and '
        f'records every result in an append-only evidence ledger">',
        "<defs>",
        f'<marker id="arr-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker>',
        f'<marker id="arr-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["accent"]}"/></marker>',
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="20" fill="{t["bg0"]}" stroke="{t["border"]}"/>',
    ]

    # 1. You / an agent
    x, y, w, h = 40, 60, 230, 250
    p.append(box(x, y, w, h, t))
    p.append(text(x + 20, y + 36, "You, or an agent", size=18, fill=t["fg"], weight=700))
    p.append(text(x + 20, y + 60, "the owner of the question", size=13, fill=t["muted"]))
    for i, cmd in enumerate(("submit-goal", "intervene", "overview", "inspect", "takeover")):
        p.append(text(x + 20, y + 100 + i * 30, "$ conjectra " + cmd, size=14, fill=t["fg"], family=MONO))

    # 2. Controller
    x, y, w, h = 340, 40, 510, 290
    p.append(box(x, y, w, h, t, stroke=t["accent"], fill=t["accent_soft"]))
    p.append(text(x + 24, y + 40, "Controller", size=22, fill=t["fg"], weight=700))
    c, _ = chip(x + w - 24 - 150, y + 20, "0 LLM tokens", fg=t["accent"], bg=t["card"], size=14)
    p.append(c)
    p.append(text(x + 24, y + 66, "one deterministic daemon · decides in milliseconds · never calls a model",
                  size=13, fill=t["muted"]))
    steps = ["propose", "fence", "lease", "dispatch", "reconcile"]
    sx = x + 24
    for i, s in enumerate(steps):
        c, cw = chip(sx, y + 92, s, fg=t["fg"], bg=t["card"], size=13, family=MONO, h=32, weight=500, pad=9)
        p.append(c)
        if i < len(steps) - 1:
            p.append(arrow(sx + cw + 2, y + 108, sx + cw + 14, y + 108, t))
        sx += cw + 16
    lines = [
        "Fair-shares one worker pool across many goals",
        "Rejects late results with a stale goal_version / human_epoch",
        "Spawns ablation + falsification for every confirmed result",
        "Sends refuted routes to a graveyard that blocks repeats",
    ]
    for i, s in enumerate(lines):
        p.append(f'<circle cx="{x + 30}" cy="{y + 165 + i * 28}" r="3.5" fill="{t["accent"]}"/>')
        p.append(text(x + 44, y + 170 + i * 28, s, size=14, fill=t["fg"]))

    # 3. Jobs
    x, y, w, h = 920, 60, 320, 250
    p.append(box(x, y, w, h, t))
    p.append(text(x + 20, y + 36, "Single-attempt jobs", size=18, fill=t["fg"], weight=700))
    p.append(text(x + 20, y + 60, "isolated git worktree · pinned model · timeout", size=13, fill=t["muted"]))
    stages = [("CODE", "fg"), ("CPU_GATE", "warn"), ("IMPL_REVIEW", "accent"), ("EXP_SCREEN", "warn"),
              ("EXP_CONFIRM", "warn"), ("RESULT_REVIEW", "accent")]
    cx0, cy0 = x + 20, y + 80
    for i, (s, k) in enumerate(stages):
        fg = t[k]
        bg = t["card_alt"] if k == "fg" else t[k + "_soft"]
        c, cw = chip(cx0, cy0, s, fg=fg, bg=bg, size=12, h=24)
        p.append(c)
        cx0 += cw + 8
        if cx0 > x + w - 120:
            cx0, cy0 = x + 20, cy0 + 32
    p.append(text(x + 20, y + 200, "Backends: Claude Code · Codex · Gemini ·", size=13, fill=t["fg"]))
    p.append(text(x + 20, y + 220, "OpenAI-compatible · any CLI · your plugin", size=13, fill=t["fg"]))
    p.append(text(x + 20, y + 240, "Reviews go to a different model family", size=13, fill=t["accent"],
                  weight=600))

    # 4. Ledger
    x, y, w, h = 40, 400, 1200, 160
    p.append(box(x, y, w, h, t))
    p.append(text(x + 24, y + 38, "Append-only evidence ledger", size=20, fill=t["fg"], weight=700))
    p.append(text(x + 24, y + 62, "SQLite WAL · one EvidenceReceipt per job · engineering / scientific / audit "
                  "verdicts kept apart", size=13, fill=t["muted"]))
    vx = x + 24
    for label, k in (("CONFIRMED", "ok"), ("REFUTED", "bad"), ("UNKNOWN", "unk")):
        c, cw = chip(vx, y + 84, label, fg=t[k], bg=t[k + "_soft"], size=14, h=30)
        p.append(c)
        vx += cw + 12
    p.append(text(vx + 10, y + 104, "← scientific verdicts", size=13, fill=t["muted"]))
    p.append(text(x + 24, y + 140, "INFRA_TRANSIENT · ENV_CONFIG_MISMATCH · ENGINEERING_DEFECT are failure "
                  "domains, never recorded as REFUTED; passing tests alone never counts as CONFIRMED.",
                  size=13, fill=t["fg"]))

    # arrows
    p.append(arrow(270, 150, 338, 150, t, label="atomic tx", ly=140, color=t["accent"]))
    p.append(text(304, 172, "+ wake", size=13, fill=t["muted"], anchor="middle"))
    p.append(arrow(850, 150, 918, 150, t, label="dispatch", ly=140, color=t["accent"]))
    p.append(arrow(1080, 310, 1080, 398, t, label="receipt", lx=1090, ly=360, anchor="start"))
    p.append(arrow(595, 398, 595, 332, t, label="reconcile", lx=605, ly=370, anchor="start"))
    p.append(arrow(155, 398, 155, 312, t, label="overview < 50 ms · takeover < 3 KB", lx=168, ly=362,
                   anchor="start"))
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (OUT / f"banner-{name}.svg").write_text(banner(t), encoding="utf-8")
        (OUT / f"how-it-works-{name}.svg").write_text(how_it_works(t), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
