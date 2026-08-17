"""Hand-authored SVG charts for the ECCV poster.

Every number here is taken from the plotting scripts under scripts/ that produced
the paper figures; nothing is invented. Sources are noted per function.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# UNC Charlotte brand palette only.
#   Charlotte Green #005035 · Niner Gold #A49665 · Quartz White #FFFFFF
#   Jasper #F1E6B2 · Pine Green #899064 · Clay Red #802F2D
#   Sky Blue #007377 · Ore Black #101820
INK = "#101820"       # Ore Black - default for ALL chart text
MUTED = "#3D4642"     # dark neutral, axis ticks only
GRID = "#D7DED9"
ACTION = "#007377"    # action stream  = Sky Blue
IDENTITY = "#802F2D"  # identity stream = Clay Red
GATE = "#7D6F3C"      # fusion gate     = Niner Gold, darkened for contrast
GOLD = "#A49665"      # Niner Gold, fills only
OURS = "#005035"      # Charlotte Green

METHOD_COLOR = {
    "Raw Skeleton": "#7C8781",
    "Gaussian Noise": "#A9B3AD",
    "DMR": "#8A7A45",
    "PMR": IDENTITY,
    "Ours": OURS,
}


def _txt(x, y, s, size=13, anchor="start", fill=INK, weight="400",
         style="normal", family="ui-sans-serif, -apple-system, Segoe UI, sans-serif"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}" '
            f'font-family="{family}">{s}</text>')


# ---------------------------------------------------------------------------
# 1. Privacy-utility scatter  (source: scripts/plot_privacy_utility_scatter.py)
# ---------------------------------------------------------------------------
SCATTER_DATA = {
    # name: (sgn_ar, sgn_ri, mix_ar, mix_ri)
    "Raw Skeleton":   (89.1, 75.4, 88.6, 72.8),
    "Gaussian Noise": (80.3, 67.8, 83.6, 70.7),
    "DMR":            (43.1, 38.1, 45.3, 38.7),
    "PMR":            (19.9, 24.2, 20.8, 25.5),
    "Ours":           (55.4, 12.2, 55.7, 12.5),
}
OURS_ERR = {"sgn_ar": 0.7, "sgn_ri": 0.2, "mix_ar": 1.1, "mix_ri": 0.4}

LABEL_OFFSET = {
    "Raw Skeleton":   (0, -22, "middle"),
    "Gaussian Noise": (0, -22, "middle"),
    "DMR":            (14, 4, "start"),
    "PMR":            (0, 26, "middle"),
    "Ours":           (16, 2, "start"),
}


def privacy_utility_scatter(w=860, h=700):
    ml, mr, mt, mb = 104, 30, 34, 92
    pw, ph = w - ml - mr, h - mt - mb
    x0, x1 = 10.0, 95.0     # AR %
    y0, y1 = 0.0, 82.0      # RI %

    def X(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return mt + ph - (v - y0) / (y1 - y0) * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    s.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    # "better" quadrant wash — bottom-right = high AR, low RI
    s.append(f'<rect x="{X(50):.1f}" y="{Y(30):.1f}" width="{X(x1)-X(50):.1f}" '
             f'height="{Y(y0)-Y(30):.1f}" fill="#005035" opacity="0.05"/>')
    s.append(_txt(X(92), Y(27), "better", 21, "end", "#005035", "600", "italic"))

    # grid + axes
    for v in range(20, 100, 20):
        s.append(f'<line x1="{X(v):.1f}" y1="{mt}" x2="{X(v):.1f}" y2="{mt+ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(_txt(X(v), mt + ph + 26, f"{v}", 22, "middle", MUTED))
    for v in range(0, 81, 20):
        s.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(_txt(ml - 12, Y(v) + 6, f"{v}", 22, "end", MUTED))

    # chance line
    s.append(f'<line x1="{ml}" y1="{Y(2.5):.1f}" x2="{ml+pw}" y2="{Y(2.5):.1f}" '
             f'stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="7 6"/>')
    s.append(_txt(ml + 8, Y(2.5) - 9, "chance re-ID (2.5%)", 18, "start", MUTED, style="italic"))

    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#3D4642" stroke-width="1.4"/>')

    # points
    for name, (sar, sri, mar, mri) in SCATTER_DATA.items():
        c = METHOD_COLOR[name]
        is_ours = name == "Ours"
        r = 15 if is_ours else 11
        if is_ours:
            for (ax, ay, ex, ey) in [(sar, sri, OURS_ERR["sgn_ar"], OURS_ERR["sgn_ri"]),
                                     (mar, mri, OURS_ERR["mix_ar"], OURS_ERR["mix_ri"])]:
                s.append(f'<line x1="{X(ax-ex):.1f}" y1="{Y(ay):.1f}" x2="{X(ax+ex):.1f}" '
                         f'y2="{Y(ay):.1f}" stroke="{c}" stroke-width="2.4"/>')
                s.append(f'<line x1="{X(ax):.1f}" y1="{Y(ay-ey):.1f}" x2="{X(ax):.1f}" '
                         f'y2="{Y(ay+ey):.1f}" stroke="{c}" stroke-width="2.4"/>')
        # SGN = circle
        s.append(f'<circle cx="{X(sar):.1f}" cy="{Y(sri):.1f}" r="{r}" fill="{c}" '
                 f'stroke="#ffffff" stroke-width="2.4"/>')
        # MixFormer = triangle
        tx, ty, t = X(mar), Y(mri), r + 2
        s.append(f'<polygon points="{tx:.1f},{ty-t:.1f} {tx-t*0.92:.1f},{ty+t*0.72:.1f} '
                 f'{tx+t*0.92:.1f},{ty+t*0.72:.1f}" fill="{c}" stroke="#ffffff" '
                 f'stroke-width="2.4"/>')

        dx, dy, anch = LABEL_OFFSET[name]
        lx, ly = (X((sar + mar) / 2) + dx, Y((sri + mri) / 2) + dy)
        s.append(_txt(lx, ly, name, 23 if is_ours else 20, anch,
                      INK if is_ours else "#101820", "700" if is_ours else "500"))

    # axis titles
    s.append(_txt(ml + pw / 2, h - 30, "Action recognition  (AR %)  → higher is better",
                  23, "middle", INK, "600"))
    s.append(f'<g transform="translate(26,{mt+ph/2}) rotate(-90)">'
             + _txt(0, 0, "Re-identification  (RI %)  ← lower is better", 20, "middle",
                    INK, "600") + "</g>")

    # legend
    lx, ly = ml + 16, mt + 22
    s.append(f'<rect x="{lx-10}" y="{ly-20}" width="286" height="72" rx="8" '
             f'fill="#ffffff" stroke="{GRID}" stroke-width="1.4" opacity="0.96"/>')
    s.append(f'<circle cx="{lx+8}" cy="{ly-1}" r="9" fill="#3D4642"/>')
    s.append(_txt(lx + 26, ly + 5, "SGN evaluator", 20, "start", "#101820"))
    s.append(f'<polygon points="{lx+8},{ly+16} {lx-1},{ly+31} {lx+17},{ly+31}" fill="#3D4642"/>')
    s.append(_txt(lx + 26, ly + 30, "MixFormer evaluator", 20, "start", "#101820"))

    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 2. Ablation bars  (source: scripts/plot_ablation_barchart.py — SGN, batch 32)
# ---------------------------------------------------------------------------
ABLATIONS = [
    ("Full model", 45.3, 13.6),
    ("Static identity", 46.6, 14.6),
    ("− Adversarial", 44.9, 14.8),
    ("Full-seq identity", 44.5, 12.9),
    ("− Orthogonality", 43.5, 13.2),
    ("− Action backbone", 40.2, 12.6),
    ("− Temporal convs", 40.0, 13.8),
]


def ablation_bars(w=860, h=556):
    ml, mr, mt, mb = 286, 82, 52, 56
    pw, ph = w - ml - mr, h - mt - mb
    row = ph / len(ABLATIONS)
    vmax = 50.0

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    s.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    for v in range(0, int(vmax) + 1, 10):
        x = ml + v / vmax * pw
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(_txt(x, mt + ph + 28, f"{v}", 21, "middle", MUTED))

    for i, (name, ar, ri) in enumerate(ABLATIONS):
        y = mt + i * row
        bh = row * 0.36
        gap = row * 0.08
        full = (i == 0)
        s.append(_txt(ml - 16, y + row / 2 + 6, name, 22, "end", INK,
                      "700" if full else "400"))
        # AR bar
        s.append(f'<rect x="{ml}" y="{y+row/2-bh-gap/2:.1f}" width="{ar/vmax*pw:.1f}" '
                 f'height="{bh:.1f}" rx="3" fill="{ACTION}" '
                 f'opacity="{1.0 if full else 0.88}"/>')
        s.append(_txt(ml + ar / vmax * pw + 10, y + row / 2 - gap / 2 - bh / 2 + 6,
                      f"{ar:.1f}", 20, "start", ACTION, "700" if full else "500"))
        # RI bar
        s.append(f'<rect x="{ml}" y="{y+row/2+gap/2:.1f}" width="{ri/vmax*pw:.1f}" '
                 f'height="{bh:.1f}" rx="3" fill="{IDENTITY}" '
                 f'opacity="{1.0 if full else 0.88}"/>')
        s.append(_txt(ml + ri / vmax * pw + 10, y + row / 2 + gap / 2 + bh / 2 + 6,
                      f"{ri:.1f}", 20, "start", IDENTITY, "700" if full else "500"))

    # PMR reference line for RI
    xr = ml + 24.2 / vmax * pw
    s.append(f'<line x1="{xr:.1f}" y1="{mt-6}" x2="{xr:.1f}" y2="{mt+ph+4}" '
             f'stroke="{IDENTITY}" stroke-width="1.8" stroke-dasharray="7 6" opacity="0.45"/>')
    s.append(_txt(xr + 8, mt - 12, "PMR re-ID 24.2%", 19, "start", IDENTITY, "600"))

    s.append(_txt(ml + pw / 2, h - 8, "percent (%)", 22, "middle", INK, "700"))
    s.append(f'<rect x="{ml}" y="{mt-34}" width="17" height="17" rx="3" fill="{ACTION}"/>')
    s.append(_txt(ml + 24, mt - 20, "AR ↑", 21, "start", ACTION, "700"))
    s.append(f'<rect x="{ml+108}" y="{mt-34}" width="17" height="17" rx="3" fill="{IDENTITY}"/>')
    s.append(_txt(ml + 132, mt - 20, "RI ↓", 21, "start", IDENTITY, "700"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 3. Beta trade-off  (source: scripts/plot_soft_retarget_tradeoff.py)
# ---------------------------------------------------------------------------
BETAS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70, 1.00]
FROZEN_AR = [89.1, 84.9, 84.8, 83.8, 82.7, 80.0, 76.8, 38.8, 42.8, 27.7, 13.0, 2.0]
FROZEN_RI = [75.4, 69.5, 61.4, 60.9, 52.0, 37.5, 17.3, 9.6, 14.3, 3.1, 8.2, 4.7]


def beta_tradeoff(w=860, h=516):
    ml, mr, mt, mb = 96, 36, 46, 88
    pw, ph = w - ml - mr, h - mt - mb

    def X(b):
        return ml + b * pw

    def Y(v):
        return mt + ph - v / 95.0 * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    s.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    for v in range(0, 91, 20):
        s.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(_txt(ml - 12, Y(v) + 6, f"{v}", 21, "end", MUTED))
    for b in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        s.append(_txt(X(b), mt + ph + 28, f"{b:.1f}", 21, "middle", MUTED))

    # sweet-spot band at beta = 0.2
    s.append(f'<rect x="{X(0.175):.1f}" y="{mt}" width="{X(0.225)-X(0.175):.1f}" '
             f'height="{ph}" fill="#A49665" opacity="0.16"/>')

    for vals, col, dash in [(FROZEN_AR, ACTION, ""), (FROZEN_RI, IDENTITY, "9 6")]:
        pts = " ".join(f"{X(b):.1f},{Y(v):.1f}" for b, v in zip(BETAS, vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="4" '
                 f'stroke-linejoin="round" stroke-linecap="round"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        for b, v in zip(BETAS, vals):
            s.append(f'<circle cx="{X(b):.1f}" cy="{Y(v):.1f}" r="5.5" fill="{col}" '
                     f'stroke="#ffffff" stroke-width="1.8"/>')

    # callout
    s.append(f'<circle cx="{X(0.2):.1f}" cy="{Y(76.8):.1f}" r="11" fill="none" '
             f'stroke="{GATE}" stroke-width="3.4"/>')
    s.append(f'<circle cx="{X(0.2):.1f}" cy="{Y(17.3):.1f}" r="11" fill="none" '
             f'stroke="{GATE}" stroke-width="3.4"/>')
    cy = mt + 0.30 * ph
    s.append(f'<rect x="{X(0.34):.1f}" y="{cy:.1f}" width="330" height="80" rx="8" '
             f'fill="#FBF7EA" stroke="{GATE}" stroke-width="1.8"/>')
    s.append(_txt(X(0.34) + 14, cy + 27, "β = 0.2 operating point", 22, "start", GATE, "700"))
    s.append(_txt(X(0.34) + 14, cy + 53, "76.8% AR   ·   17.3% RI", 22, "start", INK, "500"))

    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#3D4642" stroke-width="1.4"/>')
    s.append(_txt(ml + pw / 2, h - 28,
                  "retargeting strength  β   (0 = passthrough,  1 = full retarget)",
                  22, "middle", INK, "600"))
    s.append(f'<g transform="translate(26,{mt+ph/2}) rotate(-90)">'
             + _txt(0, 0, "AR / RI  (%)", 22, "middle", INK, "600") + "</g>")

    s.append(f'<line x1="{ml+pw-232}" y1="{mt+14}" x2="{ml+pw-196}" y2="{mt+14}" '
             f'stroke="{ACTION}" stroke-width="4"/>')
    s.append(_txt(ml + pw - 188, mt + 20, "AR ↑", 21, "start", ACTION, "700"))
    s.append(f'<line x1="{ml+pw-232}" y1="{mt+42}" x2="{ml+pw-196}" y2="{mt+42}" '
             f'stroke="{IDENTITY}" stroke-width="4" stroke-dasharray="9 6"/>')
    s.append(_txt(ml + pw - 188, mt + 48, "RI ↓", 21, "start", IDENTITY, "700"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 4. Architecture diagram
# ---------------------------------------------------------------------------
def architecture(w=980, h=640):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    s.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    s.append('<defs>'
             '<marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#101820"/></marker>'
             '<marker id="ahb" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             f'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{ACTION}"/></marker>'
             '<marker id="ahr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             f'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{IDENTITY}"/></marker>'
             '</defs>')

    def box(x, y, bw, bh, title, sub, fill, stroke, tcol=INK, ts=21, ss=17):
        g = [f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="2"/>']
        if sub:
            g.append(_txt(x + bw / 2, y + bh / 2 - 4, title, ts, "middle", tcol, "700"))
            g.append(_txt(x + bw / 2, y + bh / 2 + 19, sub, ss, "middle", MUTED))
        else:
            g.append(_txt(x + bw / 2, y + bh / 2 + 7, title, ts, "middle", tcol, "700"))
        return "".join(g)

    def arrow(x1, y1, x2, y2, col="#101820", mk="ah", wid=2.4, dash=""):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                f'stroke-width="{wid}" marker-end="url(#{mk})"'
                + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    # ---- action lane ----
    s.append(f'<rect x="14" y="14" width="700" height="222" rx="14" fill="#E8F2F2" '
             f'stroke="{ACTION}" stroke-width="2" stroke-dasharray="9 6" opacity="0.85"/>')
    s.append(_txt(30, 44, "ACTION ENCODER  ·  4.9M", 22, "start", ACTION, "800"))
    s.append(_txt(30, 68, "what is being done", 17, "start", ACTION, "500", "italic"))

    s.append(box(30, 84, 118, 74, "Source", "P₁ · action A", "#ffffff", "#7FBFC1"))
    s.append(box(168, 84, 128, 74, "pos + vel", "+ acceleration", "#ffffff", "#7FBFC1"))
    s.append(box(316, 84, 132, 74, "Multi-scale", "conv k=3,5,7", "#ffffff", "#7FBFC1"))
    s.append(box(468, 84, 128, 74, "Temporal", "attn × 2", "#ffffff", "#7FBFC1"))
    s.append(box(316, 170, 280, 52, "Skeleton-MixFormer backbone", "", "#D3E8E8", "#7FBFC1", ACTION, 19))
    s.append(box(616, 84, 82, 74, "gate", "g", "#f3efdf", GATE, GATE))

    for a, b in [(148, 168), (296, 316), (448, 468), (596, 616)]:
        s.append(arrow(a, 121, b, 121, ACTION, "ahb"))
    s.append(arrow(240, 158, 240, 196, ACTION, "ahb", 2.0, "6 5"))
    s.append(f'<line x1="240" y1="196" x2="316" y2="196" stroke="{ACTION}" stroke-width="2.0" stroke-dasharray="6 5"/>')
    s.append(f'<line x1="596" y1="196" x2="657" y2="196" stroke="{ACTION}" stroke-width="2.0"/>')
    s.append(arrow(657, 196, 657, 162, ACTION, "ahb", 2.0))

    # ---- identity lane ----
    s.append(f'<rect x="14" y="252" width="700" height="164" rx="14" fill="#F7EDEC" '
             f'stroke="{IDENTITY}" stroke-width="2" stroke-dasharray="9 6" opacity="0.85"/>')
    s.append(_txt(30, 282, "IDENTITY ENCODER  ·  0.8M", 22, "start", IDENTITY, "800"))
    s.append(_txt(30, 306, "who appears to be doing it: deliberately low capacity",
                  17, "start", IDENTITY, "500", "italic"))

    s.append(box(30, 320, 118, 74, "Target", "P₂ · any action", "#ffffff", "#C08E8C"))
    s.append(box(168, 320, 128, 74, "Static pose", "mean over T", "#ffffff", "#C08E8C"))
    s.append(box(316, 320, 132, 74, "Spatial GCN", "64→128→256", "#ffffff", "#C08E8C"))
    s.append(box(468, 320, 128, 74, "+ bone", "lengths MLP", "#ffffff", "#C08E8C"))
    s.append(box(616, 320, 82, 74, "fuse", "MLP", "#EFDEDC", IDENTITY, IDENTITY))
    for a, b in [(148, 168), (296, 316), (448, 468), (596, 616)]:
        s.append(arrow(a, 357, b, 357, IDENTITY, "ahr"))

    # ---- latents ----
    s.append(box(736, 84, 106, 74, "H_action", "T × 768", "#D3E8E8", ACTION, ACTION, 20))
    s.append(box(736, 320, 106, 74, "H_id", "256", "#EFDEDC", IDENTITY, IDENTITY, 20))
    s.append(arrow(698, 121, 736, 121, ACTION, "ahb", 3.2))
    s.append(arrow(698, 357, 736, 357, IDENTITY, "ahr", 3.2))

    # ---- decoder ----
    s.append(f'<rect x="14" y="436" width="828" height="176" rx="14" fill="#eef5f1" '
             f'stroke="{OURS}" stroke-width="2" stroke-dasharray="9 6" opacity="0.9"/>')
    s.append(_txt(30, 464, "FACTORIZED DECODER  ·  17.0M", 22, "start", OURS, "800"))
    s.append(_txt(30, 486, "6 layers · d=320 · causal, autoregressive",
                  16, "start", OURS, "500", "italic"))

    dy, dh = 508, 68
    bw, dgap, bx0 = 144, 20, 30
    dboxes = [
        ("Causal", "self-attn", "#ffffff", "#9dc3b3", INK),
        ("Cross-attn", "action", "#E8F2F2", ACTION, ACTION),
        ("Cross-attn", "identity", "#F7EDEC", IDENTITY, IDENTITY),
        ("Adaptive", "gate \u03b1", "#f3efdf", GATE, GATE),
        ("Retargeted", "output", "#dcebe4", OURS, OURS),
    ]
    xs = [bx0 + i * (bw + dgap) for i in range(5)]
    for x, (t, sub, bg, st, tc) in zip(xs, dboxes):
        s.append(box(x, dy, bw, dh, t, sub, bg, st, tc, 19, 16))
    for i in range(4):
        s.append(arrow(xs[i] + bw, dy + dh / 2, xs[i + 1], dy + dh / 2))

    # latents feed the two cross-attention blocks from above, clear of the title
    a_lane, i_lane = 478, 494
    s.append(f'<path d="M842,121 L906,121 L906,{a_lane} L{xs[1]+bw/2},{a_lane} '
             f'L{xs[1]+bw/2},{dy-4}" fill="none" stroke="{ACTION}" stroke-width="3" '
             f'marker-end="url(#ahb)"/>')
    s.append(f'<path d="M789,394 L789,{i_lane} L{xs[2]+bw/2},{i_lane} '
             f'L{xs[2]+bw/2},{dy-4}" fill="none" stroke="{IDENTITY}" stroke-width="3" '
             f'marker-end="url(#ahr)"/>')

    s.append(_txt((xs[3] + bw / 2), dy + dh + 26,
                  "\u03b1 \u00b7 Z_action + (1\u2212\u03b1) \u00b7 Z_identity",
                  17, "middle", GATE, "700", "italic"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 4b. Architecture diagram, wide  (source: assets/tmr_arch.jpg, the paper figure)
#
# A box-for-box redraw of the paper's architecture figure, including the
# post-hoc beta blend, in the poster palette: the draw.io export's pastel blue /
# rose / green does not sit with the UNC Charlotte marks, and its type lands
# near 6 pt once the figure is scaled to a poster column.
#
# The canvas is sized for a panel spanning two poster columns, so 1 unit is
# about 0.42 mm on the 140 x 100 cm board: font-size 21 reads as ~25 pt.
# ---------------------------------------------------------------------------
def architecture_wide(w=1620, h=626):
    GATE_LT, GATE_MD = "#F6F0DC", "#B79A4E"
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
         '<defs>'
         # userSpaceOnUse: the default scales the head by stroke-width, which
         # turns a 3.4-wide stream arrow into a blob over the next box.
         + "".join(
             f'<marker id="w{i}" markerUnits="userSpaceOnUse" markerWidth="17" '
             f'markerHeight="17" refX="15" refY="8.5" orient="auto">'
             f'<path d="M0.5,1 L16,8.5 L0.5,16 z" fill="{c}"/></marker>'
             for i, c in (("k", "#5A6560"), ("a", ACTION), ("i", IDENTITY),
                          ("o", OURS), ("g", GATE_MD)))
         + '</defs>']

    def SUB(tail, size=14):
        """A subscript run, with the baseline reset so the next glyph sits level."""
        return (f'<tspan font-size="{size}" dy="{size*0.30:.1f}">{tail}</tspan>'
                f'<tspan font-size="0" dy="{-size*0.30:.1f}">.</tspan>')

    def SUP(tail, size=14):
        return (f'<tspan font-size="{size}" dy="{-size*0.42:.1f}">{tail}</tspan>'
                f'<tspan font-size="0" dy="{size*0.42:.1f}">.</tspan>')

    def card(x, y, bw, bh, fill, stroke, rx=16, sw=2.6, dash=""):
        return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
                + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    def stack(x, y, lines, gap, size, col, weight="400", style="normal"):
        """Centred lines, first baseline at y."""
        return "".join(_txt(x, y + i * gap, t, size, "middle", col, weight, style)
                       for i, t in enumerate(lines))

    def arrow(pts, col="#5A6560", mk="wk", wid=2.8, dash=""):
        d = "M" + " L".join(f"{a},{b}" for a, b in pts)
        return (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wid}" '
                f'marker-end="url(#{mk})"'
                + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    R25 = "&#8477;" + SUP("25×64×3", 15)

    # ---- inputs ------------------------------------------------------------
    for y, title, who, sym in ((120, "Source", "Person A, Action X", "s" + SUB("src", 15)),
                               (400, "Target", "Person B", "s" + SUB("tgt", 15))):
        s.append(card(40, y, 270, 112, "#F5F7F6", "#8A948F", 14, 2.2))
        s.append(_txt(175, y + 40, title, 29, "middle", INK, "800"))
        s.append(_txt(175, y + 70, who, 20, "middle", MUTED, "400", "italic"))
        s.append(_txt(175, y + 98, sym + " &#8712; " + R25, 20, "middle", MUTED))

    # ---- encoders ----------------------------------------------------------
    s.append(card(390, 95, 340, 162, "#E4F1F1", ACTION, 18, 3.0))
    s.append(_txt(560, 140, "Action Encoder E" + SUB("A", 17), 30, "middle",
                  ACTION, "800"))
    s.append(stack(560, 178, ["Multi-scale Conv + Attention", "+ MixFormer Gate",
                              "4.9M params"], 28, 20, INK))

    s.append(card(390, 388, 340, 162, "#F8EBEA", IDENTITY, 18, 3.0))
    s.append(_txt(560, 433, "Identity Encoder E" + SUB("I", 17), 30, "middle",
                  IDENTITY, "800"))
    s.append(stack(560, 471, ["Spatial GCN + Bone MLP", "0.8M params"], 28, 20, INK))

    # ---- factorized decoder, drawn as a stack of six ------------------------
    for k in (2, 1):
        s.append(card(860 + k * 9, 170 + k * 9, 400, 352, "#EAF3EE",
                      "#A8CBBA", 28, 2.0))
    s.append(card(860, 170, 400, 352, "#E1EEE7", OURS, 28, 3.0))
    s.append(_txt(1060, 232, "Factorized Decoder D", 30, "middle", OURS, "800"))
    s.append(stack(1060, 276, ["Causal Self-Attn",
                               "+ Separate Cross-Attn (Action &amp; Identity)",
                               "+ Adaptive Gate α",
                               "6 layers, d" + SUB("D", 15) + " = 320",
                               "17.0M params"], 30, 20, INK))
    s.append(_txt(878, 502, "×6 layers", 20, "start", OURS, "700", "italic"))

    # ---- beta blend and output --------------------------------------------
    s.append(card(1348, 212, 232, 116, GATE_LT, GATE_MD, 16, 3.0))
    s.append(_txt(1464, 258, "β Blend", 30, "middle", GATE, "800"))
    s.append(_txt(1464, 296, "ŝ = β·ō&#160;+ (1−β)·s" + SUB("src", 14), 20,
                  "middle", INK))

    s.append(card(1330, 440, 270, 112, "#F5F7F6", "#8A948F", 14, 2.2))
    s.append(_txt(1465, 486, "Output ŝ", 29, "middle", INK, "800"))
    s.append(_txt(1465, 518, "Partially retargeted", 20, "middle", MUTED,
                  "400", "italic"))

    # ---- wiring ------------------------------------------------------------
    s.append(arrow([(310, 176), (390, 176)]))
    s.append(arrow([(310, 456), (390, 456)]))
    # each stream steps to the decoder at its own height, so nothing crosses
    s.append(arrow([(730, 176), (796, 176), (796, 262), (860, 262)],
                   ACTION, "wa", 3.4))
    s.append(arrow([(730, 469), (796, 469), (796, 430), (860, 430)],
                   IDENTITY, "wi", 3.4))
    # both stream labels clear the boxes: one above its elbow, one below
    s.append(_txt(742, 152, "H" + SUB("action", 15) + " &#8712; &#8477;"
                  + SUP("T×768", 15), 21, "start", ACTION, "600", "italic"))
    s.append(_txt(742, 588, "H" + SUB("identity", 15) + " &#8712; &#8477;"
                  + SUP("256", 15), 21, "start", IDENTITY, "600", "italic"))
    s.append(stack(700, 316, ["Information bottleneck:",
                              "768D action &#8811; 256D identity"], 28, 19,
                   MUTED, "500", "italic"))

    s.append(arrow([(1260, 270), (1348, 270)]))
    # off-centre so the feedback label to its left is not struck through
    s.append(arrow([(1555, 328), (1555, 440)]))
    # the raw source is held aside and mixed back in at test time
    s.append(arrow([(175, 120), (175, 44), (1464, 44), (1464, 212)],
                   GATE_MD, "wg", 2.8, "12 8"))
    s.append(_txt(820, 34, "(1−β) · source", 21, "middle", GATE, "700", "italic"))
    # autoregressive feedback: the emitted frame conditions the next step
    s.append(arrow([(1330, 496), (1294, 496), (1294, 460), (1260, 460)],
                   OURS, "wo", 2.6, "11 7"))
    s.append(stack(1398, 398, ["autoregressive", "feedback"], 26, 19, OURS,
                   "600", "italic"))

    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 5. Task schematic  (source → target identity → output)
# ---------------------------------------------------------------------------
def task_strip(src_svg, out_svg, cell):
    plus = _txt(0, 0, "+", 46, "middle", MUTED, "300")
    return plus, src_svg, out_svg, cell


# ---------------------------------------------------------------------------
# 6. Cross-identity quadruplet  (source: src/data/datasets.py :: Cross_Data)
#     x1 = (A,X)  x2 = (B,Y)  y1 = (A,Y)  y2 = (B,X)  -- person, action
#     model(x1, x2) is supervised against y2.
# ---------------------------------------------------------------------------
def quadruplet(w=660, h=428):
    """2x2 actor x action grid. Inputs route around the outside so no arrow
    crosses a cell; the supervision target drops straight down."""
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    s.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    s.append('<defs>'
             f'<marker id="qb" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             f'orient="auto"><path d="M0,0.5 L9,4.5 L0,8.5 z" fill="{ACTION}"/></marker>'
             f'<marker id="qr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             f'orient="auto"><path d="M0,0.5 L9,4.5 L0,8.5 z" fill="{IDENTITY}"/></marker>'
             '<marker id="qg" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             'orient="auto"><path d="M0,0.5 L9,4.5 L0,8.5 z" fill="#005035"/></marker>'
             '</defs>')

    cw, ch, gx, gy = 208, 96, 26, 18
    x0, y0 = 100, 58
    right = x0 + 2 * cw + gx
    lane_l, lane_r = 58, right + 28

    def cell(r, c):
        return x0 + c * (cw + gx), y0 + r * (ch + gy)

    # Person A/B and Action X/Y, matching the architecture figure. Numbered
    # subscripts here and lettered names there made two figures about the same
    # four sequences look like they were about different things.
    for c, lab in enumerate(["Action X", "Action Y"]):
        s.append(_txt(x0 + c * (cw + gx) + cw / 2, 42, lab, 22, "middle", INK, "700"))
    for r, lab in enumerate(["Person A", "Person B"]):
        cy = y0 + r * (ch + gy) + ch / 2
        s.append(f'<g transform="translate(30,{cy}) rotate(-90)">'
                 + _txt(0, 0, lab, 22, "middle", INK, "700") + "</g>")

    cells = [
        (0, 0, "x\u2081", "action source",      ACTION,    "#E8F2F2", True),
        (0, 1, "y\u2081", "not used",           "#3D4642", "#F4F8F6", False),
        (1, 0, "y\u2082", "supervision target", "#005035", "#EDF4F1", True),
        (1, 1, "x\u2082", "identity source",    IDENTITY,  "#F7EDEC", True),
    ]
    for r, c, tag, sub, col, bg, strong in cells:
        x, y = cell(r, c)
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="{bg}" '
                 f'stroke="{col}" stroke-width="{2.6 if strong else 1.6}"'
                 + ("" if strong else ' stroke-dasharray="7 5"') + "/>")
        s.append(_txt(x + cw / 2, y + 46, tag, 33, "middle", col, "800"))
        s.append(_txt(x + cw / 2, y + 76, sub, 18, "middle",
                      col if strong else "#3D4642", "700" if strong else "400"))

    my, mh = y0 + 2 * ch + gy + 54, 62
    mx0, mx1 = 150, right - 46
    s.append(f'<rect x="{mx0}" y="{my}" width="{mx1-mx0}" height="{mh}" rx="10" '
             f'fill="#dcebe4" stroke="{OURS}" stroke-width="2.6"/>')
    s.append(_txt((mx0 + mx1) / 2, my + 41, "DisentangledTMR", 26, "middle", OURS, "800"))

    ymid = my + mh / 2
    # x1 routes down the OUTSIDE left, x2 down the outside right
    ax, ay = cell(0, 0)[0], cell(0, 0)[1] + ch / 2
    s.append(f'<path d="M{ax},{ay} L{lane_l},{ay} L{lane_l},{ymid} L{mx0-4},{ymid}" '
             f'fill="none" stroke="{ACTION}" stroke-width="2.8" marker-end="url(#qb)"/>')
    bx, by = cell(1, 1)[0] + cw, cell(1, 1)[1] + ch / 2
    s.append(f'<path d="M{bx},{by} L{lane_r},{by} L{lane_r},{ymid} L{mx1+4},{ymid}" '
             f'fill="none" stroke="{IDENTITY}" stroke-width="2.8" marker-end="url(#qr)"/>')
    # y2 drops straight into the model as the reconstruction target
    yx = cell(1, 0)[0] + cw / 2
    s.append(f'<path d="M{yx},{cell(1,0)[1]+ch} L{yx},{my-6}" fill="none" '
             f'stroke="#005035" stroke-width="2.8" stroke-dasharray="9 6" '
             f'marker-end="url(#qg)"/>')
    s.append(_txt(yx + 18, my - 24, "reconstruction target", 17, "start", "#005035", "700"))

    s.append(_txt(w / 2, h - 13,
                  "same action, different body, so the target already exists",
                  18, "middle", MUTED, "600", "italic"))
    s.append("</svg>")
    return "".join(s)
