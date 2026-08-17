#!/usr/bin/env python3
"""Build the ECCV 2026 poster as a single self-contained HTML file.

    python poster/build_poster.py                # the ECCV board
    python poster/build_poster.py a0 36x48       # other sizes, by name

Sizes:
    140x100  1400 x 1000 mm   landscape — the ECCV 2026 board, and the only
                              size built by default: ECCV ships 140 x 100 cm
                              (55.12 x 39.37 in) landscape for both the main
                              conference and the workshops
    180x90   1800 x  900 mm   landscape — earlier draft board size
    a0        841 x 1189 mm   A0 portrait
    36x48     914 x 1219 mm   36 x 48 in portrait

Theming follows the UNC Charlotte brand (Niner Green #005035 / gold #A49665).

Every figure is either (a) inline SVG drawn from numbers in scripts/plot_*.py, or
(b) a real skeleton sequence from demo_data.json, or (c) a real generated PNG from
assets/. Nothing is mocked.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

import charts
import logos
from skeleton_svg import filmstrip, load_examples, sequence_bounds
from skeleton_svg import transform as _t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = ROOT / "assets"

PROJECT_URL = "tmr.thomasc.tech"

# Print-ready sheet geometry, per the ECCV 2026 printer's graphic guidelines:
# 5-10 mm bleed beyond the trim line, plus crop marks.
BLEED_MM = 5.0        # artwork extends this far past the trim line
MARK_MM = 10.0        # length of each crop mark, drawn outside the bleed
MARK_W_MM = 0.25      # crop-mark hairline width
MIN_IMAGE_DPI = 100.0  # the printer rejects raster art below this at 1:1

# UNC Charlotte brand
GREEN = "#005035"       # Niner Green
GREEN_DK = "#00291b"
GREEN_MD = "#046d49"
GREEN_LT = "#0a8659"
GOLD = "#A49665"       # Niner Gold
JASPER = "#F1E6B2"
PINE = "#899064"
CLAY = "#802F2D"
SKY = "#007377"
ORE_BLACK = "#101820"

# page presets: width mm, height mm, columns, type scale, header side width mm,
# label, landscape-masthead flag
PAGES = {
    # The board ECCV 2026 actually ships: 140 x 100 cm landscape. 4 columns keeps
    # each one near A0 column width, so the full-width figures stay the same
    # physical size instead of scaling up with the board.
    "140x100": (1400.0, 1000.0, 4, 1.14, 300.0,
                "ECCV landscape  1400 x 1000 mm (140 x 100 cm)", True),
    "180x90": (1800.0, 900.0, 5, 1.36, 355.0,
               "ECCV landscape  1800 x 900 mm (180 x 90 cm)", True),
    "a0":     (841.0, 1189.0, 3, 1.00, 190.0, "A0 portrait  841 x 1189 mm", False),
    "36x48":  (914.4, 1219.2, 3, 1.02, 205.0,
               "36 x 48 in portrait  914 x 1219 mm", False),
}


class SPAN2(NamedTuple):
    """A block two grid columns wide: `head` runs full width, then two sub-columns.

    Lets one panel (the architecture) break out of the column grid without
    turning the whole body into free-flowing masonry, which would scramble the
    reading order.
    """
    head: list[str]
    left: list[str]
    right: list[str]


# Which sections land in which column, per column count.
COLUMN_PLANS = {
    3: [["problem", "idea", "signal", "arch", "stages"],
        ["qual", "tsne", "results"],
        ["scatter", "ablation", "beta", "notes", "take"]],
    # Reading order runs down each column and across: setup -> model -> does it
    # work -> where it sits and how to tune it. Balanced against measured
    # section heights (balance_columns.py) so no column overflows or runs short.
    #
    # The middle entry is a two-column block: the architecture panel spans both
    # so the paper's wide diagram prints at ~680 mm and its labels land near
    # 20 pt, then the two sub-columns pick the reading order back up.
    4: [["problem", "idea", "signal", "stages", "notes"],
        SPAN2(head=["arch"], left=["qual"], right=["tsne", "scatter"]),
        ["results", "ablation", "beta", "take"]],
    # Thematic rather than strictly sequential: setup -> model -> does it work
    # -> where it sits -> how to tune it. Balanced against measured heights.
    # "losses" only fits the wide landscape board; the portrait variants are
    # already full without it.
    # "losses" is built but not placed: a 14-row table of weights reads as
    # clutter at poster distance and belongs in the paper.
    5: [["problem", "idea", "signal"],
        ["arch", "stages", "notes"],
        ["qual", "tsne"],
        ["scatter", "ablation"],
        ["results", "beta", "take"]],
}

# ECCV 2026 ships one board for main-conference and workshop posters:
# 140 x 100 cm (55.12 x 39.37 in), landscape. That is the only size we build by
# name; the rest stay in PAGES because the layout code still supports them, and
# their last builds are under archive/other-board-sizes/.
DEFAULT_PAGES = ["140x100"]

OUT_NAME = {"140x100": "eccv2026_poster_140x100.html",
            "180x90": "eccv2026_poster_180x90.html",
            "a0": "eccv2026_poster_A0.html",
            "36x48": "eccv2026_poster_36x48.html"}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def step_gradient(w: int, h: int, stops: list[tuple[float, str]],
                  skew: float = 0.55, n: int = 64) -> str:
    """A diagonal gradient drawn as n flat-filled vector slices.

    Chrome's print pipeline rasterises every real gradient, CSS or SVG, at a
    small fraction of print resolution — a 1366 mm masthead came back as a
    605 px bitmap. Flat fills survive as vectors, and at 64 slices the steps are
    about 1% of lightness apart, well below what offset printing resolves.
    """
    def rgb(c: str) -> tuple[int, int, int]:
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))

    def at(t: float) -> str:
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1 or (t1, c1) == stops[-1]:
                k = 0.0 if t1 == t0 else (min(max(t, t0), t1) - t0) / (t1 - t0)
                a, b = rgb(c0), rgb(c1)
                return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * k):02x}"
                                     for i in range(3))
        return stops[-1][1]

    dx = skew * h                      # horizontal run of the diagonal edge
    span = w + dx
    out = [f'<svg class="bg" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
           f'aria-hidden="true"><rect width="{w}" height="{h}" '
           f'fill="{stops[0][1]}"/>']
    for i in range(n):
        x0, x1 = -dx + span * i / n, -dx + span * (i + 1) / n
        # one slice wider than its share, so rounding never opens a hairline gap
        out.append(f'<polygon points="{x0 + dx:.2f},0 {x1 + dx + 0.6:.2f},0 '
                   f'{x1 + 0.6:.2f},{h} {x0:.2f},{h}" '
                   f'fill="{at((i + 0.5) / n)}"/>')
    out.append("</svg>")
    return "".join(out)


def qr_chip() -> str:
    """A white chip holding the project QR, or nothing if no code is present.

    Drop a code into poster/ under any of the names below and the next build
    picks it up. SVG is preferred: the printer's 100 DPI floor applies to raster
    art, and a QR is the one image where a soft edge costs you scans.
    """
    for name in ("qr.svg", "tmr_qr.svg", "qr.png", "tmr_qr.png",
                 "adobe-express-qr-code.png"):
        f = HERE / name
        if not f.exists():
            continue
        art = (f.read_text(encoding="utf-8") if f.suffix == ".svg"
               else f'<img src="data:image/png;base64,{b64(f)}" alt="{PROJECT_URL}">')
        return f'<div class="qr">{art}</div>'
    return ""


def png_width(path: Path) -> int:
    """Pixel width straight from the PNG header (no image library needed)."""
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return int.from_bytes(head[16:20], "big")


def scale_css(css: str, k: float) -> str:
    """Multiply every pt/mm length in the stylesheet by k.

    Bigger boards want proportionally bigger type, not more words per line. Hex
    colours and px hairlines are left alone because the pattern only fires on a
    number immediately followed by `pt` or `mm`.
    """
    if abs(k - 1.0) < 1e-9:
        return css

    def rep(m: re.Match) -> str:
        v = f"{float(m.group(1)) * k:.3f}".rstrip("0").rstrip(".")
        return f"{v}{m.group(2)}"

    return re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)(pt|mm)\b", rep, css)


# ---------------------------------------------------------------------------
# Qualitative panel: real skeletons from demo_data.json
# ---------------------------------------------------------------------------
def qualitative_panel(n_frames: int = 4) -> str:
    d = load_examples()
    bones = d["bones"]
    ex = {e["action_name"]: e for e in d["examples"]}

    # Chosen because their raw NTU captures are clean and the motion reads at a
    # distance (verified by rendering all ten candidates first).
    picks = ["salute", "jump up"]
    cell = 140
    rows = [("Source", "source", ORE_BLACK, False),
            ("Ours", "ours", GREEN, False),
            ("DMR", "dmr", "#8A7A45", False),      # Niner Gold, darkened
            ("PMR", "pmr", CLAY, False)]

    lab_w = 92
    hdr = 24

    def pick_frames(src: np.ndarray, n: int = 4) -> list[int]:
        """Evenly spread frames, drawn only from well-tracked ones.

        Raw NTU has occasional frames where the Kinect loses the legs; those
        render as a stunted skeleton and read as a bug rather than as data.
        We keep frames whose vertical extent is close to the sequence median.
        """
        span = np.array([float(np.ptp(_t(src[t])[:, 1])) for t in range(src.shape[0])])
        med = float(np.median(span))
        ok = [t for t in range(6, src.shape[0] - 3) if span[t] > 0.88 * med]
        if len(ok) < n:
            ok = list(range(6, src.shape[0] - 3))
        return [ok[round(i * (len(ok) - 1) / (n - 1))] for i in range(n)]

    blocks = []
    for name in picks:
        e = ex[name]
        seqs = {k: np.array(e[k]) for _, k, _, _ in rows}
        frames = pick_frames(seqs["source"], n_frames)
        strip_w = len(frames) * cell
        total_w = lab_w + strip_w
        total_h = len(rows) * cell + hdr
        cx, cy, half = sequence_bounds(list(seqs.values()))

        s = [f'<svg viewBox="0 0 {total_w} {total_h}" width="100%">']
        for k, fi in enumerate(frames):
            s.append(f'<text x="{lab_w + k*cell + cell/2}" y="15" font-size="16" '
                     f'text-anchor="middle" fill="#3D4642" '
                     f'font-family="ui-sans-serif,sans-serif">t={fi}</text>')
        for r, (label, key, color, pc) in enumerate(rows):
            y = hdr + r * cell
            if r:
                s.append(f'<line x1="0" y1="{y}" x2="{total_w}" y2="{y}" '
                         f'stroke="#D7DED9" stroke-width="1.5"/>')
            s.append(f'<text x="{lab_w-12}" y="{y + cell/2 + 6}" font-size="19" '
                     f'text-anchor="end" fill="{color}" font-weight="800" '
                     f'font-family="ui-sans-serif,sans-serif">{label}</text>')
            ghost = seqs["source"] if key != "source" else None
            s.append(f'<g transform="translate({lab_w},{y})">')
            s.append(filmstrip(seqs[key], bones, frames, cx, cy, half, cell,
                               color, part_colors=pc, ghost=ghost))
            s.append("</g>")
        s.append("</svg>")
        blocks.append(
            f'<div class="qblock"><div class="qtitle">{name.title()}'
            f'<span>· source actor P{e["person_id"]}</span></div>'
            f'{"".join(s)}</div>')

    return "".join(blocks)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
# Figure heights are the layout's only real tuning knob: each board packs a
# different set of sections into a different column height, so the charts are
# sized per preset and check_layout.py is the arbiter of whether it fits.
N_FRAMES = {"140x100": 7, "180x90": 8, "a0": 4, "36x48": 4}

CHART_H = {
    "140x100": {"scatter": 476, "ablation": 432, "beta": 444, "quad": 418},
    "180x90":  {"scatter": 780, "ablation": 640, "beta": 560, "quad": 470},
    "a0":      {"scatter": 760, "ablation": 620, "beta": 560, "quad": 470},
    "36x48":   {"scatter": 740, "ablation": 600, "beta": 546, "quad": 470},
}


def sections(tsne_action: str, tsne_ident: str, n_frames: int = 4,
             wide_arch: bool = False,
             ch: dict[str, int] | None = None) -> dict[str, str]:
    ch = ch or CHART_H["a0"]
    take_bg = step_gradient(400, 300,
                            [(0.0, GREEN_DK), (0.6, GREEN), (1.0, GREEN_MD)],
                            skew=0.5, n=48)
    scatter = charts.privacy_utility_scatter(h=ch["scatter"])
    ablation = charts.ablation_bars(h=ch["ablation"])
    beta = charts.beta_tradeoff(h=ch["beta"])
    arch = charts.architecture_wide() if wide_arch else charts.architecture()
    quad = charts.quadruplet(h=ch["quad"])
    qual = qualitative_panel(n_frames)

    S: dict[str, str] = {}

    S["problem"] = """
      <section class="tint">
        <h2 class="i">The problem</h2>
        <p>3D skeletons look anonymous, but bone lengths, limb proportions and
           gait are a <b>biometric signature</b>: an off-the-shelf recogniser
           re-identifies the actor in raw NTU skeletons
           <span class="hl">75.4% of the time</span> across 40 identities,
           30&times; chance.</p>
        <p>Yet skeletons are exactly the modality we want to share for health
           monitoring, elder care and behaviour analysis. Blurring or adding noise
           destroys the action along with the identity.</p>
      </section>"""

    S["idea"] = """
      <section>
        <h2>The idea</h2>
        <p><b>Replace</b> identity rather than removing it. Encode the motion of
           one person and the body of another into two separate latent streams,
           then decode a sequence that performs the <b>source action</b> on the
           <b>target skeleton</b>.</p>
        <ul>
          <li class="a"><b>Action stream</b> (768-d) captures dynamics: position,
              velocity, acceleration &rarr; multi-scale temporal convolution &rarr; attention.</li>
          <li class="i"><b>Identity stream</b> (256-d) captures structure: mean
              static pose + bone lengths &rarr; spatial GCN. Deliberately
              <b>low-capacity</b>, an information bottleneck that starves it of
              motion detail.</li>
          <li><b>Asymmetric by design.</b> The architecture does most of the
              disentangling; the contrastive, adversarial, orthogonality and
              cross-correlation losses do the rest.</li>
        </ul>
      </section>"""

    S["signal"] = f"""
      <section class="tint">
        <h2>Training signal</h2>
        <p style="margin-bottom:2mm">Every training sample is a <b>cross-identity
           quadruplet</b>: two actors who each perform the same two actions, so the
           exact retargeting target already exists in the data and the reconstruction
           term needs no adversarial proxy.</p>
        <figure>{quad}</figure>
      </section>"""

    S["arch"] = f"""
      <section>
        <h2>Architecture</h2>
        <figure>{arch}</figure>
        <div class="archnote">
          <figcaption>The decoder attends to each stream separately, then blends
            them with a learned per-channel gate <b>&alpha;</b>. Generation is
            autoregressive under a causal mask.</figcaption>
          <figcaption><b>22.7M parameters</b>, three quarters of them in the
            decoder, against 4.9M for DMR and 1.0M for PMR.</figcaption>
          <figcaption>Retargeting runs once, offline, at collection time, so
            downstream consumers pay no inference cost. They do retrain on the
            shared skeletons.</figcaption>
        </div>
      </section>""" if wide_arch else f"""
      <section>
        <h2>Architecture</h2>
        <figure>{arch}</figure>
        <figcaption>The decoder attends to each stream separately, then blends them
          with a learned per-channel gate <b>&alpha;</b>, autoregressively under a
          causal mask.
          <br><br><b>22.7M parameters</b>, three quarters of them in the decoder,
          against 4.9M for DMR and 1.0M for PMR. Retargeting runs once, offline at
          collection time, so downstream consumers pay no inference cost. They do
          retrain on the shared skeletons.</figcaption>
      </section>"""

    S["stages"] = """
      <section>
        <h2 class="g">Three-stage training</h2>
        <p class="small" style="margin-bottom:3mm">Order is mandatory. Skipping a
           stage measurably degrades disentanglement.</p>
        <ol class="steps">
          <li><b>Encoder pre-training.</b> Both encoders learn with classification
              heads plus the four disentanglement losses. Decoder frozen.</li>
          <li><b>Decoder training.</b> Encoders frozen. Reconstruction plus
              physical plausibility (bone length, smoothness, velocity,
              end-effector, foot contact, joint limits). Teacher forcing
              1.0 &rarr; 0.5.</li>
          <li><b>End-to-end fine-tuning.</b> All components jointly, teacher
              forcing 0.5 &rarr; 0.3, with output-level adversarial and
              cooperative heads.</li>
        </ol>
      </section>"""

    S["qual"] = f"""
      <section>
        <h2>Qualitative results</h2>
        <div class="qual">{qual}</div>
        <figcaption>Real sequences from NTU RGB+D 60. <b>Grey ghost</b> = source
          motion; coloured overlay = each method's output. Ours preserves the action
          on a new body; <b>DMR</b> keeps more of the source identity, and <b>PMR</b>
          shifts toward the target but degrades the motion, at 19.9% re-trained
          AR.</figcaption>
      </section>"""

    S["tsne"] = f"""
      <section class="tint">
        <h2>Is it disentangled?</h2>
        <p style="margin-bottom:3.5mm">UMAP of the <b>action embedding</b>, taken
           from the action-recognition classifier's features.</p>
        <div class="tsne">
          <div>
            <img src="data:image/png;base64,{tsne_action}" alt="action embedding coloured by action class">
            <div class="cap" style="color:var(--action)">Five of the 49 classes</div>
            <div class="sub">Five shown for legibility; each lands in its own tight
                             cluster. The action code carries the semantics.</div>
          </div>
          <div>
            <img src="data:image/png;base64,{tsne_ident}" alt="action embedding coloured by identity">
            <div class="cap" style="color:var(--identity)">All 40 identities</div>
            <div class="sub">Colouring the same embedding by actor leaves no
                             recoverable structure.</div>
          </div>
        </div>
      </section>"""

    S["results"] = """
      <section>
        <h2 class="a">Main results</h2>
        <p class="small" style="margin-bottom:1mm">NTU RGB+D 60, cross-view, ours at
          <b>&beta; = 0.2</b>. <b>Pre-trained</b> (primary): raw-trained SGN applied
          directly to retargeted output, no retraining. <b>Re-trained AR</b>: a fresh
          SGN trained on retargeted data. Chance AR &asymp; 2%, RI &asymp; 2.5%.</p>
        <table class="res">
          <colgroup><col class="m"><col class="v3"><col class="v3"><col class="v3"></colgroup>
          <thead>
            <tr><th></th><th class="grp" colspan="2">Pre-trained SGN</th>
                <th class="grp">Re-trained</th></tr>
            <tr><th>Method</th><th>AR&nbsp;&uarr;</th><th>RI&nbsp;&darr;</th>
                <th>AR&nbsp;&uarr;</th></tr>
          </thead>
          <tbody>
            <tr class="ref"><td>Raw skeleton</td><td>89.1</td><td>75.4</td><td>89.1</td></tr>
            <tr><td>DMR</td><td>49.1</td><td>25.7</td><td>43.1</td></tr>
            <tr><td>PMR</td><td>35.7</td><td>7.8</td><td>19.9</td></tr>
            <tr class="ours"><td>DisentangledTMR</td><td>75.8</td><td>18.1</td>
                <td>87.1</td></tr>
          </tbody>
        </table>
        <figcaption>Ours is mean &plusmn; std over 5 runs (&plusmn;1.1 AR, &plusmn;1.8 RI).
          <b>More than double PMR's utility under the same protocol.</b> PMR reaches a
          lower RI (7.8%) only by degrading its output until the action is unreadable
          too, at 19.9% re-trained AR.</figcaption>
      </section>"""

    S["notes"] = """
      <section>
        <h2>Scope</h2>
        <p><b>Data.</b> NTU RGB+D 60 / 120 and ETRI-Activity3D, single-person
          sequences only (two-person NTU actions excluded).</p>
        <p><b>Privacy is measured</b> against trained recognisers, not a worst-case
          adversary with full knowledge of the retargeting model.</p>
      </section>"""

    S["scatter"] = f"""
      <section>
        <h2>Privacy–utility landscape</h2>
        <figure>{scatter}</figure>
        <figcaption>Down and to the right is better. DMR leaks identity through
          motion dynamics; PMR buys its low RI by over-anonymising until the action
          goes with it. Only DisentangledTMR reaches the low-RI regime with the
          action still readable.</figcaption>
      </section>"""

    S["ablation"] = f"""
      <section>
        <h2 class="a">Which stage does the work?</h2>
        <figure>{ablation}</figure>
        <figcaption>Every configuration that includes <b>stage 3</b> lands near 82%
          AR, so end-to-end fine-tuning does most of the work. <b>Stage 1 alone</b>
          reaches the strongest privacy of any variant, 15.3% RI, but at 57.3% AR the
          output has lost the structural coherence a classifier needs.
          <br><br>These runs omit the output-level supervision suite, so both columns
          sit away from the main table: compare rows against each other, not against
          it.</figcaption>
      </section>"""

    S["beta"] = f"""
      <section class="tint">
        <h2 class="g">One knob at test time</h2>
        <figure>{beta}</figure>
        <figcaption>&beta; moves the operating point <i>after</i> training. A
          <b>pre-trained</b> recogniser reads the action while enough source structure
          survives, then falls off a cliff between &beta; = 0.20 and 0.25,
          76.8 &rarr; 38.8%. A <b>re-trained</b> one holds about 85% almost the whole
          way, so the action itself is still there: the cliff is a compatibility
          limit, not a loss of information. Deployments pick &beta; from their own
          risk budget.</figcaption>
      </section>"""

    S["losses"] = """
      <section class="tint">
        <h2 class="g">The objective</h2>
        <p class="small" style="margin-bottom:2mm">Weights from
           <span class="mono">configs/main_config.yaml</span> (Optuna-tuned).</p>
        <table class="loss">
          <colgroup><col class="m"><col class="w"></colgroup>
          <tbody>
            <tr class="hdr"><td>Disentanglement</td><td></td></tr>
            <tr><td>Adversarial (GRL on identity)</td><td>1.53</td></tr>
            <tr><td>Re-identification head</td><td>1.00</td></tr>
            <tr><td>InfoNCE contrastive</td><td>0.76</td></tr>
            <tr><td>Feature orthogonality</td><td>0.62</td></tr>
            <tr><td>Mutual-information min.</td><td>0.58</td></tr>
            <tr><td>Action recognition head</td><td>0.54</td></tr>
            <tr class="hdr"><td>Reconstruction &amp; physics</td><td></td></tr>
            <tr><td>Bone-length consistency</td><td>6.19</td></tr>
            <tr><td>Joint position (MSE)</td><td>5.32</td></tr>
            <tr><td>End-effector</td><td>4.73</td></tr>
            <tr><td>Temporal smoothness</td><td>2.87</td></tr>
            <tr><td>Velocity distribution</td><td>2.26</td></tr>
            <tr><td>Joint limits</td><td>1.77</td></tr>
            <tr><td>Foot contact</td><td>0.54</td></tr>
          </tbody>
        </table>
        <figcaption>Physical-plausibility terms outweigh the raw position error.
          A skeleton that reconstructs well but breaks its own bone lengths is
          useless downstream.</figcaption>
      </section>"""

    S["take"] = f"""
      <section class="take">
{take_bg}
        <h2>Takeaways</h2>
        <ul>
          <li>Skeleton data is <b>not</b> anonymous; treat it as biometric.</li>
          <li><b>Replacing</b> identity beats suppressing it: a single encoder that
              over-anonymises loses the action too.</li>
          <li><b>Asymmetric architecture</b> does the disentangling; the losses
              only refine it.</li>
          <li>A single post-hoc <b>&beta;</b> exposes the whole privacy–utility curve
              from one trained model.</li>
        </ul>
      </section>"""

    S["limits"] = """
      <section style="padding:4.5mm 5.5mm">
        <p class="small"><b>Limitations.</b> Single-person sequences only
          (two-person NTU actions excluded); privacy is measured against trained
          recognisers, not a worst-case adversary with full knowledge of the
          retargeting model.</p>
      </section>"""

    return S


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def build(page: str = "140x100", print_ready: bool = False) -> str:
    PW, PH, NCOL, SCALE, SIDE, _label, LAND = PAGES[page]
    tsne_action = b64(ASSETS / "tsne_action_by_action_clf_umap.png")
    tsne_ident = b64(ASSETS / "tsne_action_by_identity_clf_umap.png")

    css = """
:root{
  /* UNC Charlotte primary */
  --ours:#005035;      /* Charlotte Green  PMS 7484 */
  --gold:#A49665;      /* Niner Gold       PMS 7503 */
  /* UNC Charlotte secondary */
  --jasper:#F1E6B2; --pine:#899064; --clay:#802F2D; --sky:#007377;
  --ink:#101820;       /* Ore Black        PMS Black 6C - default text colour */
  --muted:#3D4642;     /* dark neutral; used sparingly, never for body copy */
  --line:#D7DED9; --wash:#F4F8F6;
  /* data-series roles, all drawn from the brand palette */
  --action:#007377; --identity:#802F2D; --gate:#7D6F3C;
}
*{box-sizing:border-box;margin:0;padding:0}
@page{ size:__PW__mm __PH__mm; margin:0; }
html,body{ background:#fff; }
body{
  font-family:"Inter","Helvetica Neue",ui-sans-serif,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink); -webkit-print-color-adjust:exact; print-color-adjust:exact;
}
.poster{
  width:__PW__mm; height:__PH__mm; padding:15mm 15mm 11mm;
  display:flex; flex-direction:column; background:#fff; position:relative;
  overflow:hidden;
}

/* ---------- header ---------- */
/* The masthead stacks rather than splitting into two columns: the title gets
   the full board width, and the marks sit on one row underneath beside the
   authors. Side-by-side left the logo column with a third of its height empty,
   because the marks are wide and the title block is tall. */
header{
  background:#005035;
  color:#fff; border-radius:7mm; padding:7mm 9mm 7mm; position:relative;
  overflow:hidden; flex:0 0 auto; display:flex; flex-direction:column; gap:6mm;
  border-bottom:2.6mm solid var(--gold);
}
/* CSS gradients are rasterised by the print engine, which lands well under the
   printer's 100 DPI floor on a board this size. Drawn as SVG they stay vector. */
svg.bg{ position:absolute; inset:0; width:100%; height:100%; z-index:0; }
/* Flat fills only: any gradient in the masthead comes back from the print
   pipeline as a low-resolution bitmap, well under the printer's DPI floor. */
.hmain{ position:relative; z-index:1; min-width:0; }
.hbottom{
  position:relative; z-index:1; display:flex; align-items:center; gap:8mm;
  padding-top:1mm;
}
.hwho{ flex:1 1 auto; min-width:0; }
/* The panel hugs its marks: each slot is exactly as wide as its mark needs to
   be at the shared optical height, so the panel has no slack to distribute. */
.logopanel{
  flex:0 0 auto;
  background:#fff; border-radius:3.5mm; padding:3.5mm 4mm;
  display:flex; align-items:center; gap:0;
}
.logopanel .logoslot + .logoslot{ border-left:1.5px solid #dfe5e2; }
.logoslot{
  flex:0 0 auto; display:flex; align-items:center; justify-content:center;
  padding:0 3.5mm;
}
.logoslot svg, .logoslot img{
  display:block; width:100%; height:auto; object-fit:contain;
}
/* QR chip, only rendered when a code is present in poster/ */
.qr{
  flex:0 0 auto; background:#fff; border-radius:3.5mm; padding:3mm;
  display:flex; align-items:center;
}
.qr img, .qr svg{ display:block; width:__QRW__mm; height:__QRW__mm; }
/* One plain line above the authors. Pills read as decoration, not
   information: the board is at ECCV and is obviously a poster. */
.venue{
  font-size:15pt; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color:var(--gold); margin-bottom:2.5mm;
}
h1{
  font-size:61pt; line-height:1.04; font-weight:800; letter-spacing:-.022em;
  margin-bottom:4mm;
}
h1 .lead{ color:#8fd8bb; }
.authors{ font-size:20.5pt; font-weight:600; letter-spacing:.005em; }
.authors sup{ font-size:13.5pt; color:#8fd8bb; font-weight:700; }
.affil{ font-size:14.5pt; color:#a7d8c4; margin-top:2mm; }

/* ---------- headline strip ---------- */
.headline{
  display:grid; grid-template-columns:repeat(4,1fr); gap:4.5mm;
  margin:4.5mm 0 4mm; flex:0 0 auto;
}
.stat{
  border:2px solid var(--line); border-radius:5mm; padding:4mm 4.5mm 3.5mm;
  background:var(--wash); position:relative; overflow:hidden;
}
.stat.hero{ background:#e9f3ee; border-color:#a7d8c4; }
.stat .k{ font-size:36pt; font-weight:800; letter-spacing:-.02em; line-height:1; }
.stat .k small{ font-size:19pt; font-weight:700; }
.stat .l{ font-size:16.5pt; color:var(--ink); margin-top:2mm; line-height:1.36; }
.stat .t{ font-size:14pt; font-weight:800; letter-spacing:.09em; text-transform:uppercase;
          color:var(--muted); margin-bottom:2.5mm; }

/* ---------- columns ---------- */
.cols{ display:grid; grid-template-columns:repeat(__NCOL__,1fr); gap:6.5mm; flex:1 1 auto; min-height:0; }
/* One constant gap everywhere. Distributing the leftover height with
   space-between instead made every gap a different size, which is what read as
   machine-set: the fix is to size the figures so there is no leftover. */
.col{ display:flex; flex-direction:column; gap:6.5mm; justify-content:flex-start; }
/* A block two grid columns wide: full-width head, then two sub-columns. */
.span2{ grid-column:span 2; display:flex; flex-direction:column; gap:6.5mm; min-width:0; }
.subcols{ display:grid; grid-template-columns:1fr 1fr; gap:6.5mm; flex:1 1 auto; min-height:0; }
.archnote{ display:grid; grid-template-columns:repeat(3,1fr); gap:7mm; margin-top:3.5mm; }
.archnote figcaption{ margin-top:0; }
section{
  border:2px solid var(--line); border-radius:5mm; padding:7mm 7mm 7.5mm;
  background:#fff; break-inside:avoid;
}
section.tint{ background:var(--wash); }
h2{
  font-size:27pt; font-weight:800; letter-spacing:-.008em; margin-bottom:3.5mm;
  display:flex; align-items:center; gap:3mm; line-height:1.1;
}
h2::before{
  content:""; width:3.6mm; height:9.5mm; border-radius:2px; background:var(--ours);
  flex:0 0 auto;
}
h2.a::before{ background:var(--action); }
h2.i::before{ background:var(--identity); }
h2.g::before{ background:var(--gate); }
p{ font-size:20pt; line-height:1.5; color:var(--ink); }
p+p{ margin-top:3mm; }
.small{ font-size:17.5pt; color:var(--ink); line-height:1.45; }
b,strong{ font-weight:700; }
.hl{ background:var(--jasper); padding:0 1mm; border-radius:2px; font-weight:600; }

ul{ list-style:none; }
li{
  font-size:20.5pt; line-height:1.48; padding-left:7.5mm; position:relative;
  margin-bottom:3.2mm; color:var(--ink);
}
li::before{
  content:""; position:absolute; left:0; top:3mm; width:3.4mm; height:3.4mm;
  border-radius:50%; background:var(--ours);
}
li.a::before{ background:var(--action); }
li.i::before{ background:var(--identity); }

figure{ margin-top:3mm; }
.archimg{ width:100%; height:auto; display:block; }
figcaption{ font-size:17pt; color:var(--ink); margin-top:3mm; line-height:1.42; }
svg{ display:block; }

/* ---------- stages ---------- */
ol.steps{ list-style:none; counter-reset:st; }
ol.steps li{
  counter-increment:st; font-size:19pt; line-height:1.45; color:var(--ink);
  padding-left:11mm; position:relative; margin-bottom:3.5mm;
}
ol.steps li::before{
  content:counter(st); position:absolute; left:0; top:0.6mm;
  width:7.5mm; height:7.5mm; border-radius:50%; background:var(--gate);
  color:#fff; font-size:15pt; font-weight:800; display:flex;
  align-items:center; justify-content:center;
}
.stages{ display:flex; flex-direction:column; gap:3mm; }
.stage{
  display:grid; grid-template-columns:11mm 1fr; gap:3.5mm; align-items:start;
  border-left:3px solid var(--line); padding:2.5mm 0 2.5mm 4mm;
}
.stage .n{
  width:11mm; height:11mm; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:17pt; font-weight:800; color:#fff;
}
.stage h3{ font-size:20pt; font-weight:800; margin-bottom:1mm; }
.stage p{ font-size:18.5pt; line-height:1.42; color:var(--ink); }

/* ---------- table ---------- */
table{
  width:100%; border-collapse:collapse; font-size:19.5pt; margin-top:2mm;
  table-layout:fixed;
}
th{
  text-align:right; font-size:15pt; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink); padding:0 0 2mm; font-weight:800;
}
thead tr:last-child th{ border-bottom:2px solid var(--ink); }
th.grp{
  text-align:center; border-bottom:1.5px solid var(--line); padding-bottom:1mm;
  color:var(--ink); font-size:15pt;
}
th:first-child{ text-align:left; }
td{
  padding:2.5mm 0; border-bottom:1px solid var(--line); text-align:right;
  font-variant-numeric:tabular-nums;
}
td:first-child{ text-align:left; font-weight:600; padding-right:2mm; letter-spacing:-.01em; }
table.res col.m{ width:37%; } table.res col.v{ width:15.75%; }
table.res col.v3{ width:21%; }
table.loss{ font-size:16pt; margin-top:1mm; }
table.loss col.m{ width:76%; } table.loss col.w{ width:24%; }
table.loss td{ padding:1.3mm 0; }
table.loss tr.hdr td{
  font-size:13.5pt; letter-spacing:.08em; text-transform:uppercase; font-weight:800;
  color:var(--muted); border-bottom:1.5px solid var(--ink); padding-top:3mm;
}
table.loss tr:first-child td{ padding-top:0; }
.mono{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:14.5pt; }
tr.ours td{ background:#e3efe9; font-weight:800; color:var(--ours); }
tr.ours td:first-child{ border-radius:3px 0 0 3px; padding-left:2mm; }
tr.ours td:last-child{ border-radius:0 3px 3px 0; padding-right:2mm; }
tr.ref td{ color:var(--muted); }

/* ---------- qualitative ---------- */
.qual{ display:flex; flex-direction:column; gap:11mm; }
.qblock{ border:1.5px solid var(--line); border-radius:3mm; padding:4mm 3mm 2mm; background:#fff; }
.qtitle{ font-size:19pt; font-weight:800; margin-bottom:1mm; }
.qtitle span{ font-weight:500; color:var(--ink); font-size:17pt; margin-left:2mm; }

/* ---------- tsne ---------- */
.tsne{ display:grid; grid-template-columns:1fr 1fr; gap:4mm; }
.tsne img{ width:100%; display:block; border:1.5px solid var(--line); border-radius:3mm; }
.tsne .cap{ font-size:18pt; font-weight:800; margin-top:2.5mm; line-height:1.3; }
.tsne .sub{ font-size:16.5pt; color:var(--ink); line-height:1.38; }

/* ---------- takeaways ---------- */
.take{
  background:#005035; color:#fff; position:relative; overflow:hidden;
  border:none; border-radius:5mm; border-bottom:2.2mm solid var(--gold);
}
.take h2, .take ul, .take p{ position:relative; z-index:1; }
.take h2{ color:#fff; }
.take h2::before{ background:var(--gold); }
.take li{ color:#d5efe4; font-size:19.5pt; }
.take li::before{ background:var(--gold); }
.take .small{ color:#a7d8c4; }

footer{
  margin-top:3mm; display:flex; justify-content:space-between; align-items:center;
  font-size:16pt; color:var(--ink); flex:0 0 auto;
  border-top:1.5mm solid var(--gold); padding-top:3mm;
}
.code{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:15pt; }

/* ---------- screen preview only ---------- */
@media screen{
  html{ background:#3f4247; }
  body{ display:block; }
  #wrap{ display:flex; justify-content:center; padding:24px 0; }
  /* .poster is a flex item here, so without flex:0 0 auto it SHRINKS to the
     viewport width while keeping its full height — a 1800x900 board then
     previews as a tall narrow strip. The JS transform does the fitting. */
  .poster{ flex:0 0 auto; box-shadow:0 12px 60px rgba(0,0,0,.55);
           transform-origin:top center; }
}
@media print{
  /* The screen-preview scale is applied as an INLINE style by JS, so it needs
     !important here or the whole poster prints shrunk inside the sheet. */
  #wrap{ padding:0 !important; height:auto !important; display:block !important; }
  .poster{ transform:none !important; box-shadow:none !important; }
}
"""
    css = scale_css(css, SCALE)

    # The UMAP panels are the only raster figures in the body. Cap their printed
    # width at the size that still clears the printer's 100 DPI floor; appended
    # after scale_css so the cap is an absolute physical size, not a scaled one.
    umap_px = png_width(ASSETS / "tsne_action_by_action_clf_umap.png")
    umap_mm = umap_px / MIN_IMAGE_DPI * 25.4
    css += (f"\n.tsne{{ grid-template-columns:repeat(2,{umap_mm:.1f}mm);"
            f" justify-content:space-between; }}\n")

    # Print-ready sheet: trim size plus bleed, plus a margin outside the bleed
    # for crop marks. Marks stop BLEED_MM short of the trim line so they never
    # print inside the finished poster.
    marks_html = ""
    if print_ready:
        m = BLEED_MM + MARK_MM                     # media margin around the trim
        css += f"""
@page{{ size:{PW + 2*m:g}mm {PH + 2*m:g}mm; margin:0; }}
/* Chrome rounds the sheet to whole device pixels, so an exactly page-sized
   element can spill a hundredth of a millimetre onto a second page. */
html,body{{ width:{PW + 2*m:g}mm; height:{PH + 2*m:g}mm; overflow:hidden; }}
#wrap{{ padding:0 !important; display:block !important; position:relative;
       width:{PW + 2*m:g}mm !important; height:{PH + 2*m:g}mm !important;
       background:#fff; }}
/* Absolute, not margin: a top margin on .poster would collapse out of #wrap and
   drag the whole sheet — crop marks included — down the page. */
.poster{{ position:absolute; left:{m:g}mm; top:{m:g}mm; margin:0 !important;
         transform:none !important; box-shadow:none !important; }}
.cropmark{{ position:absolute; background:#000; }}
.cropmark.h{{ width:{MARK_MM:g}mm; height:{MARK_W_MM:g}mm; }}
.cropmark.v{{ width:{MARK_W_MM:g}mm; height:{MARK_MM:g}mm; }}
"""
        # Eight marks: two per corner, offset to sit just outside the bleed.
        edges = []
        for ys, yv in (("top", m), ("bottom", m)):
            for xs, xv in (("left", m), ("right", m)):
                # horizontal mark: runs from the sheet edge in to the bleed edge
                edges.append(f'<div class="cropmark h" style="{xs}:0;'
                             f'{ys}:{yv - MARK_W_MM / 2:g}mm"></div>')
                # vertical mark
                edges.append(f'<div class="cropmark v" style="{ys}:0;'
                             f'{xs}:{xv - MARK_W_MM / 2:g}mm"></div>')
        marks_html = "".join(edges)

    if LAND:
        # The landscape boards are short. A tall portrait-style header would eat a
        # third of it, so the masthead is compressed and the logos go in one row.
        # Appended after scale_css because the landscape scale is 1.0 anyway.
        css += """
.poster.land header{ padding:7mm 9mm 6.5mm; gap:5mm; }
.poster.land .venue{ font-size:18pt; margin-bottom:3mm; }
/* the title has the whole board width here, so it sets on two lines with room
   to spare; sized to fill them rather than to fit a column */
.poster.land h1{ font-size:82pt; margin-bottom:0; }
.poster.land .hbottom{ gap:9mm; padding-top:5mm; }
.poster.land .authors{ font-size:24pt; }
.poster.land .authors sup{ font-size:15.5pt; }
.poster.land .affil{ font-size:17pt; margin-top:2mm; }
.poster.land .logopanel{ padding:3.5mm 4mm; }
.poster.land .headline{ margin:4.5mm 0 4.5mm; }
.poster.land .stat{ padding:4mm 4.5mm 3.5mm; }
.poster.land .stat .k{ font-size:36pt; }
.poster.land .stat .k small{ font-size:19pt; }
.poster.land .stat .l{ font-size:17.5pt; margin-top:2mm; }
.poster.land .stat .t{ font-size:14pt; margin-bottom:2mm; }
.poster.land footer{ font-size:17pt; margin-top:2.5mm; padding-top:2.5mm; }
"""

    # A wider filmstrip is shorter per unit width, which is what the short
    # landscape columns need; portrait has the height to spare.
    wide_arch = any(isinstance(p, SPAN2) for p in COLUMN_PLANS[NCOL])
    S = sections(tsne_action, tsne_ident, N_FRAMES[page],
                 wide_arch=wide_arch, ch=CHART_H[page])

    header_bg = step_gradient(1000, 200,
                              [(0.0, GREEN_DK), (0.45, GREEN),
                               (0.78, GREEN_MD), (1.0, GREEN_LT)],
                              skew=0.42, n=72)

    uncc, usu, eccv = (logos.logo("unc-charlotte"), logos.logo("utah-state"),
                       logos.logo("eccv"))
    # One white panel rather than three chips: the three marks have very
    # different aspect ratios (1.9 / 5.8 / 2.0), so separate cards read as
    # mismatched sizes. Equal-height slots with rules between them fix that.
    # Slot width tracks each mark's aspect ratio, so every mark ends up the
    # same height. Equal-width slots would render the tall UNCC lockup small
    # and the very wide USU lockup large.
    ar = {k: logos.aspect(k) for k in ("unc-charlotte", "utah-state", "eccv")}
    # Equal height is not equal optical weight: the UNCC lockup stacks its
    # wordmark under the crown, so it needs extra height to read at the same
    # size as the single-line USU lockup. Slot width tracks aspect x optical.
    optical = {"unc-charlotte": 1.32, "utah-state": 0.78, "eccv": 1.12}

    # Every mark is set to MARK_H tall (times its optical factor), so its slot
    # is that height times its own aspect ratio. Sizing the slots instead of
    # letting them share out a fixed panel width is what keeps the marks from
    # floating in white space when the panel is wider than they need.
    mark_h = 25.0 * SCALE
    pad = 3.5 * SCALE

    def slot(stem, art):
        w = mark_h * optical[stem] * ar[stem] + 2 * pad
        return f'<div class="logoslot" style="width:{w:.2f}mm">{art}</div>'

    logo_block = ('<div class="logopanel">'
                  + slot("unc-charlotte", uncc) + slot("utah-state", usu)
                  + slot("eccv", eccv) + '</div>')
    qr_block = qr_chip()
    def column(plan) -> str:
        if isinstance(plan, SPAN2):
            return ('<div class="span2">'
                    + "".join(S[k] for k in plan.head)
                    + '<div class="subcols">'
                    + f'<div class="col">{"".join(S[k] for k in plan.left)}</div>'
                    + f'<div class="col">{"".join(S[k] for k in plan.right)}</div>'
                    + "</div></div>")
        return f'<div class="col">{"".join(S[k] for k in plan)}</div>'

    cols = "".join(column(plan) for plan in COLUMN_PLANS[NCOL])

    land_cls = "land" if LAND else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DisentangledTMR · ECCV 2026 Poster</title>
<style>{css}</style>
</head>
<body>
<div id="wrap">{marks_html}
<div class="poster {land_cls}">

  <header>
{header_bg}
    <div class="hmain">
      <h1><span class="lead">DisentangledTMR:</span> Privacy-Preserving Skeleton
          Motion Retargeting via Factorized Transformers</h1>
    </div>
    <div class="hbottom">
      <div class="hwho">
        <div class="venue">ECCV 2026 &nbsp;·&nbsp; Malmö, Sweden
             &nbsp;·&nbsp; 8–12 September 2026</div>
        <div class="authors">
          Thomas Carr<sup>1,2</sup> &nbsp; Depeng Xu<sup>1</sup> &nbsp;
          Shuhan Yuan<sup>3</sup> &nbsp; Aidong Lu<sup>1</sup>
        </div>
        <div class="affil">
          <sup>1</sup>University of North Carolina at Charlotte &nbsp;·&nbsp;
          <sup>2</sup>Incerta Intelligence &nbsp;·&nbsp;
          <sup>3</sup>Utah State University
        </div>
      </div>
      {qr_block}
      {logo_block}
    </div>
  </header>

  <div class="headline">
    <div class="stat hero">
      <div class="t">Privacy gain</div>
      <div class="k" style="color:var(--identity)">75.4 → 18.1<small>%</small></div>
      <div class="l">Re-identification falls from <b>30× chance</b> to
                     <b>7.2×</b>, against a 2.5% floor</div>
    </div>
    <div class="stat">
      <div class="t">Utility kept</div>
      <div class="k" style="color:var(--action)">75.8<small>%</small></div>
      <div class="l">Action recognition with <b>no retraining</b>, and 87.1% once
                     downstream models are retrained</div>
    </div>
    <div class="stat">
      <div class="t">vs. PMR</div>
      <div class="k" style="color:var(--ours)">2.1<small>× utility</small></div>
      <div class="l">75.8% against PMR's 35.7% under the same protocol, at a
                     comparable privacy level</div>
    </div>
    <div class="stat">
      <div class="t">Tunable</div>
      <div class="k" style="color:var(--gate)">β<small> = 0.2</small></div>
      <div class="l">One knob sets the operating point after training:
                     <b>76.8% AR at 17.3% RI</b></div>
    </div>
  </div>

  <div class="cols">{cols}</div>

  <footer>
    <div>Carr, Xu, Yuan &amp; Lu &nbsp;·&nbsp; UNC Charlotte &amp; Utah State
         University &nbsp;·&nbsp; Incerta Intelligence</div>
    <div>Code &amp; interactive demo &nbsp;<span class="code">{PROJECT_URL}</span></div>
  </footer>

</div>
</div>

<script>
// Screen-only: scale the sheet down to fit the window. Print is untouched.
(function () {{
  var POSTER_W = __PW__ / 25.4 * 96, POSTER_H = __PH__ / 25.4 * 96;
  var el = document.querySelector('.poster'), wrap = document.getElementById('wrap');
  function fit() {{
    if (!window.matchMedia('screen').matches) return;
    var s = Math.min(1, (window.innerWidth - 48) / POSTER_W,
                        (window.innerHeight - 48) / POSTER_H);
    el.style.transform = 'scale(' + s + ')';
    wrap.style.height = (POSTER_H * s + 48) + 'px';
  }}
  fit();
  window.addEventListener('resize', fit);
  // Print must always be 1:1 — drop the preview transform around the print job.
  function unscale() {{ el.style.transform = 'none'; wrap.style.height = 'auto'; }}
  window.addEventListener('beforeprint', unscale);
  window.addEventListener('afterprint', fit);
  if (window.matchMedia) {{
    var mq = window.matchMedia('print');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change')
                         : mq.addListener.bind(mq))(function (e) {{
      e.matches ? unscale() : fit();
    }});
  }}
}})();
</script>
</body>
</html>
"""
    return (html.replace("__PW__", f"{PW:g}")
                .replace("__PH__", f"{PH:g}")
                .replace("__NCOL__", str(NCOL))
                .replace("__QRW__", f"{36 * SCALE:g}"))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    print_ready = "--print" in args
    which = [a for a in args if not a.startswith("-")] or DEFAULT_PAGES
    real = logos.using_real_art()
    print(f"logos: {'real art for ' + ', '.join(real) if real else 'generated wordmarks'}")
    for page in which:
        name = OUT_NAME[page]
        if print_ready:
            name = name.replace(".html", "_print.html")
        out = HERE / name
        out.write_text(build(page, print_ready=print_ready), encoding="utf-8")
        extra = f"  (+{BLEED_MM:g}mm bleed, crop marks)" if print_ready else ""
        print(f"wrote {out.name}  ({out.stat().st_size/1024:.0f} KB)  "
              f"{PAGES[page][5]}{extra}")
