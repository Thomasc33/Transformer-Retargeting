#!/usr/bin/env python3
"""Preflight a poster PDF against the ECCV 2026 printer's file requirements.

    python3 preflight.py 15204_Carr_1400x1000mm.pdf

Checks, in the printer's own order: one page at 1:1 scale, DeviceCMYK only,
every raster image >= 100 DPI at final size, all fonts embedded, 5-10 mm bleed
with crop marks, and a filename of {PAPER_ID}_Lastname_{W}x{H}mm.pdf.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF

MM = 72.0 / 25.4
MIN_DPI = 100.0


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {label:16s} {detail}")
    return ok


def main() -> int:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "15204_Carr_1400x1000mm.pdf")
    doc = fitz.open(pdf)
    page = doc[0]
    ok = True

    ok &= check("pages", doc.page_count == 1, f"{doc.page_count} page(s)")

    m = re.match(r"(\d+)_([A-Za-z-]+)_(\d+)x(\d+)mm\.pdf$", pdf.name)
    ok &= check("filename", bool(m), pdf.name +
                ("" if m else "  (want {ID}_Lastname_{W}x{H}mm.pdf)"))

    trim, bleed, media = page.trimbox, page.bleedbox, page.mediabox
    tw, th = trim.width / MM, trim.height / MM
    want = (float(m.group(3)), float(m.group(4))) if m else (tw, th)
    ok &= check("trim size", abs(tw - want[0]) < 0.5 and abs(th - want[1]) < 0.5,
                f"{tw:.1f} x {th:.1f} mm at 1:1 "
                f"(media {media.width/MM:.1f} x {media.height/MM:.1f} mm)")

    bl = min(trim.x0 - bleed.x0, trim.y0 - bleed.y0,
             bleed.x1 - trim.x1, bleed.y1 - trim.y1) / MM
    ok &= check("bleed", 5.0 - 0.01 <= bl <= 10.0 + 0.01, f"{bl:.1f} mm on all four sides")

    # Crop marks: hairline filled rectangles wholly outside the bleed box.
    marks = [d for d in page.get_drawings()
             if min(d["rect"].width, d["rect"].height) <= 1.0 * MM
             and max(d["rect"].width, d["rect"].height) <= 15.0 * MM
             and (d["rect"].x1 <= bleed.x0 + 0.5 or d["rect"].x0 >= bleed.x1 - 0.5 or
                  d["rect"].y1 <= bleed.y0 + 0.5 or d["rect"].y0 >= bleed.y1 - 0.5)]
    ok &= check("crop marks", len(marks) >= 8,
                f"{len(marks)} hairline marks outside the bleed (want 8)")

    spaces = set()
    for xref in range(1, doc.xref_length()):
        try:
            t = doc.xref_get_key(xref, "ColorSpace")
        except Exception:
            continue
        if t and t[0] != "null":
            spaces.add(str(t[1]))
    rgb = {s for s in spaces if "RGB" in s}
    ok &= check("colour", not rgb, "DeviceCMYK only" if not rgb
                else f"RGB objects present: {sorted(rgb)}")

    # pdffonts is the authority here: PyMuPDF reports no file extension for
    # Type 3 fonts, which carry their glyphs inline and are always embedded.
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True)
    rows = [r for r in out.stdout.splitlines()[2:] if r.strip()]
    unembedded = [r.split()[0] for r in rows if " no " in f" {r[41:47].strip()} "]
    ok &= check("fonts", not unembedded,
                f"{len(rows)} font(s), all embedded" if not unembedded
                else f"not embedded: {sorted(set(unembedded))}")

    worst = None
    for img in page.get_images(full=True):
        xref = img[0]
        info = doc.extract_image(xref)
        px_w, px_h = info["width"], info["height"]
        for rect in page.get_image_rects(xref):
            dpi = min(px_w / (rect.width / 72), px_h / (rect.height / 72))
            worst = dpi if worst is None else min(worst, dpi)
    ok &= check("image DPI", worst is None or worst >= MIN_DPI,
                "no raster images" if worst is None
                else f"lowest {worst:.0f} DPI at final size (need {MIN_DPI:.0f})")

    print(f"\n{'READY TO SUBMIT' if ok else 'NOT READY'}  "
          f"{pdf.name}  ({pdf.stat().st_size/1024/1024:.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
