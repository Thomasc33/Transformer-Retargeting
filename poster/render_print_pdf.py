#!/usr/bin/env python3
"""Render the print-ready poster PDF the ECCV 2026 printer asks for.

    python3 render_print_pdf.py                 # 140x100 board, submission name
    python3 render_print_pdf.py --page 180x90

Pipeline:
  1. build the poster HTML with bleed and crop marks (build_poster --print),
  2. print it 1:1 in headless Chrome (fonts get embedded as subsets),
  3. convert to CMYK against the FOGRA39L profile with Ghostscript,
  4. stamp TrimBox / BleedBox so the printer's RIP knows where the trim is.

The printer's requirements this covers: 1:1 scale, CMYK (Fogra 39), embedded
fonts, >=100 DPI images, 5-10 mm bleed, crop marks, and the
{PAPER_ID}_Lastname_{W}x{H}mm.pdf filename.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

import build_poster as bp

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# FOGRA39L (ISO Coated v2), the characterisation the printer asks for. Kept in
# the repo so a build does not depend on a TeX installation.
ICC = HERE / "FOGRA39L_coated.icc"

PAPER_ID = "15204"
LAST_NAME = "Carr"

MM = 72.0 / 25.4          # mm -> PDF points


def chrome_pdf(html: Path, pdf: Path) -> None:
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=30000", f"--print-to-pdf={pdf}",
         f"file://{html.resolve()}"],
        check=True, capture_output=True)


def icc_dir() -> Path:
    """A profile directory holding Ghostscript's defaults plus FOGRA39L.

    Ghostscript resolves -sOutputICCProfile by name inside -sICCProfilesDir and
    loads its own defaults from the same place, so pointing it straight at the
    FOGRA profile fails ("Unable to open the initial device"). Clone its profile
    directory into a temp dir and drop ours in beside them.
    """
    if not ICC.exists():
        raise SystemExit(f"missing ICC profile: {ICC}")
    gs = Path(shutil.which("gs") or "").resolve()
    src = next((p.parent for p in gs.parents[1].rglob("iccprofiles/default_cmyk.icc")),
               None)
    if src is None:
        raise SystemExit("could not find Ghostscript's iccprofiles directory")
    d = Path(tempfile.mkdtemp(prefix="poster-icc-"))
    for f in src.glob("*.icc"):
        shutil.copy2(f, d)
    shutil.copy2(ICC, d)
    return d


def to_cmyk(src: Path, dst: Path) -> None:
    """Convert to DeviceCMYK against FOGRA39L, keeping everything vector."""
    profiles = icc_dir()
    subprocess.run(
        ["gs", "-dBATCH", "-dNOPAUSE", "-dSAFER", "-dQUIET",
         "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress",
         "-dCompatibilityLevel=1.6",
         "-dProcessColorModel=/DeviceCMYK",
         "-sColorConversionStrategy=CMYK",
         f"-sICCProfilesDir={profiles}/", f"-sOutputICCProfile={ICC.name}",
         "-dRenderIntent=1",
         # keep every mark vector and every image at full resolution
         "-dAutoFilterColorImages=false", "-dColorImageFilter=/FlateEncode",
         "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
         "-dDownsampleMonoImages=false",
         "-dEmbedAllFonts=true", "-dSubsetFonts=true",
         f"-sOutputFile={dst}", str(src)],
        check=True, capture_output=True)
    shutil.rmtree(profiles, ignore_errors=True)


def stamp_boxes(pdf: Path, trim_w_mm: float, trim_h_mm: float,
                bleed_mm: float) -> tuple[float, float]:
    """Set TrimBox and BleedBox around the centred artwork.

    Chrome rounds the sheet to whole device pixels, so the MediaBox can be a
    fraction of a millimetre over. The trim is therefore measured out from the
    centre at the exact finished size rather than inset from the media edge.
    """
    doc = fitz.open(pdf)
    page = doc[0]
    media = page.mediabox
    cx, cy = (media.x0 + media.x1) / 2, (media.y0 + media.y1) / 2
    hw, hh = trim_w_mm * MM / 2, trim_h_mm * MM / 2
    trim = fitz.Rect(cx - hw, cy - hh, cx + hw, cy + hh)
    bleed = fitz.Rect(trim.x0 - bleed_mm * MM, trim.y0 - bleed_mm * MM,
                      trim.x1 + bleed_mm * MM, trim.y1 + bleed_mm * MM)
    page.set_cropbox(media)
    page.set_trimbox(trim)
    page.set_bleedbox(bleed)
    doc.saveIncr()
    doc.close()
    return trim.width / MM, trim.height / MM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="140x100", choices=list(bp.PAGES))
    ap.add_argument("--keep-rgb", action="store_true",
                    help="skip the CMYK conversion (proofing only)")
    a = ap.parse_args()

    pw, ph = bp.PAGES[a.page][0], bp.PAGES[a.page][1]

    html = HERE / bp.OUT_NAME[a.page].replace(".html", "_print.html")
    html.write_text(bp.build(a.page, print_ready=True), encoding="utf-8")
    print(f"html   {html.name}  ({html.stat().st_size/1024:.0f} KB)")

    rgb = HERE / f"_rgb_{a.page}.pdf"
    chrome_pdf(html, rgb)
    print(f"pdf    rendered 1:1 in Chrome  ({rgb.stat().st_size/1024:.0f} KB)")

    out = HERE / f"{PAPER_ID}_{LAST_NAME}_{pw:.0f}x{ph:.0f}mm.pdf"
    if a.keep_rgb:
        rgb.replace(out)
    else:
        to_cmyk(rgb, out)
        rgb.unlink()
        print(f"cmyk   converted against {ICC.name}")

    tw, th = stamp_boxes(out, pw, ph, bp.BLEED_MM)
    print(f"boxes  TrimBox {tw:.1f} x {th:.1f} mm, "
          f"BleedBox +{bp.BLEED_MM:g}mm")
    print(f"wrote  {out.name}  ({out.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
