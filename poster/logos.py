"""Logo lockups for the poster header.

Official institutional artwork is not redistributable, so this module draws
typographic wordmarks in each institution's brand colours by default. Drop the
real files into poster/logos/ and they are picked up automatically:

    poster/logos/unc-charlotte.(svg|png)
    poster/logos/utah-state.(svg|png)
    poster/logos/eccv.(svg|png)

Anything found there wins over the generated wordmark.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"

# UNC Charlotte brand palette
NINER_GREEN = "#005035"
UNCC_GOLD = "#A49665"
# Utah State brand palette
AGGIE_BLUE = "#00263A"

_FONT = "Inter,'Helvetica Neue',ui-sans-serif,sans-serif"


def _find(stem: str) -> Path | None:
    for ext in (".svg", ".png", ".jpg", ".jpeg", ".webp"):
        p = LOGO_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _embed(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        svg = path.read_text(encoding="utf-8")
        # An inline <svg> cannot carry the XML prolog or a DOCTYPE, and fixed
        # width/height attributes would defeat the card's responsive sizing.
        svg = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.S)
        svg = re.sub(r"<!DOCTYPE.*?>", "", svg, flags=re.S)
        svg = re.sub(r'(<svg\b[^>]*?)\s+width="[^"]*"', r"\1", svg, count=1)
        svg = re.sub(r'(<svg\b[^>]*?)\s+height="[^"]*"', r"\1", svg, count=1)
        return svg.strip()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{data}" alt="">'


def _unc_charlotte() -> str:
    """Typographic lockup: green rule + stacked wordmark."""
    return f'''<svg viewBox="0 0 420 108" role="img" aria-label="UNC Charlotte">
  <rect x="0" y="0" width="9" height="108" fill="{NINER_GREEN}"/>
  <rect x="0" y="0" width="9" height="34" fill="{UNCC_GOLD}"/>
  <text x="26" y="40" font-family="{_FONT}" font-size="33" font-weight="800"
        letter-spacing="0.02em" fill="{NINER_GREEN}">UNC CHARLOTTE</text>
  <line x1="26" y1="54" x2="404" y2="54" stroke="{UNCC_GOLD}" stroke-width="2.5"/>
  <text x="26" y="76" font-family="{_FONT}" font-size="16.5" font-weight="600"
        letter-spacing="0.05em" fill="#3f4a45">COLLEGE OF COMPUTING AND</text>
  <text x="26" y="97" font-family="{_FONT}" font-size="16.5" font-weight="600"
        letter-spacing="0.05em" fill="#3f4a45">INFORMATICS</text>
</svg>'''


def _utah_state() -> str:
    return f'''<svg viewBox="0 0 420 78" role="img" aria-label="Utah State University">
  <rect x="0" y="0" width="9" height="78" fill="{AGGIE_BLUE}"/>
  <text x="26" y="34" font-family="{_FONT}" font-size="29" font-weight="800"
        letter-spacing="0.02em" fill="{AGGIE_BLUE}">UTAH STATE</text>
  <line x1="26" y1="46" x2="330" y2="46" stroke="#9aa7b0" stroke-width="2"/>
  <text x="26" y="67" font-family="{_FONT}" font-size="16.5" font-weight="600"
        letter-spacing="0.14em" fill="#4a5761">UNIVERSITY</text>
</svg>'''


def _eccv() -> str:
    return f'''<svg viewBox="0 0 300 92" role="img" aria-label="ECCV 2026">
  <text x="4" y="46" font-family="{_FONT}" font-size="42" font-weight="800"
        letter-spacing="-0.01em" fill="#111827">ECCV</text>
  <text x="152" y="46" font-family="{_FONT}" font-size="42" font-weight="300"
        letter-spacing="-0.01em" fill="#6b7280">2026</text>
  <line x1="4" y1="58" x2="286" y2="58" stroke="#d1d5db" stroke-width="2"/>
  <text x="4" y="80" font-family="{_FONT}" font-size="17" font-weight="600"
        letter-spacing="0.07em" fill="#6b7280">MALMÖ · SWEDEN</text>
</svg>'''


_GENERATED = {
    "unc-charlotte": _unc_charlotte,
    "utah-state": _utah_state,
    "eccv": _eccv,
}


def logo(stem: str) -> str:
    """Real artwork from poster/logos/ if present, else the generated wordmark."""
    found = _find(stem)
    return _embed(found) if found else _GENERATED[stem]()


# Nominal aspect ratios for the generated wordmarks, used when no real art
# is present. Real files are measured instead.
_FALLBACK_ASPECT = {"unc-charlotte": 420 / 108, "utah-state": 420 / 78,
                    "eccv": 300 / 92}


def aspect(stem: str) -> float:
    """Width / height of the mark, so slots can be sized to render all marks
    at the same optical height instead of the same width."""
    found = _find(stem)
    if found is None:
        return _FALLBACK_ASPECT[stem]
    if found.suffix.lower() == ".svg":
        m = re.search(r'viewBox="\s*[\d.\-]+[ ,]+[\d.\-]+[ ,]+([\d.]+)[ ,]+([\d.]+)',
                      found.read_text(encoding="utf-8"))
        return float(m.group(1)) / float(m.group(2)) if m else 2.0
    from PIL import Image
    with Image.open(found) as im:
        return im.width / im.height


def using_real_art() -> list[str]:
    return [s for s in _GENERATED if _find(s) is not None]
