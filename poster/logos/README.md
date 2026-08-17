# Logo drop-in

`build_poster.py` picks up real artwork automatically. Save files here using these
exact stems (any of `.svg`, `.png`, `.jpg`, `.webp` — SVG preferred for print):

| file                  | what to download                                   |
|-----------------------|----------------------------------------------------|
| `unc-charlotte.svg`   | UNC Charlotte primary or unit logo, Charlotte Green |
| `utah-state.svg`      | Utah State University wordmark                      |
| `eccv.svg`            | ECCV 2026 conference mark                           |

Notes:
- Logos sit on **white cards** on the green header, so use the standard
  (dark/full-colour) version, not the white knockout.
- Horizontal lockups fit best; the card scales the art to full card width.
- If a file is missing, a typographic wordmark in the correct brand colours is
  generated instead, so the poster always builds.

Rebuild after adding files:

    python3 build_poster.py
