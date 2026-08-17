#!/usr/bin/env python3
"""Verify every poster column fits its page. Run after editing build_poster.py.

    python3 check_layout.py                 # all built posters
    python3 check_layout.py eccv2026_poster_180x90.html

Column overflow is silent: a section that runs past the bottom of a column is
simply clipped by `.poster{overflow:hidden}`, so it looks fine in the HTML and
disappears in the PDF. This measures the real print layout in headless Chrome
and fails loudly if any column exceeds 100%.

Note: `#wrap` is `display:flex` on screen, which makes `.poster` shrink to the
viewport. Print uses `display:block`, so the probe forces block layout first —
otherwise every measurement is taken at the wrong width.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = Path(__file__).resolve().parent

PROBE = """<script>
window.addEventListener('load', function(){ try{
  var w=document.getElementById('wrap');
  document.querySelector('.poster').style.transform='none';
  w.style.height='auto'; w.style.display='block'; w.style.padding='0';
  var p=document.querySelector('.poster'), cols=document.querySelector('.cols');
  var avail=cols.getBoundingClientRect().height, out=[];
  function stack(el){                       // height of a flex column's children
    var used=0;
    [].slice.call(el.children).forEach(function(s){
      used+=s.getBoundingClientRect().height; });
    return used+(el.children.length-1)*parseFloat(getComputedStyle(el).rowGap||0);
  }
  [].slice.call(cols.children).forEach(function(col){
    var sub=col.querySelector('.subcols');
    if(!sub){ out.push(Math.round(stack(col)/avail*100)); return; }
    // A span-2 block: head sections plus the taller of the two sub-columns.
    var gap=parseFloat(getComputedStyle(col).rowGap||0), head=0;
    [].slice.call(col.children).forEach(function(s){
      if(s!==sub) head+=s.getBoundingClientRect().height+gap; });
    [].slice.call(sub.children).forEach(function(sc){
      out.push(Math.round((head+stack(sc))/avail*100)); });
  });
  var r=p.getBoundingClientRect();
  document.title='PROBE '+Math.round(r.width)+'x'+Math.round(r.height)
    +' '+out.join(',');
}catch(e){ document.title='PROBE ERR '+e.message; } });
</script>"""


def measure(html: Path) -> tuple[int, int, list[int]]:
    tmp = html.with_name("_check_" + html.name)
    tmp.write_text(html.read_text().replace("</body>", PROBE + "</body>"))
    try:
        r = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--virtual-time-budget=15000",
             "--dump-dom", f"file://{tmp.resolve()}"],
            capture_output=True, text=True, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    m = re.search(r"<title>PROBE (\d+)x(\d+) ([\d,]+)</title>", r.stdout)
    if not m:
        err = re.search(r"<title>(PROBE ERR[^<]*)</title>", r.stdout)
        raise RuntimeError(err.group(1) if err else "probe did not run")
    return int(m.group(1)), int(m.group(2)), [int(v) for v in m.group(3).split(",")]


def main() -> int:
    targets = ([Path(a) for a in sys.argv[1:]]
               or sorted(HERE.glob("eccv2026_poster_*.html")))
    bad = 0
    for html in targets:
        try:
            w, h, fills = measure(html)
        except RuntimeError as e:
            print(f"FAIL  {html.name}: {e}")
            bad += 1
            continue
        mm = f"{w/96*25.4:.0f} x {h/96*25.4:.0f} mm"
        orient = "landscape" if w > h else "portrait"
        over = [i + 1 for i, f in enumerate(fills) if f > 100]
        status = "OVERFLOW col " + ",".join(map(str, over)) if over else "ok"
        if over:
            bad += 1
        print(f"{'FAIL' if over else 'PASS'}  {html.name:30s} {mm:>16s} {orient:9s} "
              f"cols={fills}  {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
