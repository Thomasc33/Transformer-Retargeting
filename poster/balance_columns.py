#!/usr/bin/env python3
"""Measure each poster section and pick the best contiguous column split.

    python3 balance_columns.py eccv2026_poster_140x100.html

Sections are laid out in reading order, so a column plan should be a contiguous
partition of that order — anything else makes the eye jump. This measures every
section at print size in headless Chrome, then does an exact DP over contiguous
partitions to minimise the tallest column, and reports the resulting fills.
"""

from __future__ import annotations

import json
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
  var cols=document.querySelector('.cols');
  var avail=cols.getBoundingClientRect().height, hs=[], gap=0;
  [].slice.call(cols.children).forEach(function(col){
    gap=parseFloat(getComputedStyle(col).rowGap||0);
    [].slice.call(col.children).forEach(function(s){
      hs.push(Math.round(s.getBoundingClientRect().height)); });
  });
  document.title='PROBE '+JSON.stringify({avail:Math.round(avail),gap:gap,h:hs});
}catch(e){ document.title='PROBE ERR '+e.message; } });
</script>"""


def measure(html: Path) -> dict:
    tmp = html.with_name("_bal_" + html.name)
    tmp.write_text(html.read_text().replace("</body>", PROBE + "</body>"))
    try:
        r = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--virtual-time-budget=15000",
             "--dump-dom", f"file://{tmp.resolve()}"],
            capture_output=True, text=True, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    m = re.search(r"<title>PROBE (\{.*?\})</title>", r.stdout)
    if not m:
        raise RuntimeError("probe did not run")
    return json.loads(m.group(1))


def best_split(h: list[float], gap: float, k: int) -> tuple[list[list[int]], float]:
    """Minimise the tallest column over contiguous partitions into k groups."""
    n = len(h)

    def cost(i: int, j: int) -> float:            # sections [i, j)
        return sum(h[i:j]) + gap * (j - i - 1)

    INF = float("inf")
    # dp[c][i] = best max-column cost splitting h[i:] into c groups
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][n] = 0.0
    for c in range(1, k + 1):
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n + 1):
                if dp[c - 1][j] == INF:
                    continue
                v = max(cost(i, j), dp[c - 1][j])
                if v < dp[c][i]:
                    dp[c][i], cut[c][i] = v, j
    groups, i = [], 0
    for c in range(k, 0, -1):
        j = cut[c][i]
        groups.append(list(range(i, j)))
        i = j
    return groups, dp[k][0]


def best_packing(h: list[float], gap: float, k: int) -> tuple[list[list[int]], float]:
    """Minimise the tallest column over *any* grouping (subset DP over 2^n)."""
    from functools import lru_cache

    n = len(h)
    full = (1 << n) - 1
    cost = [0.0] * (1 << n)
    for m in range(1, 1 << n):
        b = m.bit_length() - 1
        rest = m & ~(1 << b)
        cost[m] = cost[rest] + h[b] + (gap if rest else 0.0)

    @lru_cache(maxsize=None)
    def solve(mask: int, c: int) -> tuple[float, int]:
        """Best tallest-column value for `mask` split into `c` groups."""
        if c == 1:
            return cost[mask], mask
        best, arg = float("inf"), 0
        low = mask & -mask                     # anchor to kill permutations
        sub = mask
        while sub:
            if sub & low:
                v = max(cost[sub], solve(mask & ~sub, c - 1)[0])
                if v < best:
                    best, arg = v, sub
            sub = (sub - 1) & mask
        return best, arg

    groups, mask = [], full
    for c in range(k, 0, -1):
        _, g = solve(mask, c)
        groups.append([i for i in range(n) if g >> i & 1])
        mask &= ~g
    return groups, solve(full, k)[0]


def main() -> int:
    import build_poster as bp

    html = Path(sys.argv[1] if len(sys.argv) > 1
                else "eccv2026_poster_140x100.html")
    page = next(k for k, v in bp.OUT_NAME.items() if v == html.name)
    ncol = bp.PAGES[page][2]
    order = [k for col in bp.COLUMN_PLANS[ncol] for k in col]

    d = measure(HERE / html)
    h, gap, avail = d["h"], d["gap"], d["avail"]
    for name, ht in zip(order, h):
        print(f"  {name:9s} {ht/avail*100:5.1f}%")
    print(f"total {sum(h)/avail*100:.1f}% of one column, "
          f"{sum(h)/avail*100/ncol:.1f}% average over {ncol}")

    for label, fn in (("contiguous", best_split), ("free", best_packing)):
        groups, worst = fn(h, gap, ncol)
        plan = [[order[i] for i in g] for g in groups]
        fills = [round((sum(h[i] for i in g) + gap * (len(g) - 1)) / avail * 100)
                 for g in groups]
        print(f"{label} plan  fills {fills}  tallest {worst/avail*100:.1f}%")
        print(json.dumps(plan))
    groups, worst = best_split(h, gap, ncol)
    plan = [[order[i] for i in g] for g in groups]
    fills = [round((sum(h[i] for i in g) + gap * (len(g) - 1)) / avail * 100)
             for g in groups]
    print("best contiguous plan:")
    print(json.dumps(plan, indent=4))
    print(f"fills {fills}   tallest {worst/avail*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
