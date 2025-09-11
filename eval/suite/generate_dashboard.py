#!/usr/bin/env python3
"""
Generate a single HTML dashboard (results.html at repo root) aggregating:
- All experiments from evaluation_suite/configs/experiments.yaml
- Completion status based on results/experiments/<exp>/*/results.json
- Links to metrics and config
- Embedded GIFs/images/videos from per-experiment visualizations and from results/visualizations

Usage:
  python evaluation_suite/generate_dashboard.py
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# After moving under eval/suite, repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_ROOT / "experiments"
STANDALONE_VIZ_DIR = RESULTS_ROOT / "visualizations"
CONFIG_PATH = REPO_ROOT / "eval" / "suite" / "configs" / "experiments.yaml"
OUTPUT_HTML = REPO_ROOT / "results.html"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
VID_EXTS = {".mp4", ".webm"}


def load_config() -> Dict[str, Any]:
    if yaml is None:
        # Minimal fallback: return empty config to still build the page
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_runs_for_experiment(exp_name: str) -> List[Path]:
    exp_dir = EXPERIMENTS_DIR / exp_name
    if not exp_dir.exists():
        return []
    # Return run directories sorted by mtime desc
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name != "latest"]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def latest_run_for_experiment(exp_name: str) -> Optional[Path]:
    runs = find_runs_for_experiment(exp_name)
    return runs[0] if runs else None


def collect_media(exp_run_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find media files in visualizations/ and artifacts/ of a run."""
    imgs: List[Path] = []
    vids: List[Path] = []
    for sub in ["visualizations", "artifacts"]:
        d = exp_run_dir / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    ext = p.suffix.lower()
                    if ext in IMG_EXTS:
                        imgs.append(p)
                    elif ext in VID_EXTS:
                        vids.append(p)
    return imgs, vids


def collect_standalone_visualizations() -> Tuple[List[Path], List[Path]]:
    imgs: List[Path] = []
    vids: List[Path] = []
    if not STANDALONE_VIZ_DIR.exists():
        return imgs, vids
    for p in STANDALONE_VIZ_DIR.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMG_EXTS:
                imgs.append(p)
            elif ext in VID_EXTS:
                vids.append(p)
    # Sort
    imgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return imgs, vids


def relpath_html(p: Path) -> str:
    # Make path relative to repo root for href/src
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(p)


def render_header() -> str:
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Evaluation Results Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a; /* slate-900 */
      --card: #111827; /* gray-900 */
      --muted: #94a3b8; /* slate-400 */
      --text: #e5e7eb; /* gray-200 */
      --accent: #22d3ee; /* cyan-400 */
      --accent2: #a78bfa; /* violet-400 */
      --ok: #34d399; /* green-400 */
      --warn: #fbbf24; /* amber-400 */
      --bad: #f87171; /* red-400 */
    }}
    html, body {{ background: var(--bg); color: var(--text); }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 24px; }}
    h1, h2, h3 {{ margin: 0.6em 0 0.4em; }}
    .meta {{ color: var(--muted); font-size: 0.9em; }}
    .exp {{ background: var(--card); border: 1px solid #1f2937; border-radius: 10px; padding: 14px; margin: 16px 0; box-shadow: 0 1px 0 #000; }}
    .status {{ font-weight: 600; }}
    .completed {{ color: var(--ok); }}
    .pending {{ color: var(--warn); }}
    .failed {{ color: var(--bad); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); grid-gap: 14px; }}
    .thumb {{ border: 1px solid #1f2937; border-radius: 8px; padding: 8px; background: #0b1220; }}
    img {{ max-width: 100%; height: auto; display: block; border-radius: 6px; }}
    video {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    .small {{ font-size: 0.85em; color: var(--muted); }}
    .toc a {{ margin-right: 12px; color: var(--accent); text-decoration: none; }}
    .toc a:hover {{ color: var(--accent2); }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; background: #0b1220; margin-left: 6px; font-size: 0.85em; color: var(--accent); border: 1px solid #1f2937; }}
    .footer {{ margin-top: 24px; color: var(--muted); font-size: 0.9em; }}
    .missing {{ padding: 8px; border: 1px dashed #334155; border-radius: 8px; color: var(--muted); }}
    .files {{ margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Evaluation Results Dashboard</h1>
  <div class=\"meta\">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  <p class=\"small\">This page aggregates experiment results, status (to-do list), and visualizations (GIFs/images/videos). It auto-refreshes after each experiment run.</p>
  <hr/>
"""


def render_footer() -> str:
    return """
  <div class=\"footer\">Made by evaluation_suite/generate_dashboard.py — Missing items are indicated below each experiment.</div>
</body>
</html>
"""


def render_experiment_section(title: str, experiments: Dict[str, Dict[str, Any]]) -> str:
    html = [f"<h2 id='{title.replace(' ', '_').lower()}'>{title}</h2>"]
    for exp_name, exp_cfg in experiments.items():
        latest = latest_run_for_experiment(exp_name)
        status = "Completed" if latest else "Pending"
        status_class = "completed" if latest else "pending"
        html.append(f"<div class='exp'>")
        html.append(f"<h3>{exp_name} <span class='badge'>{exp_cfg.get('estimated_time','')}</span></h3>")
        html.append(f"<div class='meta'>{exp_cfg.get('description','')}</div>")
        html.append(f"<div class='status {status_class}'>Status: {status}</div>")

        if latest:
            results_path = latest / "results.json"
            metrics_path = latest / "metrics.json"
            cfg_path = latest / "config.json"
            files_missing = []
            links = []
            for p in [results_path, metrics_path, cfg_path]:
                if p.exists():
                    links.append(f"<a href='{relpath_html(p)}' target='_blank'>{p.name}</a>")
                else:
                    files_missing.append(p.name)
            if links:
                html.append("<div class='files'>Files: " + " | ".join(links) + "</div>")
            if files_missing:
                html.append("<div class='missing small'>Missing: " + ", ".join(files_missing) + "</div>")

            imgs, vids = collect_media(latest)
            if imgs or vids:
                html.append("<div class='grid'>")
                for img in imgs[:12]:
                    html.append(f"<div class='thumb'><a href='{relpath_html(img)}' target='_blank'><img src='{relpath_html(img)}' alt='{img.name}'/></a><div class='small'>{img.name}</div></div>")
                for vid in vids[:6]:
                    html.append(f"<div class='thumb'><video controls src='{relpath_html(vid)}'></video><div class='small'>{vid.name}</div></div>")
                html.append("</div>")
            else:
                html.append("<div class='missing small'>No visualizations found in latest run.</div>")
        else:
            html.append("<div class='missing small'>No runs found yet. This experiment is pending.</div>")
        html.append("</div>")
    return "\n".join(html)


def main():
    config = load_config()

    cats = [
        ("Primary Experiments", config.get("primary_experiments", {})),
        ("Loss Analysis Experiments", config.get("loss_analysis_experiments", {})),
        ("Pretraining Experiments", config.get("pretraining_experiments", {})),
        ("Robustness Experiments", config.get("robustness_experiments", {})),
        ("Generalization & Efficiency Experiments", config.get("generalization_experiments", {})),
        ("Qualitative Experiments", config.get("qualitative_experiments", {})),
    ]

    parts: List[str] = [render_header()]

    # TOC
    parts.append("<div class='toc'><strong>Jump to:</strong> " + " ".join([f"<a href='#" + t.replace(' ', '_').lower() + f"'>{t}</a>" for t, _ in cats]) + "</div>")

    # Sections per category
    for title, exps in cats:
        parts.append(render_experiment_section(title, exps))
        parts.append("<hr/>")

    # Standalone visualizations section
    parts.append("<h2 id='standalone_visualizations'>Standalone Visualizations</h2>")
    if STANDALONE_VIZ_DIR.exists():
        imgs, vids = collect_standalone_visualizations()
        if imgs or vids:
            parts.append("<div class='grid'>")
            for img in imgs[:24]:
                parts.append(f"<div class='thumb'><a href='{relpath_html(img)}' target='_blank'><img src='{relpath_html(img)}' alt='{img.name}'/></a><div class='small'>{img.name}</div></div>")
            for vid in vids[:12]:
                parts.append(f"<div class='thumb'><video controls src='{relpath_html(vid)}'></video><div class='small'>{vid.name}</div></div>")
            parts.append("</div>")
        else:
            parts.append("<p class='small'>No standalone visualizations found under results/visualizations.</p>")
    else:
        parts.append("<p class='small'>No standalone visualizations directory yet.</p>")

    parts.append(render_footer())

    OUTPUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote dashboard: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

