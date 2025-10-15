"""
Results Manager
A small utility to standardize where evaluations save artifacts, metrics, and reports.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class ResultsManager:
    """Centralized helper for organizing evaluation results under a unified directory.

    Layout:
      <base_results_dir>/
        experiments/
          <experiment_name>/
            <experiment_id>/
              results.json
              metrics.json
              config.json
              metadata.json
              visualizations/
              artifacts/
              logs/
    """

    def __init__(self, base_results_dir: str = "results") -> None:
        self.base_dir = Path(base_results_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "experiments").mkdir(parents=True, exist_ok=True)

    def _get_git_commit(self) -> Optional[str]:
        """Return short git commit hash if available; else None."""
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            return commit
        except Exception:
            return None

    def init_experiment(self, experiment_name: str) -> Tuple[Path, str]:
        """Create a new experiment directory and return (exp_dir, experiment_id)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit = self._get_git_commit()
        exp_id = f"exp_{ts}{('_' + commit) if commit else ''}"

        root = self.base_dir / "experiments" / experiment_name / exp_id
        (root / "visualizations").mkdir(parents=True, exist_ok=True)
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)

        # Maintain a convenient latest symlink
        try:
            latest_link = self.base_dir / "experiments" / experiment_name / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(root.name)
        except Exception:
            # Symlinks may be restricted on some filesystems; ignore
            pass

        # Write minimal metadata
        metadata = {
            "experiment_name": experiment_name,
            "experiment_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "git_commit": commit,
        }
        with open(root / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return root, exp_id

    def save_results_bundle(self, results: Dict[str, Any], exp_dir: Path) -> None:
        """Save results.json, metrics.json, config.json, and an index file."""
        # Main bundle
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        if "metrics" in results:
            with open(exp_dir / "metrics.json", "w") as f:
                json.dump(results["metrics"], f, indent=2, default=str)
        if "config" in results:
            with open(exp_dir / "config.json", "w") as f:
                json.dump(results["config"], f, indent=2, default=str)

        # Lightweight index
        lines = [
            "# Experiment Results",
            f"- Experiment: {results.get('experiment_name', 'N/A')}",
            f"- ID: {results.get('experiment_id', 'N/A')}",
            f"- Timestamp: {results.get('timestamp', '')}",
            "",
            "## Files",
            "- results.json",
            "- metrics.json",
            "- config.json",
            "- metadata.json",
            "- visualizations/",
            "- artifacts/",
            "- logs/",
        ]
        with open(exp_dir / "README.md", "w") as f:
            f.write("\n".join(lines))

    def record_slurm_info(self, exp_dir: Path, job_id: Optional[str], script_path: Optional[str]) -> None:
        """Save Slurm submission info into metadata.json."""
        meta_path = exp_dir / "metadata.json"
        try:
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            metadata = {}
        if job_id:
            metadata["slurm_job_id"] = job_id
        if script_path:
            metadata["slurm_script_path"] = script_path
        meta_path.write_text(json.dumps(metadata, indent=2))

    def artifacts_dir(self, exp_dir: Path) -> Path:
        return exp_dir / "artifacts"

    def visualizations_dir(self, exp_dir: Path) -> Path:
        return exp_dir / "visualizations"

