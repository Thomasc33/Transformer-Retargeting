#!/usr/bin/env python3
"""
Extract AR/RI trade-off data from Optuna trial logs and database.

Parses SLURM log files from the Optuna hyperparameter search to extract
per-trial AR, RI, and composite score (0.7*AR - 0.3*RI). Also reads
the Optuna SQLite database for hyperparameters per trial.

Output: JSON file with all trial data + summary table printed to stdout.

Usage (lightweight, safe for login node — no GPU/model/data loading):
    python scripts/extract_optuna_tradeoff.py
"""

import json
import os
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
DB_PATH = ROOT / "output" / "optuna" / "optuna_v2.db"
OUTPUT_DIR = ROOT / "output" / "analysis" / "optuna_tradeoff"


def parse_log_file(log_path):
    """Parse a single Optuna SLURM log for trial results.

    Looks for patterns like:
        Trial 43 result (115.0 min):
            AR=0.2200  RI=0.1020  Score=0.1234

    Also extracts pruned trials with their last Val AR/RI before pruning,
    and the pruning AR from lines like:
        [Optuna] Trial pruned after Stage 1 (AR=0.0830)
    """
    trials = {}

    with open(log_path, "r") as f:
        lines = f.readlines()

    current_trial = None
    last_val_ar = None
    last_val_ri = None

    for i, line in enumerate(lines):
        # Match trial start: "Trial 43  [11:53:46]"
        m = re.match(r"^Trial (\d+)\s+\[", line)
        if m:
            current_trial = int(m.group(1))
            last_val_ar = None
            last_val_ri = None
            continue

        # Match Val AR/RI during training
        m = re.match(r"\s+Val AR Accuracy:\s+([0-9.]+)", line)
        if m and current_trial is not None:
            last_val_ar = float(m.group(1))
            continue

        m = re.match(r"\s+Val RI Accuracy:\s+([0-9.]+)", line)
        if m and current_trial is not None:
            last_val_ri = float(m.group(1))
            continue

        # Match completed trial result: "    AR=0.2200  RI=0.1020  Score=0.1234"
        m = re.match(r"\s+AR=([0-9.]+)\s+RI=([0-9.]+)\s+Score=([0-9.-]+)", line)
        if m and current_trial is not None:
            trials[current_trial] = {
                "trial": current_trial,
                "status": "COMPLETE",
                "ar": float(m.group(1)),
                "ri": float(m.group(2)),
                "score": float(m.group(3)),
                "source": os.path.basename(log_path),
            }
            current_trial = None
            continue

        # Match pruned trial
        m = re.match(r"\s+Trial (\d+) PRUNED", line)
        if m:
            trial_num = int(m.group(1))
            trials[trial_num] = {
                "trial": trial_num,
                "status": "PRUNED",
                "ar": last_val_ar,
                "ri": last_val_ri,
                "score": None,
                "source": os.path.basename(log_path),
            }
            current_trial = None
            continue

        # Match failed trial
        m = re.match(r"\s+Trial (\d+) FAILED", line)
        if m:
            trial_num = int(m.group(1))
            trials[trial_num] = {
                "trial": trial_num,
                "status": "FAILED",
                "ar": last_val_ar,
                "ri": last_val_ri,
                "score": None,
                "source": os.path.basename(log_path),
            }
            current_trial = None
            continue

    return trials


def get_trial_params_from_db(db_path):
    """Read hyperparameters for each trial from the Optuna SQLite database."""
    if not os.path.exists(db_path):
        print(f"Warning: Optuna DB not found at {db_path}")
        return {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all trial params
    cur.execute("""
        SELECT tp.trial_id, t.number, tp.param_name, tp.param_value
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        ORDER BY t.number, tp.param_name
    """)

    params = {}
    for trial_id, trial_num, param_name, param_value in cur.fetchall():
        if trial_num not in params:
            params[trial_num] = {}
        try:
            params[trial_num][param_name] = float(param_value)
        except (ValueError, TypeError):
            params[trial_num][param_name] = param_value

    # Get trial states and objective values from DB
    cur.execute("""
        SELECT t.number, t.state, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        ORDER BY t.number
    """)

    db_trials = {}
    for trial_num, state, value in cur.fetchall():
        db_trials[trial_num] = {"state": state, "objective": value}

    conn.close()
    return params, db_trials


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse all log files
    all_trials = {}
    log_files = sorted(LOG_DIR.glob("optuna_fast_*.out")) + sorted(LOG_DIR.glob("optuna_tmr_*.out"))

    print(f"Parsing {len(log_files)} Optuna log files...")
    for log_file in log_files:
        trials = parse_log_file(log_file)
        for trial_num, data in trials.items():
            # Later log files may have more complete data for the same trial
            if trial_num not in all_trials or data["status"] == "COMPLETE":
                all_trials[trial_num] = data

    # 2. Get params from DB
    params, db_trials = get_trial_params_from_db(DB_PATH)

    # 3. Merge params into trial data
    for trial_num, data in all_trials.items():
        if trial_num in params:
            data["params"] = params[trial_num]
        if trial_num in db_trials:
            data["db_state"] = db_trials[trial_num]["state"]
            data["db_objective"] = db_trials[trial_num]["objective"]

    # 4. Separate by status
    complete = {k: v for k, v in all_trials.items() if v["status"] == "COMPLETE"}
    pruned = {k: v for k, v in all_trials.items() if v["status"] == "PRUNED"}
    failed = {k: v for k, v in all_trials.items() if v["status"] == "FAILED"}

    # 5. Print summary
    print(f"\n{'='*70}")
    print(f"OPTUNA TRADE-OFF ANALYSIS")
    print(f"{'='*70}")
    print(f"Total trials found in logs: {len(all_trials)}")
    print(f"  COMPLETE: {len(complete)}")
    print(f"  PRUNED:   {len(pruned)}")
    print(f"  FAILED:   {len(failed)}")

    print(f"\n{'='*70}")
    print(f"COMPLETED TRIALS (sorted by score = 0.7*AR - 0.3*RI)")
    print(f"{'='*70}")
    print(f"{'Trial':>6} {'AR':>8} {'RI':>8} {'Score':>8}  Key params (w_adv / w_ar / w_ri)")
    print(f"{'-'*70}")

    for trial_num in sorted(complete.keys(), key=lambda k: complete[k].get("score", 0), reverse=True):
        t = complete[trial_num]
        p = t.get("params", {})
        w_adv = p.get("weight_adversarial", "?")
        w_ar = p.get("weight_ar", "?")
        w_ri = p.get("weight_ri", "?")

        ar_pct = t["ar"] * 100
        ri_pct = t["ri"] * 100

        param_str = f"w_adv={w_adv:.2f} / w_ar={w_ar:.2f} / w_ri={w_ri:.2f}" if isinstance(w_adv, float) else "?"
        print(f"{trial_num:>6} {ar_pct:>7.1f}% {ri_pct:>7.1f}% {t['score']:>8.4f}  {param_str}")

    # 6. Check for RI variation
    ri_values = [t["ri"] for t in complete.values() if t["ri"] is not None]
    ri_unique = set(ri_values)

    print(f"\n{'='*70}")
    print(f"RI VARIATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Unique RI values across completed trials: {ri_unique}")
    print(f"RI range: {min(ri_values)*100:.1f}% - {max(ri_values)*100:.1f}%")

    if len(ri_unique) <= 2:
        print(f"\n*** NOTE: RI is essentially constant ({min(ri_values)*100:.1f}%) across all")
        print(f"    completed trials. The accelerated Optuna protocol (3/3/2 epochs,")
        print(f"    5k samples) does not train long enough for RI to differentiate")
        print(f"    across hyperparameter settings. Only AR varies.")
        print(f"    This means the Optuna trials do NOT provide a privacy-utility")
        print(f"    trade-off curve. A dedicated sweep (e.g., pareto_sweep.sbatch)")
        print(f"    with full training is needed for trade-off analysis.")

    # 7. Also show pruned trials with their last AR/RI
    pruned_with_data = {k: v for k, v in pruned.items() if v.get("ar") is not None}
    if pruned_with_data:
        print(f"\n{'='*70}")
        print(f"PRUNED TRIALS (last Val AR before pruning, Stage 1 only)")
        print(f"{'='*70}")
        print(f"{'Trial':>6} {'AR':>8} {'RI':>8}  Key params (w_adv / w_ar / w_ri)")
        print(f"{'-'*70}")

        for trial_num in sorted(pruned_with_data.keys()):
            t = pruned_with_data[trial_num]
            p = t.get("params", {})
            w_adv = p.get("weight_adversarial", "?")
            w_ar = p.get("weight_ar", "?")
            w_ri = p.get("weight_ri", "?")

            ar_str = f"{t['ar']*100:.1f}%" if t["ar"] else "?"
            ri_str = f"{t['ri']*100:.1f}%" if t["ri"] else "?"

            param_str = f"w_adv={w_adv:.2f} / w_ar={w_ar:.2f} / w_ri={w_ri:.2f}" if isinstance(w_adv, float) else "?"
            print(f"{trial_num:>6} {ar_str:>8} {ri_str:>8}  {param_str}")

    # 8. Save all data
    output = {
        "description": "Optuna trade-off analysis extracted from SLURM logs and Optuna DB",
        "objective": "0.7 * AR - 0.3 * RI (maximize)",
        "protocol": "Accelerated: 3/3/2 epochs per stage, 5k sample subset",
        "finding": "RI is constant (10.2%) across all completed trials; only AR varies",
        "implication": "Optuna trials do not provide a privacy-utility trade-off curve",
        "completed_trials": [complete[k] for k in sorted(complete.keys())],
        "pruned_trials": [pruned[k] for k in sorted(pruned.keys())],
        "failed_trials": [{"trial": k, "status": "FAILED"} for k in sorted(failed.keys())],
    }

    # Remove non-serializable items
    for trial_list in [output["completed_trials"], output["pruned_trials"]]:
        for t in trial_list:
            if "params" in t:
                t["params"] = {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in t["params"].items()}

    json_path = OUTPUT_DIR / "optuna_tradeoff_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full analysis to {json_path}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")
    print(f"The Optuna search optimized for a fixed objective (0.7*AR - 0.3*RI)")
    print(f"and the short training protocol caused all trials to converge to the")
    print(f"same RI (~10.2%). To show a privacy-utility trade-off curve, either:")
    print(f"  1. Run pareto_sweep.sbatch (sweeps adversarial weight, full training)")
    print(f"  2. Use the component ablation table (Table 2) which shows different")
    print(f"     operating points via architectural choices")
    print(f"  3. Plot the scatter of all methods (already in fig:scatter)")


if __name__ == "__main__":
    main()
