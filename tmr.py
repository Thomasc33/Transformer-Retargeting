#!/usr/bin/env python3
"""
Primary entrypoint (TMR Orchestrator)
- Defaults to interactive mode when no args are supplied
- Provides consistent commands for:
  * eval: run experiment sets or a single experiment
  * dash: regenerate the HTML dashboard
  * clean-results: remove old evaluation outputs (keeps training/saved models)
  * validate: quick environment checks (torch, CUDA, data paths)
  * migrate-models: optional migration of saved models/outputs to data/models*
  * refactor-structure: move legacy dirs into target structure

This is a facade over the unified evaluation suite. It does NOT require YAML.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_MODEL_PATH = ROOT / 'data' / 'models_output' / 'model.pth'

# Helpers

def _merge_move(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                _merge_move(item, target)
            else:
                shutil.move(str(item), str(target))
        else:
            if target.exists():
                # Avoid overwrite; keep existing
                continue
            shutil.move(str(item), str(target))
    # Remove now-empty source
    try:
        src.rmdir()
    except Exception:
        shutil.rmtree(src, ignore_errors=True)


def _ensure_backup_model(src_path: Path) -> Path:
    """Create a timestamped backup of src_path if it exists. Returns backup path or src_path if missing."""
    if not src_path or not src_path.exists():
        return src_path
    ts = __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_dir = src_path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{src_path.stem}.bak-{ts}{src_path.suffix}"
    shutil.copy2(str(src_path), str(backup))
    print(f"[TMR] Backed up reference model to: {backup}")
    return backup


# Lazy imports to keep startup fast

def cmd_eval(args):
    """Run evaluation jobs via the evaluation suite."""
    # Dry-run / list support
    if getattr(args, 'list', False) or getattr(args, 'dry_run', False):
        try:
            from eval.suite.run_all_evaluations import collect_all_experiments
        except Exception:
            from evaluation_suite.run_all_evaluations import collect_all_experiments
        exps = collect_all_experiments(args.set)
        names = list(exps.keys())
        print(f"Experiment set '{args.set}': {len(names)} experiments")
        for n in names:
            print(f" - {n}")
        if args.dry_run:
            print("[EVAL] Dry run: no jobs were submitted.")
        return 0

    if args.one:
        try:
            from eval.suite.run_single_experiment import main as run_one
        except Exception:
            from evaluation_suite.run_single_experiment import main as run_one
        sys.argv = [sys.argv[0], '--experiment', args.one]
        return run_one()
    else:
        try:
            from eval.suite.run_all_evaluations import main as run_all
        except Exception:
            from evaluation_suite.run_all_evaluations import main as run_all
        sys.argv = [sys.argv[0], '--set', args.set]
        if args.local:
            sys.argv.extend(['--local'])
        return run_all()


def cmd_dash(args):
    try:
        from eval.suite.generate_dashboard import main as dash
    except Exception:
        from evaluation_suite.generate_dashboard import main as dash
    return dash()


def cmd_clean_results(args):
    """Remove old evaluation results while preserving training/saved models."""
    targets = [ROOT / 'eval' / 'results', ROOT / 'results', ROOT / 'evaluation_suite' / 'results', ROOT / 'eval' / 'suite' / 'results']
    removed = []
    for t in targets:
        if t.exists():
            try:
                shutil.rmtree(t)
                removed.append(str(t))
            except Exception as e:
                print(f"Warning: failed to remove {t}: {e}")
    # Recreate minimal structure so tools don't fail
    (ROOT / 'eval' / 'results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'eval' / 'results' / 'experiments').mkdir(parents=True, exist_ok=True)
    print("Removed old results from:")
    for r in removed:
        print(f" - {r}")
    # Refresh dashboard so it's clearly empty with Missing messages
    try:
        from eval.suite.generate_dashboard import main as dash
    except Exception:
        try:
            from evaluation_suite.generate_dashboard import main as dash
        except Exception:
            dash = None
    if dash:
        try:
            dash()
        except Exception:
            pass



def cmd_train(args):
    """Run training locally or submit to Slurm.
    Tasks: retarget -> src/training/main.py
           pretrain -> src/training/pretrain.py
           sgn      -> src/training/train_sgn.py
           mixformer-> src/training/pretrain.py (MLM pretraining using Mixformer encoder)
           pmr/dmr  -> requires --entry to point to training script
    """
    import subprocess, textwrap, tempfile, shutil as _shutil

    task = (args.task or 'retarget').lower()
    # Default mapping
    script_map = {
        'retarget': ROOT / 'src' / 'training' / 'main.py',
        'pretrain': ROOT / 'src' / 'training' / 'pretrain.py',
        'sgn': ROOT / 'src' / 'training' / 'train_sgn.py',
        'mixformer': ROOT / 'src' / 'training' / 'pretrain.py',  # trains mixformer encoder via MLM
    }
    entry = None
    if task in script_map:
        entry = script_map[task]
    else:
        # pmr/dmr/custom fall back to --entry
        if not getattr(args, 'entry', None):
            print(f"No built-in training entry for task='{task}'. Provide --entry pointing to your training script (.py or .bash).")
            return 1
        entry = Path(args.entry)
        if not entry.is_absolute():
            entry = ROOT / entry

    if not entry.exists():
        print(f"Training entry not found: {entry}")
        return 1

    # Determine how to invoke (python vs bash)
    use_bash = entry.suffix in {'.sh', '.bash'}

    extra = list(getattr(args, 'extra', []) or [])

    # Safeguards and defaults for retarget training
    if task == 'retarget':
        # Ensure we don't overwrite the reference model; back it up and resume from it if not specified
        if DEFAULT_MODEL_PATH.exists():
            _ensure_backup_model(DEFAULT_MODEL_PATH)
            if '--resume-from' not in extra:
                extra.extend(['--resume-from', str(DEFAULT_MODEL_PATH)])
        # Ensure output path is unique by default
        if '--output-model-path' not in extra:
            ts = __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')
            default_out = ROOT / 'data' / 'models_output' / f'model_resume_run-{ts}.pth'
            default_out.parent.mkdir(parents=True, exist_ok=True)
            extra.extend(['--output-model-path', str(default_out)])

    if args.local:
        cmd = (["bash", str(entry)] if use_bash else [sys.executable, str(entry)]) + extra
        print("Running locally:", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(ROOT))

    # Slurm submission
    if not _shutil.which('sbatch'):
        print("sbatch not found. Printing the command you can run manually:")
        print(f"python {entry} {' '.join(extra)}")
        return 1

    gpus = getattr(args, 'gpus', 4)
    nodes = getattr(args, 'nodes', 1)
    time = getattr(args, 'time', '24:00:00')
    part = getattr(args, 'partition', None)
    email = getattr(args, 'email', None)
    cpus = getattr(args, 'cpus_per_task', 4)

    job_name = f"tmr-{task}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={'128GB' if gpus >= 4 else '64GB'}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={ROOT / 'logs' / 'slurm' / (job_name + '-%j.out')}",
    ]
    if part:
        lines.append(f"#SBATCH --partition={part}")
    if email:
        lines += [
            f"#SBATCH --mail-user={email}",
            "#SBATCH --mail-type=BEGIN,END,FAIL",
        ]

    cmd_str = (f"bash \"{entry}\"" if use_bash else f"python \"{entry}\"")
    body = textwrap.dedent(f"""
    set -euo pipefail
    module load pytorch/2.3.0-cuda12.1 || true
    cd "{ROOT}"
    echo "[TMR] Starting {task} at $(date) on $(hostname)"
    {cmd_str} {' '.join(extra)}
    echo "[TMR] Finished at $(date)"
    """)
    script_text = "\n".join(lines) + "\n\n" + body

    (ROOT / 'logs' / 'slurm').mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.sbatch', dir=str(ROOT / 'logs' / 'slurm')) as fh:
        fh.write(script_text)
        sbatch_path = fh.name
    print("Submitting Slurm job:", sbatch_path)
    return subprocess.call(['sbatch', sbatch_path], cwd=str(ROOT))


def cmd_preprocess(args):
    """List or run preprocessing scripts under data/{ntu,ntu120,etri}."""
    import subprocess
    roots = {
        'ntu': ROOT / 'data' / 'ntu',
        'ntu120': ROOT / 'data' / 'ntu120',
        'etri': ROOT / 'data' / 'etri',
    }

    def list_scripts(d):
        if not d.exists():
            return []
        return sorted([p.name for p in d.glob('*.py')])

    if getattr(args, 'list', False) or not getattr(args, 'dataset', None):
        print("Available preprocessing scripts:")
        for k, d in roots.items():
            print(f"- {k}:")
            for name in list_scripts(d):
                print(f"   {name}")
        return 0

    ds = args.dataset
    droot = roots.get(ds)
    if not droot or not droot.exists():
        print(f"Dataset folder not found: {ds} -> {droot}")
        return 1

    if args.run:
        script = Path(args.run)
        if not script.is_absolute():
            script = droot / script
        if not script.exists():
            print(f"Script not found: {script}")
            return 1
        cmd = [sys.executable, str(script)]
        print("Running:", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(ROOT))

    # Default: list dataset-specific scripts
    for name in list_scripts(droot):
        print(name)
    return 0


def cmd_validate(args):
    """Quick environment and data checks."""
    ok = True
    try:
        import torch  # type: ignore
        print(f"Python: {sys.executable}")
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"Torch import failed: {e}")
        ok = False

    # Basic data path checks
    data_roots = [ROOT / 'data' / 'nturgbd_raw', ROOT / 'data' / 'ntu', ROOT / 'data' / 'ntu120']
    for d in data_roots:
        print(f"Data path {d}: {'OK' if d.exists() else 'MISSING'}")

    print("Validation: {}".format("OK" if ok else "issues detected"))
    return 0 if ok else 1


def cmd_migrate_models(args):
    """Plan and migrate saved models to data/models* dirs with backward-compatible fallbacks."""
    # New targets
    models_dir = ROOT / 'data' / 'models'
    outputs_dir = ROOT / 'data' / 'models_output'
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Candidates to move (if present)
    candidates = [
        (ROOT / 'trained_models', models_dir),
        (ROOT / 'output', outputs_dir),
    ]
    for src, dst_root in candidates:
        if not src.exists():
            continue
        try:
            dst = dst_root / src.name
            if dst.exists():
                print(f"Skipping move: destination exists {dst}")
                continue
            print(f"Moving {src} -> {dst}")
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"Warning: could not move {src}: {e}")

    print("Model migration complete. eval_model.py includes fallbacks to look in data/models* if old paths are missing.")


def cmd_smoke(args):
    """Smoke test everything: importability and syntax across key modules.
    - Compiles Python files under src/ and eval/suite/ and this orchestrator.
    - Checks presence of critical paths and default model.
    - Does NOT run heavy training/evaluation.
    """
    import compileall
    ok = True
    targets = [ROOT / 'src', ROOT / 'eval' / 'suite', ROOT / 'tmr.py']
    for t in targets:
        try:
            if t.is_file():
                print(f"[SMOKE] Compiling file: {t}")
                compileall.compile_file(str(t), force=True, quiet=1)
            else:
                print(f"[SMOKE] Compiling dir: {t}")
                compileall.compile_dir(str(t), force=True, quiet=1)
        except Exception as e:
            print(f"[SMOKE] Compile failed for {t}: {e}")
            ok = False
    # Light CLI checks (no heavy runs) - only orchestrator help to avoid heavy imports
    import subprocess
    checks = [
        [sys.executable, str(ROOT / 'tmr.py'), '--help'],
        [sys.executable, str(ROOT / 'tmr.py'), 'train', '--help'],
        [sys.executable, str(ROOT / 'tmr.py'), 'eval', '--help'],
        [sys.executable, str(ROOT / 'tmr.py'), 'preprocess', '--help'],
        [sys.executable, str(ROOT / 'tmr.py'), 'pipeline', '--help'],
    ]
    for cmd in checks:
        try:
            print(f"[SMOKE] CLI: {' '.join(str(c) for c in cmd)}")
            subprocess.run(cmd, cwd=str(ROOT), check=False, timeout=30)
        except Exception as e:
            print(f"[SMOKE] CLI failed: {e}")


def _results_status(exp_name: str) -> str:
    """Return 'DONE' if results.json exists under latest for this experiment, else 'MISSING'."""
    candidates = [ROOT/'eval'/'results'/'experiments'/exp_name, ROOT/'results'/'experiments'/exp_name]
    for base in candidates:
        latest = base/'latest'
        if latest.exists() and (latest/'results.json').exists():
            return 'DONE'
        # Also consider any exp directory with results.json
        if base.exists():
            for child in base.iterdir():
                if child.is_dir() and (child/'results.json').exists():
                    return 'DONE'
    return 'MISSING'


def cmd_state(args):
    """Print current state: model presence and experiment completion status."""
    # Models
    print("[STATE] Models:")
    print(f" - Default transformer model: {DEFAULT_MODEL_PATH} -> {'OK' if DEFAULT_MODEL_PATH.exists() else 'MISSING'}")
    # Gather referenced model/eval model paths from primary experiments
    try:
        from eval.suite.experiments.primary import PrimaryExperiments
    except Exception:
        PrimaryExperiments = None
    model_paths = set()
    if PrimaryExperiments:
        cfgs = PrimaryExperiments.get_experiment_configs()
        for cfg in cfgs.values():
            for section_key in ('models','eval_models'):
                section = cfg.get(section_key, {})
                for _, entry in section.items():
                    p = entry.get('path')
                    if p and p != 'raw':
                        model_paths.add(p)
    for p in sorted(model_paths):
        path = Path(p)
        if not path.is_absolute():
            path = ROOT / p
        print(f" - Ref: {path} -> {'OK' if path.exists() else 'MISSING'}")

    # Dataset presence checks
    print("[STATE] Datasets (paired subsets):")
    dataset_checks = {
        'ntu_cv': [
            'data/ntu_cv_paired_10000_2000.pt',
            'data/ntu_cv_paired_comprehensive.pt',
            'data/ntu/pretraining_data_cv_comprehensive.pt',
            'data/ntu/pretraining_data.pt',
        ],
        'ntu_cs': [
            'data/ntu_cs_paired_10000_2000.pt',
            'data/ntu_cs_paired_comprehensive.pt',
            'data/ntu/pretraining_data.pt',
        ],
        'ntu120_cv': [
            'data/ntu120_cv_paired_10000_2000.pt',
            'data/ntu120_cv_paired_comprehensive.pt',
        ],
        'ntu120_cs': [
            'data/ntu120_cs_paired_10000_2000.pt',
            'data/ntu120_cs_paired_comprehensive.pt',
        ],
        'etri_cv': [
            'data/etri_paired_data.pt',
            'data/etri/pretraining_data.pt',
        ]
    }
    for name, paths in dataset_checks.items():
        found = next((p for p in paths if (ROOT/p).exists()), None)
        print(f" - {name}: {'OK' if found else 'MISSING'}{f' ({found})' if found else ''}")

    # Experiments status
    try:
        from eval.suite.run_all_evaluations import collect_all_experiments
        exps = collect_all_experiments('critical')
    except Exception:
        exps = {}
    if exps:
        print("[STATE] Experiments (critical set):")
        completed = 0
        missing = []
        for name in sorted(exps.keys()):
            status = _results_status(name)
            if status == 'DONE':
                completed += 1
            else:
                missing.append(name)
            print(f" - {name}: {status}")
        print(f"[STATE] Summary: {completed} done, {len(missing)} missing")
    else:
        print("[STATE] Experiments: could not import collect_all_experiments; skip status.")
    return 0




def cmd_quick(args):
    """Quick mode: ensure minimal prerequisites, then queue critical evaluations via Slurm and refresh dashboard.
    - If default model exists, set EVAL_DEFAULT_MODEL for child processes.
    - Does not modify training outputs; creates backup of reference model.
    """
    if DEFAULT_MODEL_PATH.exists():
        # Check if backup already exists to avoid multiple backups
        backup_dir = ROOT / 'data' / 'models_output'
        existing_backups = list(backup_dir.glob('model.bak-*.pth'))

        if not existing_backups:
            # Only create backup if none exists
            backup_path = _ensure_backup_model(DEFAULT_MODEL_PATH)
            print(f"[TMR] Backed up reference model to: {backup_path}")
        else:
            print(f"[TMR] Backup already exists, skipping: {existing_backups[0].name}")

        os.environ['EVAL_DEFAULT_MODEL'] = str(DEFAULT_MODEL_PATH)
        print(f"[QUICK] EVAL_DEFAULT_MODEL set -> {DEFAULT_MODEL_PATH}")
    # Submit critical set via suite
    rc = cmd_eval(argparse.Namespace(one=None, set='critical', local=False))
    # Refresh dashboard regardless
    try:
        cmd_dash(argparse.Namespace())
    except Exception:
        pass
    return rc


def cmd_pipeline(args):
    """Queue a dependent Slurm pipeline: [optional pretrain] -> retarget train -> evaluations -> dashboard.
    Uses sbatch --dependency to guarantee ordering while you're away.
    """
    import subprocess, tempfile, textwrap
    if not shutil.which('sbatch'):
        print("[PIPELINE] sbatch not found on PATH — cannot submit pipeline.")
        return 1
    (ROOT / 'logs' / 'slurm').mkdir(parents=True, exist_ok=True)

    submitted_ids = []

    def submit_job(name: str, body_cmd: str, template: dict = None, depends: str = None):
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --gres=gpu:{template.get('gpus', 1) if template else 1}",
            f"#SBATCH --cpus-per-task={template.get('cpus', 4) if template else 4}",
            f"#SBATCH --mem={template.get('mem', '128GB' if template and template.get('gpus', 1) >= 4 else '64GB') if template else '64GB'}",
            f"#SBATCH --time={template.get('time', '12:00:00') if template else '12:00:00'}",
            f"#SBATCH --output={ROOT / 'logs' / 'slurm' / (name + '-%j.out')}",
        ]
        if args.partition:
            lines.append(f"#SBATCH --partition={args.partition}")
        if args.email:
            lines += [f"#SBATCH --mail-user={args.email}", "#SBATCH --mail-type=BEGIN,END,FAIL"]
        dep_part = f" --dependency=afterok:{depends}" if depends else ""
        script = "\n".join(lines) + "\n\n" + textwrap.dedent(f"""
        set -euo pipefail
        module load pytorch/2.3.0-cuda12.1 || true
        cd "{ROOT}"
        echo "[PIPELINE] {name} start $(date)"
        {body_cmd}
        echo "[PIPELINE] {name} done $(date)"
        """)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.sbatch', dir=str(ROOT / 'logs' / 'slurm')) as fh:
            fh.write(script)
            path = fh.name
        cmd = ['sbatch'] + ([f"--dependency=afterok:{depends}"] if depends else []) + [path]
        out = subprocess.check_output(cmd, cwd=str(ROOT), text=True).strip()
        print(out)
        # Extract Job ID (typically last token)
        jid = out.split()[-1]
        submitted_ids.append((name, jid))
        return jid

    # Optional pretrain
    last_id = None
    if args.pretrain:
        pre_cmd = f"python src/training/pretrain.py {' '.join(args.pretrain_extra or [])}"
        last_id = submit_job('tmr-pretrain', pre_cmd, template={'gpus': args.gpus, 'cpus': args.cpus, 'time': args.pretrain_time})

    # Retarget training (resume-safe)
    if DEFAULT_MODEL_PATH.exists():
        _ensure_backup_model(DEFAULT_MODEL_PATH)
    train_extra = list(args.train_extra or [])
    if '--resume-from' not in train_extra and DEFAULT_MODEL_PATH.exists():
        train_extra += ['--resume-from', str(DEFAULT_MODEL_PATH)]
    if '--output-model-path' not in train_extra:
        ts = __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')
        out_path = ROOT / 'data' / 'models_output' / f'model_resume_run-{ts}.pth'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        train_extra += ['--output-model-path', str(out_path)]
    # Add teacher forcing = 0 and 5 epochs for long training
    if '--teacher-forcing-ratio' not in ' '.join(train_extra):
        train_extra += ['--teacher-forcing-ratio', '0.0']
    if '--epochs' not in ' '.join(train_extra):
        train_extra += ['--epochs', '5']
    train_cmd = f"python src/training/main.py {' '.join(train_extra)}"
    last_id = submit_job('tmr-train-retarget', train_cmd, template={'gpus': args.gpus, 'cpus': args.cpus, 'time': args.train_time}, depends=last_id)

    # Evaluations (critical set)
    eval_cmd = f"python eval/suite/run_all_evaluations.py"
    last_id = submit_job('tmr-eval-critical', eval_cmd, template={'gpus': 1, 'cpus': 2, 'time': args.eval_time}, depends=last_id)

    # Dashboard refresh
    dash_cmd = f"python eval/suite/generate_dashboard.py"
    last_id = submit_job('tmr-dash', dash_cmd, template={'gpus': 0, 'cpus': 1, 'time': '01:00:00'}, depends=last_id)

    print("[PIPELINE] Submitted jobs:")
    for name, jid in submitted_ids:
        print(f" - {name}: {jid}")
    return 0

def cmd_refactor_structure(args):
    """Move legacy directories into target structure: data, configs, docs, eval, logs, src."""
    moves = [
        (ROOT / 'slurm_out', ROOT / 'logs' / 'slurm'),
        (ROOT / 'overnight_logs', ROOT / 'logs' / 'overnight'),
        (ROOT / 'experiments', ROOT / 'docs' / 'experiments'),
        (ROOT / 'evaluation_results', ROOT / 'docs' / 'evaluation_results'),
        (ROOT / 'bash', ROOT / 'docs' / 'scripts'),
        (ROOT / 'utils', ROOT / 'src' / 'utils'),
        (ROOT / 'visualize', ROOT / 'src' / 'utils' / 'visualize'),
        (ROOT / 'test_visualizations', ROOT / 'docs' / 'examples' / 'test_visualizations'),
    ]
    for src, dst in moves:
        if src.exists():
            print(f"Moving {src} -> {dst}")
            _merge_move(src, dst)
    print("Refactor complete. You may need to adjust imports for moved modules (utils/visualize).")


def interactive_menu():
    print("TMR Orchestrator (interactive)")
    print("1) Evaluate: critical set")
    print("2) Evaluate: all")
    print("3) Evaluate: one experiment")
    print("4) Dashboard: regenerate")
    print("5) Clean results (delete evaluation outputs)")
    print("6) Validate environment")
    print("7) Train (local or Slurm)")
    print("8) Preprocess data")
    print("9) Migrate models (trained_models/output -> data/models*)")
    print("10) Refactor structure (move legacy dirs)")
    print("0) Exit")
    choice = input("> ").strip()
    if choice == '1':
        return cmd_eval(argparse.Namespace(one=None, set='critical', local=False))
    if choice == '2':
        return cmd_eval(argparse.Namespace(one=None, set='complete', local=False))
    if choice == '3':
        exp = input("Experiment name: ").strip()
        return cmd_eval(argparse.Namespace(one=exp, set='critical', local=False))
    if choice == '4':
        return cmd_dash(argparse.Namespace())
    if choice == '5':
        return cmd_clean_results(argparse.Namespace())
    if choice == '6':
        return cmd_validate(argparse.Namespace())
    if choice == '7':
        task = input("Train task [retarget|pretrain|sgn|mixformer|pmr|dmr|custom] (default retarget): ").strip() or 'retarget'
        local = input("Run locally? [y/N]: ").strip().lower() == 'y'
        entry = input("Custom entry (.py or .bash, used if task=custom) [empty for none]: ").strip() or None
        return cmd_train(argparse.Namespace(task=task, entry=entry, local=local, gpus=4, nodes=1, time='24:00:00', partition=None, email=None, cpus_per_task=4, extra=[]))
    if choice == '8':
        return cmd_preprocess(argparse.Namespace(dataset=None, list=True, run=None))
    if choice == '9':
        return cmd_migrate_models(argparse.Namespace())
    if choice == '10':
        return cmd_refactor_structure(argparse.Namespace())
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TMR unified entrypoint")
    sub = p.add_subparsers(dest='cmd')

    p_eval = sub.add_parser('eval', help='Run evaluation jobs')
    p_eval.add_argument('--set', choices=['critical','complete','quick','paper_ready'], default='critical')
    p_eval.add_argument('--one', type=str, help='Run a single experiment by name')
    p_eval.add_argument('--local', action='store_true', help='Run locally instead of Slurm')
    p_eval.add_argument('--dry-run', action='store_true', help='Do not submit/run; just print what would be executed')
    p_eval.add_argument('--list', action='store_true', help='List experiments in the selected set and exit')

    p_train = sub.add_parser('train', help='Run training (local or Slurm)')
    p_train.add_argument('--task', choices=['retarget','pretrain','sgn','mixformer','pmr','dmr','custom'], default='retarget')
    p_train.add_argument('--entry', type=str, default=None, help='Custom training entry (.py or .bash); used when --task=custom or to override mapping')
    p_train.add_argument('--local', action='store_true', help='Run locally instead of Slurm')
    p_train.add_argument('--gpus', type=int, default=4)
    p_train.add_argument('--nodes', type=int, default=1)
    p_train.add_argument('--time', type=str, default='24:00:00')
    p_train.add_argument('--partition', type=str, default=None)
    p_train.add_argument('--email', type=str, default='tcarr23@charlotte.edu')
    p_train.add_argument('--cpus-per-task', dest='cpus_per_task', type=int, default=4)
    p_train.add_argument('extra', nargs='*', help='Extra args passed to training script')

    p_prep = sub.add_parser('preprocess', help='List or run preprocessing steps')
    p_prep.add_argument('--dataset', choices=['ntu','ntu120','etri'], default=None)
    p_prep.add_argument('--list', action='store_true', help='List available preprocessing scripts')
    p_prep.add_argument('--run', type=str, default=None, help='Run a specific script path')

    sub.add_parser('smoke', help='Fast syntax/import smoke test for src/, eval/suite/, and tmr.py')
    sub.add_parser('quick', help='Quick mode: set defaults, submit critical evals, refresh dashboard')

    p_pipe = sub.add_parser('pipeline', help='Submit dependent Slurm pipeline (pretrain -> train -> eval -> dash)')
    p_pipe.add_argument('--gpus', type=int, default=4)
    p_pipe.add_argument('--cpus', type=int, default=4)
    p_pipe.add_argument('--partition', type=str, default=None)
    p_pipe.add_argument('--email', type=str, default='tcarr23@charlotte.edu')
    p_pipe.add_argument('--pretrain', action='store_true', help='Include pretraining step')
    p_pipe.add_argument('--pretrain-time', dest='pretrain_time', type=str, default='12:00:00')
    p_pipe.add_argument('--pretrain-extra', nargs='*', default=[], help='Extra args for pretrain step')
    p_pipe.add_argument('--train-time', dest='train_time', type=str, default='240:00:00')
    p_pipe.add_argument('--train-extra', nargs='*', default=[], help='Extra args for train step')
    p_pipe.add_argument('--eval-time', dest='eval_time', type=str, default='12:00:00')

    sub.add_parser('dash', help='Regenerate dashboard')
    sub.add_parser('clean-results', help='Remove old evaluation outputs')
    sub.add_parser('validate', help='Quick environment checks')
    sub.add_parser('migrate-models', help='Move trained_models and output to data/models*')
    sub.add_parser('refactor-structure', help='Move legacy dirs into target structure')
    sub.add_parser('state', help='Print current state: models present, experiments completed/missing')

    args = p.parse_args(argv)
    return p, args


def main(argv=None):
    p, args = parse_args(argv)
    if not args.cmd:
        return interactive_menu()
    if args.cmd == 'eval':
        return cmd_eval(args)
    if args.cmd == 'dash':
        return cmd_dash(args)
    if args.cmd == 'clean-results':
        return cmd_clean_results(args)
    if args.cmd == 'validate':
        return cmd_validate(args)
    if args.cmd == 'train':
        return cmd_train(args)
    if args.cmd == 'preprocess':
        return cmd_preprocess(args)
    if args.cmd == 'smoke':
        return cmd_smoke(args)
    if args.cmd == 'state':
        return cmd_state(args)
    if args.cmd == 'quick':
        return cmd_quick(args)
    if args.cmd == 'pipeline':
        return cmd_pipeline(args)
    if args.cmd == 'migrate-models':
        return cmd_migrate_models(args)
    if args.cmd == 'refactor-structure':
        return cmd_refactor_structure(args)
    p.print_help(); return 0


if __name__ == '__main__':
    raise SystemExit(main())

