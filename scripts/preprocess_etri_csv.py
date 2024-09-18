#!/usr/bin/env python3
"""
Convert ETRI skeleton CSV data to the .pkl format expected by the downstream
training pipeline (datasets.py / train_downstream_models.py).

Target .pkl format
------------------
A Python dict mapping ETRI-style filenames to numpy arrays:

    {
        "A001P001G001C001": np.ndarray(T, 75),  # float32
        "A001P002G001C002": np.ndarray(T, 75),
        ...
    }

Each value is a 2-D array of shape (T, 75) where T is the number of frames
(padded/truncated to --max_frames, default 64) and 75 = 25 joints x 3 coords
(x, y, z concatenated in joint order).

ETRI filename convention
------------------------
    A{action:03d}P{person:03d}G{group:03d}C{camera:03d}

Actions 1..55, Persons 1..100, Groups 1..2, Cameras 1..8.

This matches the parse_file_name() function in src/data/datasets.py:
    A = int(filename[1:4])
    P = int(filename[5:8])
    G = int(filename[9:12])
    C = int(filename[13:16])

Supported CSV formats
---------------------
The primary format is the Kinect v2 CSV from the ETRI dataset:

1. **Kinect v2 format (253 columns)** -- the main supported format.
   Header row with columns: frameNum, bodyindexID, trackingID,
   then per joint (25 joints, 10 cols each):
       joint{N}_3dX, joint{N}_3dY, joint{N}_3dZ,
       joint{N}_depthX, joint{N}_depthY,
       joint{N}_orientationX/Y/Z/W, joint{N}_trackingState
   Only the 3D cartesian columns (joint{N}_3dX/Y/Z) are extracted.

2. **Simple 75-column format** -- 75 numeric columns per row, already
   containing just x,y,z for 25 joints. No extraction needed.

3. **Metadata embedded in filename** -- the CSV filename follows the ETRI
   convention (e.g. A001_P001_G001_C001.csv or A001P001G001C001.csv).
   Underscores between components are optional.

4. **Single "mega-CSV"** with all samples concatenated, using a column like
   "filename" or "sample_id" to delimit samples. Use --group_col to specify
   the column name that identifies each sample.

Usage examples
--------------
    # Directory of per-sample CSVs named like A001P001G001C001.csv
    python scripts/preprocess_etri_csv.py \\
        --input_dir data/etri/csv/ \\
        --output_path data/etri/etri.pkl

    # Single CSV file with known metadata
    python scripts/preprocess_etri_csv.py \\
        --input_dir data/etri/single_sample.csv \\
        --output_path data/etri/etri.pkl \\
        --action 1 --person 1 --group 1 --camera 1

    # Preview mode (prints first rows, does not save)
    python scripts/preprocess_etri_csv.py \\
        --input_dir data/etri/csv/ \\
        --preview

    # Mega-CSV with a grouping column
    python scripts/preprocess_etri_csv.py \\
        --input_dir data/etri/all_skeletons.csv \\
        --output_path data/etri/etri.pkl \\
        --group_col filename

This script is lightweight (pandas + numpy only, no GPU, no model loading)
and is safe to run on the HPC login node.
"""

import argparse
import os
import re
import sys
import pickle
from pathlib import Path

import numpy as np

# Optional pandas -- fall back to pure numpy/csv if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# ETRI filename helpers
# ---------------------------------------------------------------------------

ETRI_FILENAME_RE = re.compile(
    r"A(\d{3})_?P(\d{3})_?G(\d{3})_?C(\d{3})"
)


def parse_etri_filename(name: str):
    """
    Try to extract (action, person, group, camera) from an ETRI-style filename.
    Returns a dict or None if the pattern does not match.
    """
    m = ETRI_FILENAME_RE.search(name)
    if m:
        return {
            "A": int(m.group(1)),
            "P": int(m.group(2)),
            "G": int(m.group(3)),
            "C": int(m.group(4)),
        }
    return None


def make_etri_key(action: int, person: int, group: int, camera: int) -> str:
    """Build the canonical ETRI filename key used as the dict key in the .pkl."""
    return f"A{action:03d}P{person:03d}G{group:03d}C{camera:03d}"


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------

def read_csv_to_joints(filepath: str, num_joints: int = 25) -> np.ndarray:
    """
    Read a single ETRI CSV file and extract 3D joint positions.

    ETRI Kinect CSV format (253 columns):
        frameNum, bodyindexID, trackingID,
        joint1_3dX, joint1_3dY, joint1_3dZ,
        joint1_depthX, joint1_depthY,
        joint1_orientationX/Y/Z/W, joint1_trackingState,
        joint2_3dX, ...   (10 cols per joint × 25 joints)

    We extract only the 3D cartesian columns (joint{N}_3dX/Y/Z) for N=1..25,
    yielding 75 values per frame in joint order.

    Returns shape (num_frames, 75) as float32.
    """
    filepath = str(filepath)

    if HAS_PANDAS:
        df = pd.read_csv(filepath)

        # Check if this is the Kinect format with named columns
        if "joint1_3dX" in df.columns:
            # Extract joint{N}_3dX, joint{N}_3dY, joint{N}_3dZ in order
            cols_3d = []
            for j in range(1, num_joints + 1):
                cols_3d.extend([f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ"])

            missing = [c for c in cols_3d if c not in df.columns]
            if missing:
                print(f"  WARNING: Missing columns: {missing[:5]}...", file=sys.stderr)

            present = [c for c in cols_3d if c in df.columns]
            arr = df[present].values.astype(np.float32)

            # If some joints are missing, pad to 75 columns
            if arr.shape[1] < num_joints * 3:
                padded = np.zeros((arr.shape[0], num_joints * 3), dtype=np.float32)
                padded[:, :arr.shape[1]] = arr
                arr = padded

            return arr

        # Fallback: no recognized header — try positional extraction
        # 3 metadata cols + 10 cols per joint; 3D coords are at offsets 0,1,2
        numeric_df = df.select_dtypes(include=[np.number])
        num_cols = numeric_df.shape[1]

        if num_cols >= 3 + num_joints * 10:
            # Kinect format without named 3dX/Y/Z headers
            data = numeric_df.values.astype(np.float32)
            indices = []
            for j in range(num_joints):
                base = 3 + j * 10  # skip 3 metadata cols
                indices.extend([base, base + 1, base + 2])
            return data[:, indices]

        elif num_cols >= num_joints * 3:
            # Already 75+ numeric columns — take first 75
            return numeric_df.values[:, :num_joints * 3].astype(np.float32)

        else:
            print(f"  WARNING: Only {num_cols} numeric columns, expected >= 75. "
                  f"Padding with zeros.", file=sys.stderr)
            padded = np.zeros((numeric_df.shape[0], num_joints * 3), dtype=np.float32)
            padded[:, :num_cols] = numeric_df.values.astype(np.float32)
            return padded

    else:
        # Pure numpy fallback — skip header, use positional extraction
        arr = np.loadtxt(filepath, delimiter=",", dtype=np.float32, skiprows=1)
        num_cols = arr.shape[1]
        if num_cols >= 3 + num_joints * 10:
            indices = []
            for j in range(num_joints):
                base = 3 + j * 10
                indices.extend([base, base + 1, base + 2])
            return arr[:, indices]
        elif num_cols >= num_joints * 3:
            return arr[:, :num_joints * 3]
        else:
            padded = np.zeros((arr.shape[0], num_joints * 3), dtype=np.float32)
            padded[:, :num_cols] = arr
            return padded


def pad_or_truncate(arr: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad (repeat last frame) or truncate a sequence to exactly max_frames.
    Removes all-zero frames first, matching the behavior of load_data() in
    src/data/datasets.py.

    Args:
        arr: shape (T, 75)
        max_frames: target number of frames

    Returns:
        np.ndarray of shape (max_frames, 75), dtype float32
    """
    # Remove all-zero frames
    non_zero_mask = ~np.all(arr == 0, axis=1)
    non_zero = arr[non_zero_mask]
    num_frames = len(non_zero)

    if num_frames == 0:
        # Edge case: entirely zero sequence
        return np.zeros((max_frames, arr.shape[1]), dtype=np.float32)

    if num_frames >= max_frames:
        # Truncate to max_frames
        return non_zero[:max_frames].astype(np.float32)
    else:
        # Pad by repeating the last frame
        last_frame = non_zero[-1:]
        num_pad = max_frames - num_frames
        padded = np.vstack([non_zero] + [last_frame] * num_pad)
        return padded.astype(np.float32)


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def process_single_csv(filepath: str, max_frames: int,
                       action: int = None, person: int = None,
                       group: int = None, camera: int = None) -> dict:
    """
    Process one CSV file into {key: np.array(max_frames, 75)}.

    If action/person/group/camera are not provided, tries to extract them
    from the filename.
    """
    fname = Path(filepath).stem
    meta = parse_etri_filename(fname)

    if meta is not None:
        a = meta["A"]
        p = meta["P"]
        g = meta["G"]
        c = meta["C"]
    elif action is not None and person is not None and group is not None and camera is not None:
        a, p, g, c = action, person, group, camera
    else:
        raise ValueError(
            f"Cannot determine ETRI metadata for '{filepath}'. "
            f"Filename does not match A###P###G###C### pattern and "
            f"--action/--person/--group/--camera were not all provided."
        )

    key = make_etri_key(a, p, g, c)

    joints = read_csv_to_joints(filepath)
    processed = pad_or_truncate(joints, max_frames)

    return {key: processed}


def process_mega_csv(filepath: str, group_col: str, max_frames: int) -> dict:
    """
    Process a single large CSV that contains multiple samples, separated by
    a grouping column (e.g. 'filename', 'sample_id').

    Each unique value in group_col should be an ETRI-style filename or
    contain the A###P###G###C### pattern.
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for --group_col processing. "
                           "Install it with: pip install pandas")

    df = pd.read_csv(filepath)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in {filepath}. "
                         f"Available columns: {list(df.columns)}")

    result = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove the group column from numeric cols if it happens to be numeric
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)

    # Build 3D column list if available
    cols_3d = []
    for j in range(1, 26):
        cols_3d.extend([f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ"])
    use_named_cols = all(c in df.columns for c in cols_3d)

    for sample_id, sub_df in df.groupby(group_col):
        meta = parse_etri_filename(str(sample_id))
        if meta is None:
            print(f"  WARNING: Skipping sample '{sample_id}' -- cannot parse ETRI metadata.",
                  file=sys.stderr)
            continue

        key = make_etri_key(meta["A"], meta["P"], meta["G"], meta["C"])
        if use_named_cols:
            arr = sub_df[cols_3d].values.astype(np.float32)
        else:
            arr = sub_df[numeric_cols].values.astype(np.float32)
            if arr.shape[1] >= 253:
                indices = []
                for j in range(25):
                    base = 3 + j * 10
                    indices.extend([base, base + 1, base + 2])
                arr = arr[:, indices]
            elif arr.shape[1] > 75:
                arr = arr[:, :75]
        processed = pad_or_truncate(arr, max_frames)
        result[key] = processed

    return result


def process_directory(input_dir: str, max_frames: int) -> dict:
    """
    Process a directory of CSV files. Each CSV should be one skeleton
    sequence, named following the ETRI convention (e.g. A001P001G001C001.csv).
    """
    result = {}
    csv_files = sorted(Path(input_dir).glob("*.csv"))

    if not csv_files:
        # Try recursively
        csv_files = sorted(Path(input_dir).rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    skipped = 0
    for i, csv_path in enumerate(csv_files):
        try:
            sample = process_single_csv(str(csv_path), max_frames)
            result.update(sample)
        except ValueError as e:
            print(f"  SKIP [{i+1}/{len(csv_files)}] {csv_path.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if (i + 1) % 1000 == 0 or (i + 1) == len(csv_files):
            print(f"  Processed {i+1}/{len(csv_files)} files "
                  f"({skipped} skipped, {len(result)} samples)")

    return result


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_csv(filepath: str, num_rows: int = 5):
    """Print the first few rows of a CSV for inspection."""
    print(f"\n=== Preview: {filepath} ===")
    if HAS_PANDAS:
        # Read raw to see exactly what's in the file
        try:
            df_raw = pd.read_csv(filepath, header=None, nrows=num_rows + 1)
            print(f"Shape (raw, no header): {df_raw.shape}")
            print(f"First {num_rows + 1} rows (raw):")
            print(df_raw.to_string())
        except Exception as e:
            print(f"  Error reading raw: {e}")

        print()
        # Also try with header
        try:
            df_hdr = pd.read_csv(filepath, nrows=num_rows)
            print(f"Shape (with header): {df_hdr.shape}")
            print(f"Columns: {list(df_hdr.columns)}")
            print(f"Dtypes:\n{df_hdr.dtypes}")
            print(f"First {num_rows} rows:")
            print(df_hdr.to_string())
        except Exception as e:
            print(f"  Error reading with header: {e}")
    else:
        # Pure Python fallback
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i > num_rows:
                    break
                print(f"  Row {i}: {line.rstrip()[:200]}{'...' if len(line) > 200 else ''}")

    # Try to parse the filename
    meta = parse_etri_filename(Path(filepath).stem)
    if meta:
        print(f"\nFilename metadata: Action={meta['A']}, Person={meta['P']}, "
              f"Group={meta['G']}, Camera={meta['C']}")
    else:
        print(f"\nFilename '{Path(filepath).stem}' does not match ETRI pattern A###P###G###C###")

    # Read and show extracted 3D joint data
    try:
        joints = read_csv_to_joints(filepath)
        print(f"\nExtracted 3D joint array shape: {joints.shape}  (expected: (T, 75))")
        print(f"Value range: [{joints.min():.4f}, {joints.max():.4f}]")
        print(f"Mean: {joints.mean():.4f}, Std: {joints.std():.4f}")
        non_zero_rows = np.sum(~np.all(joints == 0, axis=1))
        print(f"Non-zero frames: {non_zero_rows}/{joints.shape[0]}")
        # Show first joint's first frame as sanity check
        if joints.shape[0] > 0:
            print(f"Frame 0, Joint 1 (x,y,z): ({joints[0,0]:.4f}, {joints[0,1]:.4f}, {joints[0,2]:.4f})")
    except Exception as e:
        print(f"\nError reading joints: {e}")


def preview_input(input_path: str, num_rows: int = 5, max_files: int = 3):
    """Preview the input (file or directory)."""
    p = Path(input_path)
    if p.is_file():
        preview_csv(str(p), num_rows)
    elif p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(p.rglob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {p}")
        for csv_path in csv_files[:max_files]:
            preview_csv(str(csv_path), num_rows)
        if len(csv_files) > max_files:
            print(f"\n... and {len(csv_files) - max_files} more files")
    else:
        print(f"Error: {input_path} is not a file or directory")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(data: dict):
    """Print summary statistics about the output data dict."""
    num_samples = len(data)
    if num_samples == 0:
        print("WARNING: Output dict is empty!")
        return

    # Gather metadata
    actions = set()
    persons = set()
    cameras = set()
    shapes = set()

    for key, arr in data.items():
        meta = parse_etri_filename(key)
        if meta:
            actions.add(meta["A"])
            persons.add(meta["P"])
            cameras.add(meta["C"])
        shapes.add(arr.shape)

    print(f"\n=== Output Validation ===")
    print(f"Total samples: {num_samples}")
    print(f"Unique actions: {len(actions)} (range {min(actions)}-{max(actions)})" if actions else "No action metadata")
    print(f"Unique persons: {len(persons)} (range {min(persons)}-{max(persons)})" if persons else "No person metadata")
    print(f"Unique cameras: {len(cameras)} (values: {sorted(cameras)})" if cameras else "No camera metadata")
    print(f"Unique array shapes: {shapes}")

    # Check consistency
    sample_key = next(iter(data))
    sample_arr = data[sample_key]
    print(f"Sample key: '{sample_key}'")
    print(f"Sample shape: {sample_arr.shape}, dtype: {sample_arr.dtype}")
    print(f"Sample value range: [{sample_arr.min():.4f}, {sample_arr.max():.4f}]")

    # Verify all shapes match
    if len(shapes) == 1:
        print("All arrays have consistent shape.")
    else:
        print(f"WARNING: Arrays have inconsistent shapes: {shapes}")

    # Cross-reference with expected ETRI config from datasets.py
    # (55 actions, 100 subjects)
    if actions:
        expected_actions = 55
        expected_persons = 100
        if len(actions) < expected_actions:
            print(f"NOTE: Only {len(actions)}/{expected_actions} actions present.")
        if len(persons) < expected_persons:
            print(f"NOTE: Only {len(persons)}/{expected_persons} persons present.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ETRI CSV skeleton data to .pkl format for the training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Path to a directory of CSV files, a single CSV file, or a mega-CSV.",
    )
    p.add_argument(
        "--output_path",
        default=None,
        help="Where to save the output .pkl file. Required unless --preview is set.",
    )
    p.add_argument(
        "--max_frames", type=int, default=64,
        help="Target sequence length T. Sequences are padded or truncated to this. Default: 64.",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Just preview the CSV structure without converting or saving.",
    )
    p.add_argument(
        "--preview_rows", type=int, default=5,
        help="Number of rows to show in preview mode. Default: 5.",
    )
    p.add_argument(
        "--group_col", default=None,
        help="For mega-CSV mode: column name that identifies each sample "
             "(e.g. 'filename', 'sample_id').",
    )

    # Metadata overrides for single-file mode
    meta = p.add_argument_group("Metadata overrides (single-file mode)")
    meta.add_argument("--action", type=int, default=None, help="Action class (1-55)")
    meta.add_argument("--person", type=int, default=None, help="Person/subject ID (1-100)")
    meta.add_argument("--group", type=int, default=None, help="Group ID (1-2)")
    meta.add_argument("--camera", type=int, default=None, help="Camera ID (1-8)")

    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_dir)

    # --- Preview mode ---
    if args.preview:
        preview_input(str(input_path), num_rows=args.preview_rows)
        return

    # --- Require output_path for non-preview ---
    if args.output_path is None:
        print("Error: --output_path is required when not using --preview.", file=sys.stderr)
        sys.exit(1)

    # --- Process ---
    if input_path.is_dir():
        print(f"Processing directory: {input_path}")
        data = process_directory(str(input_path), args.max_frames)

    elif input_path.is_file():
        if args.group_col:
            print(f"Processing mega-CSV: {input_path} (group_col='{args.group_col}')")
            data = process_mega_csv(str(input_path), args.group_col, args.max_frames)
        else:
            print(f"Processing single CSV: {input_path}")
            data = process_single_csv(
                str(input_path), args.max_frames,
                action=args.action, person=args.person,
                group=args.group, camera=args.camera,
            )
    else:
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    # --- Validate ---
    validate_output(data)

    # --- Save ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(data)} samples to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
