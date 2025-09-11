# Compatibility wrapper; keep canonical eval implementation at root for now
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_model import *  # re-export

