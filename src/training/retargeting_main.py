# Thin wrapper to host the main training program under src
from pathlib import Path
import sys

# Ensure project root on sys.path for existing imports like model.*, data
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the original main entry
from main import main as _root_main

def main():
    return _root_main()

