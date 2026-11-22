from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
SRC_DIR = PROJECT_ROOT / "src"

AERO_DIR = SRC_DIR / "aero"
ANALYSIS_DIR = SRC_DIR / "analysis"
CONSTRAINTS_DIR = SRC_DIR / "constraints"
PLOTTING_DIR = SRC_DIR / "plotting"
TOOLS_DIR = SRC_DIR / "tools"
UTILS_DIR = SRC_DIR / "utils"

INPUT_DATA = INPUT_DIR / "database.input"