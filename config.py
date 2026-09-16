"""
Configuration for the L2 proficiency Monte Carlo / counterfactual analysis.

All paths are relative to this file, so the project runs from any working
directory. Nothing here has side effects at import time.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "FINAL_preprocessed_data.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Modelling parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BOOT = 150          # bootstrap refits per target
N_ESTIMATORS = 250    # trees per random forest
PI_REPEATS = 5        # permutation-importance repeats
TOPK_IMPORTANCE = 15

TARGETS = ["Grammar", "Vocabulary", "Pronunciation_TOTAL"]

# ---------------------------------------------------------------------------
# Columns that must never be predictors
# ---------------------------------------------------------------------------
# Row identifiers.
ID_COLUMNS = ["Unnamed: 0", "P_ID", "ID"]

# Outcome measures. Pronunciation_TOTAL is built from these three sub-scores
# (they correlate with it at r = 0.98), so feeding them in as predictors lets
# the model read the answer off the input. They are excluded for every target,
# not just pronunciation, because they are outcomes rather than learner
# characteristics.
OUTCOME_COLUMNS = [
    "Pronunciation_discrimination",
    "Pronunciation_stress",
    "Pronunciation_comprehension",
]

# Attention-check flags are data-quality markers, not learner characteristics.
QUALITY_FLAG_COLUMNS = ["Attn_check_BLP", "Attn_check_input"]

EXCLUDED_PREDICTORS = ID_COLUMNS + OUTCOME_COLUMNS + QUALITY_FLAG_COLUMNS

# ---------------------------------------------------------------------------
# Counterfactual analysis
# ---------------------------------------------------------------------------
# Only features a learner could change through behaviour or practice. Age,
# non-verbal intelligence and working memory are fixed traits; Proficiency_L2
# is a self-rating of the outcome itself, so "raise your self-rated
# proficiency" would be circular advice. Search bounds are the observed range
# of each feature in the data, computed at run time.
EDITABLE_FEATURES = [
    "L2_input",
    "Usage_L2",
    "L2_attitude",
    "Phonological_Awareness",
]

CF_STEPS = 2000       # proposals per search start
CF_PATIENCE = 400     # stop a start after this many proposals without improvement
CF_MULTISTART = 5     # baseline plus the best single-feature sweeps
CF_STEP_SIZE = 0.10   # fraction of a feature's range moved per proposal

# Settings for `run_analysis.py --quick`: a smoke test, not a result.
QUICK = {"N_BOOT": 20, "N_ESTIMATORS": 50, "CF_STEPS": 300, "CF_PATIENCE": 100}
