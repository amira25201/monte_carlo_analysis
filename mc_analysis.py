"""
Monte Carlo (bootstrap) and counterfactual analysis of second-language
proficiency: what predicts Grammar, Vocabulary and Pronunciation scores, how
stable those predictions are, and which learner characteristics would need to
change for a typical learner to approach the best observed outcome.

Method in one paragraph. One random forest is fitted per outcome on an 80/20
split. Uncertainty comes from refitting each forest on bootstrap resamples of
the training set and predicting the held-out test set every time. Predictor
importance is permutation importance on the test set. The counterfactual
search starts from a population-typical learner (median of every numeric
feature, mode of every text feature), sets the goal to the best out-of-sample
prediction, and runs a bounded random local search over the editable
features only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config

try:
    from tqdm import tqdm
except ImportError:  # progress bar only

    def tqdm(iterable, **_):
        return iterable


plt.rcParams.update(
    {
        "figure.figsize": (9, 4.8),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    }
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV and strip stray whitespace from text values."""
    df = pd.read_csv(path or config.DATA_PATH)
    for col in df.select_dtypes(exclude=["number", "bool"]).columns:
        df[col] = df[col].str.strip()
    return df


def split_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Predictor matrix with targets, identifiers, outcome measures and quality flags removed."""
    drop = [c for c in config.TARGETS + config.EXCLUDED_PREDICTORS if c in df.columns]
    X = df.drop(columns=drop).copy()
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return X, numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def model_kwargs(n_estimators: int | None = None) -> dict:
    return {
        "n_estimators": n_estimators or config.N_ESTIMATORS,
        "max_features": "sqrt",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
    }


def make_pipeline(preprocessor: ColumnTransformer, n_estimators: int | None = None) -> Pipeline:
    """A fresh pipeline with its own copy of the preprocessor, never a shared fitted one."""
    return Pipeline([("prep", clone(preprocessor)), ("model", RandomForestRegressor(**model_kwargs(n_estimators)))])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class FittedTarget:
    target: str
    pipe: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    rmse: float
    r2: float
    mc_preds: np.ndarray | None = None
    importances: pd.Series | None = None


def train_target_pipelines(
    df: pd.DataFrame,
    X_full: pd.DataFrame,
    preprocessor: ColumnTransformer,
    models_dir: Path | None = None,
    n_estimators: int | None = None,
) -> dict[str, FittedTarget]:
    fitted = {}
    for target in config.TARGETS:
        y_all = pd.to_numeric(df[target], errors="coerce")
        mask = y_all.notna()
        X = X_full.loc[mask].reset_index(drop=True)
        y = y_all.loc[mask].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        pipe = make_pipeline(preprocessor, n_estimators).fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        fitted[target] = FittedTarget(
            target, pipe, X_tr, X_te, y_tr, y_te, root_mean_squared_error(y_te, y_hat), r2_score(y_te, y_hat)
        )
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, models_dir / f"pipeline_{target}.pkl")
    return fitted


# ---------------------------------------------------------------------------
# Monte Carlo (bootstrap refits)
# ---------------------------------------------------------------------------
def bootstrap_predictions(
    model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """Refit `model` on `n_boot` bootstrap resamples; return (n_boot, n_eval) predictions."""
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    preds = np.empty((n_boot, X_eval.shape[0]), dtype=np.float32)
    for b in tqdm(range(n_boot), desc="bootstrap refits", leave=False):
        idx = rng.randint(0, n, size=n)
        m = clone(model).fit(X_train[idx], y_train[idx])
        preds[b] = m.predict(X_eval)
    return preds


def run_monte_carlo(ft: FittedTarget, figures_dir: Path, n_boot: int | None = None) -> FittedTarget:
    """Bootstrap uncertainty and permutation importance for one already-fitted target."""
    prep = ft.pipe.named_steps["prep"]
    X_tr = np.asarray(prep.transform(ft.X_train), dtype=np.float32)
    X_te = np.asarray(prep.transform(ft.X_test), dtype=np.float32)
    ft.mc_preds = bootstrap_predictions(
        ft.pipe.named_steps["model"], X_tr, ft.y_train, X_te, n_boot or config.N_BOOT, config.RANDOM_STATE
    )

    imp = permutation_importance(
        ft.pipe, ft.X_test, ft.y_test, n_repeats=config.PI_REPEATS, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    ft.importances = pd.Series(imp.importances_mean, index=ft.X_test.columns).sort_values(ascending=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_prediction_intervals(ft, figures_dir)
    plot_importance(ft, figures_dir)
    return ft


def plot_prediction_intervals(ft: FittedTarget, figures_dir: Path) -> None:
    lo, hi = np.percentile(ft.mc_preds, [2.5, 97.5], axis=0)
    mean = ft.mc_preds.mean(axis=0)
    order = np.argsort(ft.y_test)
    plt.figure(figsize=(8.5, 5))
    plt.errorbar(
        ft.y_test[order], mean[order], yerr=[mean[order] - lo[order], hi[order] - mean[order]],
        fmt="o", ms=4, alpha=0.8, ecolor="grey", capsize=2, label="bootstrap mean and 95% range",
    )
    lim = [min(ft.y_test.min(), lo.min()), max(ft.y_test.max(), hi.max())]
    plt.plot(lim, lim, "--", color="black", lw=1, label="perfect prediction")
    plt.xlabel(f"Actual {ft.target}")
    plt.ylabel(f"Predicted {ft.target}")
    plt.title(f"{ft.target}: held-out predictions with bootstrap uncertainty (n={len(ft.y_test)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"prediction_intervals_{ft.target}.png", dpi=200)
    plt.close()


def plot_importance(ft: FittedTarget, figures_dir: Path) -> None:
    top = ft.importances.head(config.TOPK_IMPORTANCE).iloc[::-1]
    plt.figure(figsize=(8, 5.5))
    plt.barh(top.index, top.values)
    plt.xlabel("Permutation importance (mean drop in R² on the test set)")
    plt.title(f"{ft.target}: top {len(top)} predictors")
    plt.tight_layout()
    plt.savefig(figures_dir / f"feature_importance_{ft.target}.png", dpi=200)
    plt.close()


def mc_summary(fitted: dict[str, FittedTarget]) -> pd.DataFrame:
    """Point-estimate and bootstrap-distribution metrics per target."""
    rows = []
    for ft in fitted.values():
        rmses = [root_mean_squared_error(ft.y_test, p) for p in ft.mc_preds]
        r2s = [r2_score(ft.y_test, p) for p in ft.mc_preds]
        rows.append(
            {
                "target": ft.target,
                "n_test": len(ft.y_test),
                "rmse": ft.rmse,
                "r2": ft.r2,
                "bootstrap_rmse_median": float(np.median(rmses)),
                "bootstrap_rmse_2.5": float(np.percentile(rmses, 2.5)),
                "bootstrap_rmse_97.5": float(np.percentile(rmses, 97.5)),
                "bootstrap_r2_median": float(np.median(r2s)),
                "bootstrap_r2_2.5": float(np.percentile(r2s, 2.5)),
                "bootstrap_r2_97.5": float(np.percentile(r2s, 97.5)),
                "n_boot": ft.mc_preds.shape[0],
            }
        )
    return pd.DataFrame(rows)


def plot_metric_comparison(fitted: dict[str, FittedTarget], figures_dir: Path) -> pd.DataFrame:
    rows = []
    for ft in fitted.values():
        for b, p in enumerate(ft.mc_preds):
            rows.append({"target": ft.target, "run": b, "RMSE": root_mean_squared_error(ft.y_test, p), "R2": r2_score(ft.y_test, p)})
    perf = pd.DataFrame(rows)
    points = pd.DataFrame([{"target": ft.target, "RMSE": ft.rmse, "R2": ft.r2} for ft in fitted.values()])
    for metric in ("RMSE", "R2"):
        plt.figure(figsize=(7.6, 4.4))
        sns.boxplot(data=perf, x="target", y=metric, boxprops={"alpha": 0.6})
        sns.scatterplot(data=points, x="target", y=metric, color="red", s=120, marker="D", label="model fitted on full training set", zorder=5)
        plt.title(f"{metric} across bootstrap refits")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(figures_dir / f"bootstrap_{metric.lower()}_comparison.png", dpi=200)
        plt.close()
    return perf


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------
def predict_total(models: list[Pipeline], X: pd.DataFrame) -> np.ndarray:
    """TOTAL_L2 = mean of the domain models' predictions, one predict call per model."""
    return np.column_stack([m.predict(X) for m in models]).mean(axis=1)


def compute_ceilings(fitted: dict[str, FittedTarget]) -> pd.DataFrame:
    """
    Best out-of-sample prediction per target. Test-set predictions only, so a
    forest's near-perfect fit on its own training rows does not inflate the goal.
    TOTAL_L2 is the best averaged prediction for any single test learner, which
    is always reachable by construction; the mean of the three separate maxima
    might belong to no real learner at all.
    """
    rows = []
    for ft in fitted.values():
        preds = ft.pipe.predict(ft.X_test)
        rows.append({"target": ft.target, "ceiling": float(preds.max()), "test_row": int(ft.X_test.index[preds.argmax()])})
    ref = next(iter(fitted.values()))
    if all(ft.X_test.index.equals(ref.X_test.index) for ft in fitted.values()):
        total = predict_total([ft.pipe for ft in fitted.values()], ref.X_test)
        rows.append({"target": "TOTAL_L2", "ceiling": float(total.max()), "test_row": int(ref.X_test.index[total.argmax()])})
    else:
        rows.append({"target": "TOTAL_L2", "ceiling": float(np.mean([r["ceiling"] for r in rows])), "test_row": -1})
    return pd.DataFrame(rows)


def population_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """A typical learner: median of each numeric column, mode of each text column."""
    row = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            row[c] = float(X[c].median())
        else:
            modes = X[c].dropna().mode()
            row[c] = modes.iloc[0] if len(modes) else np.nan
    return pd.DataFrame([row], columns=X.columns)


def observed_bounds(X: pd.DataFrame, features: list[str]) -> dict[str, tuple[float, float]]:
    return {f: (float(X[f].min()), float(X[f].max())) for f in features if f in X.columns}


def sweep_feature(predict, x0: pd.DataFrame, feature: str, goal: float, bounds: dict, n_points: int = 201) -> dict:
    """Move one feature across its observed range, all else fixed."""
    lo, hi = bounds[feature]
    grid = np.linspace(lo, hi, n_points)
    block = pd.concat([x0] * n_points, ignore_index=True)
    block[feature] = grid
    preds = np.asarray(predict(block), dtype=float)
    y0 = float(predict(x0)[0])
    best = int(preds.argmax())
    reached = np.flatnonzero(preds >= goal)
    return {
        "feature": feature,
        "current_value": float(x0[feature].iloc[0]),
        "current_prediction": y0,
        "best_value": float(grid[best]),
        "best_prediction": float(preds[best]),
        "gain": float(preds[best] - y0),
        "goal_reachable_alone": bool(reached.size),
        "value_needed_for_goal": float(grid[reached[0]]) if reached.size else np.nan,
        "range_low": lo,
        "range_high": hi,
    }


@dataclass
class SearchSettings:
    steps: int = field(default_factory=lambda: config.CF_STEPS)
    patience: int = field(default_factory=lambda: config.CF_PATIENCE)
    multistart: int = field(default_factory=lambda: config.CF_MULTISTART)
    step_size: float = field(default_factory=lambda: config.CF_STEP_SIZE)
    seed: int = field(default_factory=lambda: config.RANDOM_STATE)


def joint_search(predict, x0: pd.DataFrame, goal: float, editable: list[str], bounds: dict, settings: SearchSettings | None = None) -> tuple[float, pd.DataFrame]:
    """
    Random local search over the editable features, clipped to observed bounds.
    Proposals move a random third of the features a small step towards the
    goal; a proposal is kept if it closes the gap. Restarts from the baseline
    and from the best point of each single-feature sweep.
    """
    s = settings or SearchSettings()
    rng = np.random.RandomState(s.seed)
    k = max(1, int(np.ceil(len(editable) * 0.35)))

    def run_from(x_start: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        x_best = x_start.copy()
        y_best = float(predict(x_best)[0])
        idle = 0
        for _ in range(s.steps):
            if abs(goal - y_best) <= 1e-3:
                break
            x_try = x_best.copy()
            for c in rng.choice(editable, size=k, replace=False):
                lo, hi = bounds[c]
                direction = 1.0 if y_best < goal else -1.0
                proposal = float(x_try[c].iloc[0]) + direction * s.step_size * (hi - lo) * (0.8 + 0.4 * rng.rand())
                x_try[c] = float(np.clip(proposal, lo, hi))
            y_try = float(predict(x_try)[0])
            if abs(y_try - goal) < abs(y_best - goal):
                x_best, y_best, idle = x_try, y_try, 0
            else:
                idle += 1
                if idle > s.patience:
                    break
        return y_best, x_best

    starts = [x0.copy()]
    for c in editable:
        info = sweep_feature(predict, x0, c, goal, bounds, n_points=101)
        xs = x0.copy()
        xs[c] = info["best_value"]
        starts.append(xs)

    best_y, best_x = float(predict(x0)[0]), x0.copy()
    for xs in starts[: s.multistart]:
        y, x = run_from(xs)
        if abs(y - goal) < abs(best_y - goal):
            best_y, best_x = y, x
    return best_y, best_x


def changes_vs_baseline(x0: pd.DataFrame, x_best: pd.DataFrame, editable: list[str]) -> pd.DataFrame:
    rows = [
        {"feature": c, "baseline": float(x0[c].iloc[0]), "needed": float(x_best[c].iloc[0]), "change": float(x_best[c].iloc[0] - x0[c].iloc[0])}
        for c in editable
        if abs(float(x_best[c].iloc[0]) - float(x0[c].iloc[0])) > 1e-9
    ]
    out = pd.DataFrame(rows, columns=["feature", "baseline", "needed", "change"])
    return out.sort_values("change", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def run_counterfactual(name: str, predict, x0: pd.DataFrame, goal: float, editable: list[str], bounds: dict, results_dir: Path, settings: SearchSettings | None = None) -> dict:
    """Single-feature sweeps and a joint search; both tables are written to results/."""
    results_dir.mkdir(parents=True, exist_ok=True)
    sweeps = pd.DataFrame([sweep_feature(predict, x0, f, goal, bounds) for f in editable]).sort_values("gain", ascending=False)
    sweeps.to_csv(results_dir / f"counterfactual_{name}_sweeps.csv", index=False)
    y_best, x_best = joint_search(predict, x0, goal, editable, bounds, settings)
    changes = changes_vs_baseline(x0, x_best, editable)
    changes.to_csv(results_dir / f"counterfactual_{name}_joint.csv", index=False)
    return {"name": name, "baseline_prediction": float(predict(x0)[0]), "goal": goal, "achieved": y_best, "sweeps": sweeps, "changes": changes}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def write_findings(summary: pd.DataFrame, fitted: dict[str, FittedTarget], ceilings: pd.DataFrame, cf: dict[str, dict], baseline_label: str, path: Path) -> str:
    lines = ["# Findings", "", "Generated by `run_analysis.py`. Every number below comes from `results/*.csv`.", ""]
    lines += ["## Model fit on the held-out test set", "", "| Outcome | n test | RMSE | R² | Bootstrap R² 95% range |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.target} | {int(r.n_test)} | {r.rmse:.3f} | {r.r2:.2f} | {r['bootstrap_r2_2.5']:.2f} to {r['bootstrap_r2_97.5']:.2f} |")
    lines += ["", "## Strongest predictors (permutation importance, test set)", ""]
    for ft in fitted.values():
        top = ft.importances.head(5)
        lines.append(f"- **{ft.target}**: " + ", ".join(f"{k} ({v:.3f})" for k, v in top.items()))
    lines += ["", f"## Counterfactuals, starting from {baseline_label}", ""]
    lines += ["| Outcome | Baseline prediction | Goal (best test prediction) | Best achieved | Largest single-feature gain |", "|---|---|---|---|---|"]
    for name, r in cf.items():
        best = r["sweeps"].iloc[0]
        lines.append(f"| {name} | {r['baseline_prediction']:.3f} | {r['goal']:.3f} | {r['achieved']:.3f} | {best.feature}: +{best.gain:.3f} |")
    lines += ["", "Feature changes proposed by the joint search:", ""]
    for name, r in cf.items():
        if r["changes"].empty:
            lines.append(f"- **{name}**: no editable feature moved the prediction.")
        else:
            lines.append(f"- **{name}**: " + "; ".join(f"{c.feature} {c.baseline:.2f} to {c.needed:.2f}" for c in r["changes"].itertuples()))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text
