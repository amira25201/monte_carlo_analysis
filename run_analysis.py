"""
Run the whole analysis top to bottom and write figures/, results/ and models/.

    python run_analysis.py            # full run, about 5 minutes
    python run_analysis.py --quick    # smoke test with small settings, under a minute
    python run_analysis.py --row 12   # start the counterfactuals from data row 12
                                      # instead of the population-typical learner

The data file is not in the repository; see data/README.md.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import config
import mc_analysis as mc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=config.DATA_PATH, help="path to the CSV (default: data/FINAL_preprocessed_data.csv)")
    parser.add_argument("--quick", action="store_true", help="small settings for a smoke test")
    parser.add_argument("--row", type=int, default=None, help="start counterfactuals from this data row instead of the population baseline")
    args = parser.parse_args()

    n_boot, n_estimators, search = config.N_BOOT, config.N_ESTIMATORS, mc.SearchSettings()
    if args.quick:
        q = config.QUICK
        n_boot, n_estimators = q["N_BOOT"], q["N_ESTIMATORS"]
        search = mc.SearchSettings(steps=q["CF_STEPS"], patience=q["CF_PATIENCE"])
        print("QUICK MODE: reduced settings, results are a smoke test only")

    t0 = time.time()
    df = mc.load_data(args.data)
    X_full, numeric, categorical = mc.split_predictors(df)
    print(f"{len(df)} rows; {len(numeric)} numeric and {len(categorical)} text predictors; excluded: {config.EXCLUDED_PREDICTORS}")

    # 1. One pipeline per outcome
    fitted = mc.train_target_pipelines(df, X_full, mc.build_preprocessor(numeric, categorical), config.MODELS_DIR, n_estimators)
    for ft in fitted.values():
        print(f"[{ft.target}] test RMSE={ft.rmse:.4f}  R2={ft.r2:.3f}")

    # 2. Bootstrap uncertainty and permutation importance
    for ft in fitted.values():
        print(f"bootstrap: {ft.target}")
        mc.run_monte_carlo(ft, config.FIGURES_DIR, n_boot)
    mc.plot_metric_comparison(fitted, config.FIGURES_DIR)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = mc.mc_summary(fitted)
    summary.to_csv(config.RESULTS_DIR / "model_summary.csv", index=False)
    for ft in fitted.values():
        ft.importances.rename("importance").to_csv(config.RESULTS_DIR / f"feature_importance_{ft.target}.csv")

    # 3. Goals: best out-of-sample prediction per outcome
    ceilings = mc.compute_ceilings(fitted)
    ceilings.to_csv(config.RESULTS_DIR / "ceilings.csv", index=False)
    print(ceilings.to_string(index=False))

    # 4. Counterfactuals from a population-typical learner (or one chosen row)
    ref = fitted[config.TARGETS[0]]
    X_all = pd.concat([ref.X_train, ref.X_test]).sort_index()
    if args.row is None:
        x0, baseline_label = mc.population_baseline(X_all), "the population-typical learner (median/mode profile)"
    else:
        x0, baseline_label = X_all.loc[[args.row]].copy(), f"data row {args.row}"
    editable = [f for f in config.EDITABLE_FEATURES if f in X_all.columns]
    bounds = mc.observed_bounds(X_all, editable)
    for ft in fitted.values():  # single-row predictions are faster without a thread pool
        ft.pipe.named_steps["model"].set_params(n_jobs=1)

    cf = {}
    goal = dict(zip(ceilings.target, ceilings.ceiling))
    for ft in fitted.values():
        print(f"counterfactual: {ft.target}")
        cf[ft.target] = mc.run_counterfactual(ft.target, ft.pipe.predict, x0, goal[ft.target], editable, bounds, config.RESULTS_DIR, search)
    print("counterfactual: TOTAL_L2")
    models = [ft.pipe for ft in fitted.values()]
    cf["TOTAL_L2"] = mc.run_counterfactual("TOTAL_L2", lambda X: mc.predict_total(models, X), x0, goal["TOTAL_L2"], editable, bounds, config.RESULTS_DIR, search)

    # 5. Findings in words
    findings = mc.write_findings(summary, fitted, ceilings, cf, baseline_label, config.RESULTS_DIR / "findings.md")
    print("\n" + findings)
    print(f"Done in {time.time() - t0:.0f}s. Figures in {config.FIGURES_DIR.name}/, tables in {config.RESULTS_DIR.name}/, models in {config.MODELS_DIR.name}/ (models are git-ignored).")


if __name__ == "__main__":
    main()
