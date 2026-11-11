
"""
ML Zoomcamp 2026 - Homework 6 (Decision Trees and Ensemble Learning)

Reproduces every graded answer for Homework 6 against the pinned 2026
car fuel-efficiency release.

Run from the repository root:
    uv run --project hw6 python hw6_solve.py

Dataset preparation follows the homework statement exactly:
  * fill missing values with zeros
  * train_test_split(df, test_size=0.2, random_state=1) then
    train_test_split(df_full_train, test_size=0.25, random_state=1)
  * DictVectorizer(sparse=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hw6" / "car_fuel_efficiency_2026.csv"

TARGET = "fuel_efficiency_mpg"
SEED = 1


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def load_data():
    df = pd.read_csv(DATA)

    # Fill missing values with zeros, then split with the exact calls from
    # the homework statement.
    df = df.fillna(0)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=SEED
    )

    y_train = df_train.pop(TARGET)
    y_val = df_val.pop(TARGET)

    dv = DictVectorizer(sparse=True)
    train_matrix = dv.fit_transform(df_train.to_dict(orient="records"))
    val_matrix = dv.transform(df_val.to_dict(orient="records"))
    features = df_train.columns
    return dv, train_matrix, val_matrix, y_train, y_val, features, df


def main() -> None:
    print("=" * 62)
    print("ML ZOOMCAMP 2026 - HOMEWORK 6 (Decision Trees and Ensembles)")
    print("=" * 62)

    dv, X_train, X_val, y_train, y_val, features, target = load_data()
    print(f"\ntrain rows: {len(y_train)}  val rows: {len(y_val)}")
    print(f"features ({len(features)}): {list(features)}")

    # ---------- Q1: decision tree max_depth=1 splitting feature ----------
    print("\n--- Q1: Original splitting feature (DecisionTree max_depth=1) ---")
    tree = DecisionTreeRegressor(max_depth=1, random_state=SEED)
    tree.fit(X_train, y_train)
    split_feature = dv.feature_names_[tree.tree_.feature[0]]
    original_feature = split_feature.split("=", 1)[0]
    print(f"vectorized split feature: {split_feature}")
    print(f"Q1 Answer: {original_feature}")

    # ---------- Q2 & Q3: n_estimators sweep ----------
    print("\n--- Q2 / Q3: n_estimators sweep (random_state=1) ---")
    forest_scores: dict[int, float] = {}
    for n_estimators in [10, 50, 100, 150]:
        forest = RandomForestRegressor(
            n_estimators=n_estimators, random_state=SEED, n_jobs=-1
        )
        forest.fit(X_train, y_train)
        score = rmse(y_val, forest.predict(X_val))
        forest_scores[n_estimators] = score
        print(f"  n_estimators={n_estimators:<4} -> val RMSE = {score:.6f}")

    q2 = forest_scores[10]
    best_n_estimators = min(forest_scores, key=forest_scores.get)
    # The graded options are decade-scaled; pick the closest one.
    q2_options = [0.1837, 1.837, 18.37, 183.7]
    q2_option = min(q2_options, key=lambda opt: abs(opt - q2))
    print(f"Q2 Answer (n_estimators=10): {q2:.4f} -> closest option: {q2_option}")
    print(f"Q3 Answer (best n_estimators): {best_n_estimators}")

    # ---------- Q4: best max_depth (mean RMSE over n_estimators) ----------
    print("\n--- Q4: max_depth sweep (mean RMSE over n_estimators) ---")
    depth_scores: dict[int, float] = {}
    for depth in [10, 15, 20, 25]:
        scores = []
        for n_estimators in [10, 50, 100, 150]:
            forest = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=SEED,
                n_jobs=-1,
            )
            forest.fit(X_train, y_train)
            scores.append(rmse(y_val, forest.predict(X_val)))
        depth_scores[depth] = float(np.mean(scores))
        print(f"  max_depth={depth:<3} -> mean RMSE = {depth_scores[depth]:.6f}")

    best_max_depth = min(depth_scores, key=depth_scores.get)
    print(f"Q4 Answer (best max_depth): {best_max_depth}")

    # ---------- Q5: most important feature ----------
    print("\n--- Q5: Most important feature (n_estimators=10, max_depth=20) ---")
    importance_forest = RandomForestRegressor(
        n_estimators=10, max_depth=20, random_state=SEED, n_jobs=-1
    )
    importance_forest.fit(X_train, y_train)
    importances = dict(zip(dv.feature_names_, importance_forest.feature_importances_))

    candidates = [
        "vehicle_weight",
        "horsepower",
        "acceleration",
        "engine_displacement",
    ]
    for name in candidates:
        print(f"  {name:<22} = {importances.get(name, 0.0):.6f}")

    best_feature = max(candidates, key=lambda name: importances.get(name, 0.0))
    print(f"Q5 Answer: {best_feature}")

    # ---------- Q6: XGBoost eta comparison ----------
    print("\n--- Q6: XGBoost eta (0.3 vs 0.1) ---")
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "val")]

    eta_scores: dict[float, float] = {}
    for eta in [0.3, 0.1]:
        params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": SEED,
            "verbosity": 1,
        }
        bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
        # XGBoost 3.x returns a plain string like "[0]  val-rmse:1.234567"
        raw = bst.eval(dval)
        if isinstance(raw, dict):
            score = float(raw["rmse"])
        else:
            score = float(str(raw).rsplit(":", 1)[-1].strip(" ]"))
        eta_scores[eta] = score
        print(f"  eta={eta} -> val RMSE = {score:.6f}")

    best_eta = min(eta_scores, key=eta_scores.get)
    if eta_scores[0.3] == eta_scores[0.1]:
        q6_answer = "Both give equal value"
    else:
        q6_answer = f"{best_eta}"
    print(f"Q6 Answer: {q6_answer}")

    print("\n" + "=" * 62)
    print("SUMMARY OF ANSWERS")
    print("=" * 62)
    print(f"Q1. Splitting feature        : {original_feature}")
    print(f"Q2. RF val RMSE (n=10)       : {q2_option}  (computed {q2:.4f})")
    print(f"Q3. Best n_estimators        : {best_n_estimators}")
    print(f"Q4. Best max_depth           : {best_max_depth}")
    print(f"Q5. Most important feature   : {best_feature}")
    print(f"Q6. Best XGBoost eta         : {q6_answer}")


if __name__ == "__main__":
    main()
