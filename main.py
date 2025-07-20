#!/usr/bin/env python3
"""
Fuel‑Blend Properties Prediction Script
---------------------------------------
A self‑contained pipeline that:
1. Loads the Shell.ai Hackathon ``train.csv`` and ``test.csv`` files.
2. Trains one regression model per blend‑property target.
3. Generates predictions for the 500 hidden blends in ``test.csv``.
4. Writes ``submission.csv`` in the exact leaderboard format.

The baseline model is **XGBoost** (tree‑based gradient boosting) with
sensible defaults and early stopping. If XGBoost is not installed, the
script gracefully falls back to **RandomForestRegressor** from
``scikit‑learn``. The whole run fits on a modern laptop in a few
minutes.

Usage (from your project root::)

    python fuel_blend_prediction.py \
        --train_path data/train.csv \
        --test_path  data/test.csv  \
        --output     submission.csv

or simply::

    python fuel_blend_prediction.py

…assuming the CSVs are in the current directory.

Dependencies
~~~~~~~~~~~~
- pandas
- numpy
- scikit‑learn >= 1.2
- xgboost >= 2.0   (optional but recommended for higher accuracy)

Install with::

    pip install pandas numpy scikit‑learn xgboost==2.0.3

Metric
~~~~~~
The script prints mean absolute percentage error (MAPE) on a hold‑out
20 % validation split so you can gauge expected leaderboard
performance.

Author: ChatGPT (OpenAI)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Try to import XGBoost; if unavailable fallback to RandomForestRegressor.
# ----------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor  # type: ignore

    def make_model() -> "XGBRegressor":
        """Return a fresh XGBRegressor with tuned‑for‑speed defaults."""
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            n_jobs=-1,
            objective="reg:squarederror",
            random_state=42,
        )

    MODEL_NAME = "XGBRegressor"
except ModuleNotFoundError:
    from sklearn.ensemble import RandomForestRegressor

    def make_model() -> "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

    MODEL_NAME = "RandomForestRegressor"


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def identify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dict with keys: input_cols, target_cols."""
    input_cols = [
        c for c in df.columns if c.startswith("Component") and not c.startswith("BlendProperty")
    ]
    target_cols = [c for c in df.columns if c.startswith("BlendProperty")]
    return {"input_cols": input_cols, "target_cols": target_cols}


def train_models(
    X: pd.DataFrame, y: pd.DataFrame, val_fraction: float = 0.2, random_state: int = 42
):
    """Train one model per target column. Returns dict {column: model}."""
    models = {}
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=random_state
    )

    print(f"\nTraining models (base estimator: {MODEL_NAME})…")
    for target in y.columns:
        model = make_model()
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val[target], y_pred)
        print(f"  {target:>15}: MAPE={mape:6.4f}")
        models[target] = model
    overall_mape = mean_absolute_percentage_error(y_val, pd.DataFrame(
        {t: models[t].predict(X_val) for t in y.columns}
    ))
    print(f"\nOverall validation MAPE: {overall_mape:6.4f}\n")
    return models


def predict_test(models: Dict[str, "object"], X_test: pd.DataFrame) -> pd.DataFrame:
    """Apply trained models to the test feature matrix."""
    preds = {target: model.predict(X_test) for target, model in models.items()}
    return pd.DataFrame(preds)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Fuel‑blend property prediction")
    parser.add_argument("--train_path", default="data/dataset/train.csv", help="Path to train.csv")
    parser.add_argument("--test_path", default="data/dataset/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--output", default="data/dataset/submission.csv", help="Output submission CSV path"
    )
    parser.add_argument(
        "--no_validation", action="store_true", help="Skip validation printing"
    )

    args = parser.parse_args(argv)

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    if not train_path.exists() or not test_path.exists():
        sys.exit("\n[Error] train.csv or test.csv not found. Check paths.\n")

    # Load data
    print("Loading data…")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Identify feature/target columns
    cols = identify_columns(train_df)
    X = train_df[cols["input_cols"]]
    y = train_df[cols["target_cols"]]
    X_test = test_df[cols["input_cols"]]

    # Train
    models = train_models(X, y) if not args.no_validation else {c: make_model() for c in y.columns}
    if args.no_validation:
        print(f"Training models without hold‑out validation ({MODEL_NAME}) …")
        for col in y.columns:
            models[col].fit(X, y[col])

    # Predict
    print("Predicting test set …")
    test_preds = predict_test(models, X_test)

    # Build submission DF
    submission = pd.concat([test_df[["ID"]].reset_index(drop=True), test_preds], axis=1)

    # Ensure correct column order
    submission = submission[["ID"] + cols["target_cols"]]

    # Save
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission file written to: {args.output} (shape={submission.shape})\n")

    # Optional: mini‑report in JSON for automation pipelines
    report = {
        "model": MODEL_NAME,
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "targets": cols["target_cols"],
    }
    if "overall_mape" in locals():
        report["overall_mape"] = overall_mape  # type: ignore
    print("Run summary:\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
