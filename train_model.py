"""
train_model.py  (v2)

Changes vs the original:

1. Trains FOUR separate single-target XGBRegressor models (Inside_Temp,
   Solar_Gain, Conduction_Loss, Infiltration_Loss) instead of one
   multi-output model. Reason: each physical quantity has a DIFFERENT,
   known monotonic relationship with the inputs (e.g. Inside_Temp should
   rise with R_Value, but Conduction_Loss should fall with R_Value) --
   XGBoost's native multi-output mode shares constraint direction across
   all outputs, which can't represent that correctly. Separate models let
   each one get its own physically-correct constraint vector.

2. Adds monotone_constraints per model: this is what stops the model from
   producing physically nonsensical outputs (e.g. "more insulation makes
   it colder") even when extrapolating slightly past the training range --
   directly answers "what happens outside your training data?" in judging.

3. Holds out a test split and prints RMSE / R^2 per target. This is the
   single number that answers "how do you know your model is accurate" --
   put it on a slide.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURES = [
    "Shape_Code", "Length", "Width", "Height", "Shape_Factor",
    "R_Value", "Thermal_Mass_Factor", "Window_Area", "Window_U_Value",
    "Orientation_Factor", "ACH", "Occupants", "Elevation",
    "Outside_Temp", "Solar_Power", "Wind_Speed", "Previous_Temp",
]

# Monotonic constraint per feature, in FEATURES order.
#  +1 -> output must be non-decreasing as this feature increases
#  -1 -> output must be non-increasing as this feature increases
#   0 -> no constraint (relationship isn't monotonic / isn't known a priori)
#
# Elevation is left unconstrained in every target. Thinner air cuts the
# infiltration conductance, which warms a shelter losing heat but cools one
# gaining it from warmer outside air -- the sign genuinely depends on the
# temperature difference, so asserting one would be wrong.
MONOTONE = {
    "Inside_Temp": (
        0, 0, 0, 0, 0,        # Shape_Code, Length, Width, Height, Shape_Factor
        1, 0,                 # R_Value (+), Thermal_Mass_Factor (ambiguous over a day)
        0, -1,                # Window_Area (trade-off), Window_U_Value (-)
        1, -1, 1, 0,          # Orientation_Factor (+), ACH (-), Occupants (+), Elevation
        1, 1, -1, 1,          # Outside_Temp (+), Solar_Power (+), Wind_Speed (-), Previous_Temp (+)
    ),
    "Solar_Gain": (
        0, 0, 0, 0, 0,
        0, 0,
        1, 0,                 # Window_Area (+)
        1, 0, 0, 0,           # Orientation_Factor (+)
        0, 1, 0, 0,           # Solar_Power (+)
    ),
    "Conduction_Loss": (
        0, 0, 0, 0, 0,
        -1, 0,                # R_Value (-)
        0, 1,                 # Window_U_Value (+)
        0, 0, 0, 0,
        -1, 0, 1, 1,          # Outside_Temp (-), Wind_Speed (+), Previous_Temp (+)
    ),
    "Infiltration_Loss": (
        0, 0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 1, 0, 0,           # ACH (+)
        -1, 0, 0, 1,          # Outside_Temp (-), Previous_Temp (+)
    ),
}

TARGETS = list(MONOTONE.keys())


def train_all():
    print("Step 1: Reading training_data.csv...")
    df = pd.read_csv("training_data.csv")
    X = df[FEATURES]

    models = {}
    metrics = {}

    for target in TARGETS:
        print(f"\nStep 2: Training model for '{target}'...")
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            monotone_constraints=MONOTONE[target],
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        metrics[target] = {"rmse": round(rmse, 3), "r2": round(r2, 4)}
        print(f"  {target}: RMSE={rmse:.3f}  R^2={r2:.4f}")

        models[target] = model

    return models, metrics


if __name__ == "__main__":
    models, metrics = train_all()

    bundle = {
        "models": models,
        "features": FEATURES,
        "targets": TARGETS,
        "metrics": metrics,
    }
    joblib.dump(bundle, "thermal_model_v4.pkl")

    print("\n" + "=" * 50)
    print("SUCCESS: thermal_model_v4.pkl saved.")
    print("Validation metrics (held-out 20% test split):")
    for target, m in metrics.items():
        print(f"  {target:20s} RMSE={m['rmse']:>10.3f}   R^2={m['r2']:.4f}")
    print("=" * 50)
    print("\nPut this metrics table directly in your PPT/report -- it's your")
    print("answer to 'how do you know the model is accurate.'")
