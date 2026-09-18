"""Train the mushroom (edible e / poisonous p) classifier.

Good-practice notes (and what was fixed vs the previous version):
  - Drops identifier / leakage columns (ID, mushroom_id) — the old script
    trained on them, which is why validation accuracy was a bogus 1.0.
  - Drops zero-variance columns (veil-type, veil-color are constant).
  - Explicit numeric vs categorical split. The old script used
    select_dtypes, which misclassified 'ring-number' (contains 'None'
    strings, so object dtype) as categorical.
  - Cleans data: '' / whitespace-only -> NaN (kept as 'missing' category);
    'None' strings in odor / ring-type are REAL categories (no odor / no
    ring) and are preserved. 'ring-number' is coerced to numeric.
  - All preprocessing lives inside a Pipeline so imputers/encoders are fit
    on training folds only (no leakage into validation).
  - Stratified train/validation split + stratified 5-fold CV + a small
    GridSearchCV (scoring=f1_weighted) instead of a single unvalidated fit.
  - Reports weighted + macro precision/recall/F1, confusion matrix,
    classification report and ROC-AUC, plus train-vs-val gap to spot
    overfitting.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
MODEL_PKL = os.path.join(BASE_DIR, "model.pkl")
METRICS_JSON = os.path.join(BASE_DIR, "metrics.json")

TARGET = "class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Identifiers / row keys: pure leakage, never features.
ID_COLS = ["ID", "mushroom_id"]
# Genuinely numeric (ring-number has 'None' strings -> coerced to NaN).
NUMERIC_FEATURES = ["number_of_bruises", "ring-number"]


def load_and_clean(path=TRAIN_CSV):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()

    # Drop leakage identifiers.
    dropped = [c for c in ID_COLS if c in df.columns]
    df = df.drop(columns=dropped)

    # '' / whitespace-only -> NaN (becomes 'missing' category later).
    # NOTE: the literal string 'None' is a valid category for odor
    # (no odor) and ring-type (no ring), so it is preserved.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # Coerce declared numerics ('None' -> NaN -> median-imputed).
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop zero-variance columns (e.g. veil-type / veil-color are constant).
    for col in [c for c in df.columns if c != TARGET]:
        if df[col].nunique(dropna=False) <= 1:
            dropped.append(col)
    df = df.drop(columns=[c for c in dropped if c in df.columns and c != TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    cat_features = [c for c in X.columns if c not in NUMERIC_FEATURES]
    num_features = [c for c in NUMERIC_FEATURES if c in X.columns]
    return X, y, cat_features, num_features, dropped


def build_pipeline(cat_features, num_features, clf_params=None):
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_features),
            ("num", num_pipe, num_features),
        ]
    )
    clf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, **(clf_params or {})
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def summarize(y_true, y_pred):
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_w), 4),
        "recall": round(float(recall_w), 4),
        "f1_score": round(float(f1_w), 4),
        "precision_macro": round(float(precision_m), 4),
        "recall_macro": round(float(recall_m), 4),
        "f1_macro": round(float(f1_m), 4),
    }


def main():
    X, y, cat_features, num_features, dropped = load_and_clean()
    labels = sorted(y.unique().tolist())
    print(f"Rows: {len(X)} | features: {len(X.columns)} "
          f"({len(cat_features)} cat + {len(num_features)} num)")
    print(f"Dropped (leakage/constant): {dropped}")
    print(f"Class balance: {y.value_counts().to_dict()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    base = build_pipeline(cat_features, num_features)
    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 25],
        "classifier__min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        base, param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best params: {search.best_params_} "
          f"| CV f1_weighted: {search.best_score_:.4f}")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_metrics = summarize(y_train, y_train_pred)
    val_metrics = summarize(y_val, y_val_pred)
    print(f"Train: {train_metrics}")
    print(f"Val:   {val_metrics}")

    # ROC-AUC (positive class = 'p' / poisonous).
    roc_auc = None
    try:
        pos = list(model.classes_).index("p")
        roc_auc = round(
            float(roc_auc_score((y_val == "p").astype(int),
                                model.predict_proba(X_val)[:, pos])), 4
        )
        print(f"Val ROC-AUC (p): {roc_auc}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"ROC-AUC skipped: {exc}")

    cm = confusion_matrix(y_val, y_val_pred, labels=labels).tolist()
    report = classification_report(y_val, y_val_pred, output_dict=True,
                                   zero_division=0)

    # Top-10 importances for the metrics dashboard / debugging.
    top_features = []
    try:
        names = list(model.named_steps["preprocessor"].get_feature_names_out())
        imps = model.named_steps["classifier"].feature_importances_
        for i in np.argsort(imps)[::-1][:10]:
            top_features.append(
                {"feature": names[i], "importance": round(float(imps[i]), 4)}
            )
    except Exception as exc:
        print(f"Importances skipped: {exc}")

    joblib.dump(model, MODEL_PKL)

    metrics_data = {
        "target": TARGET,
        "labels": labels,
        "dropped_columns": dropped,
        "categorical_features": cat_features,
        "numeric_features": num_features,
        "model": "RandomForestClassifier",
        "model_params": search.best_params_,
        "cv": {"n_splits": 5, "scoring": "f1_weighted",
               "best_score": round(float(search.best_score_), 4)},
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_roc_auc_poisonous": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "top_features": top_features,
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_data, f, indent=4)

    gap = train_metrics["f1_score"] - val_metrics["f1_score"]
    print(f"Train-val F1 gap: {gap:.4f} (large gap => overfitting)")
    print(f"Saved {MODEL_PKL} and {METRICS_JSON}")


if __name__ == "__main__":
    main()
