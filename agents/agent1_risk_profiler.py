"""
Agent 1 -- Risk Profiler (FULLY AUTO)

Input:  raw quote-level data (household, driver, vehicle signals)
Output: {risk_tier: Low/Medium/High, risk_score: 0-1 (P(High)+0.5*P(Medium)),
         confidence: max class probability}

Trains a gradient-boosted tree on the engineered Risk_Label (see data_prep.py)
using only features known at quote time (no premium/bind leakage into label).
"""
import sys, os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import build_dataset

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "agent1_risk_profiler.joblib")


def train(save: bool = True):
    df, feats, risk_label, _ = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        feats, risk_label, test_size=0.2, random_state=42, stratify=risk_label
    )

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("=== Agent 1: Risk Profiler -- holdout performance ===")
    print(classification_report(y_test, preds))

    if save:
        joblib.dump({"model": clf, "feature_columns": list(feats.columns)}, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH}")

    return clf, list(feats.columns)


def load():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_columns"]


def run(feature_row: pd.DataFrame, model=None, feature_columns=None) -> dict:
    """
    feature_row: single-row (or batch) DataFrame already built via
    data_prep.build_feature_frame, aligned to feature_columns.
    """
    if model is None:
        model, feature_columns = load()
    feature_row = feature_row.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(feature_row)
    classes = list(model.classes_)
    results = []
    for row in proba:
        p = dict(zip(classes, row))
        tier = classes[np.argmax(row)]
        risk_score = p.get("High", 0) + 0.5 * p.get("Medium", 0)
        results.append({
            "risk_tier": tier,
            "risk_score": round(float(risk_score), 4),
            "confidence": round(float(max(row)), 4),
        })
    return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    train()
