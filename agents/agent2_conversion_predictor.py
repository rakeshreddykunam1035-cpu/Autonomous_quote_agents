"""
Agent 2 -- Conversion Predictor (FULLY AUTO)

Input:  raw quote-level data + Agent 1's risk_tier (chained downstream)
Output: {bind_probability: 0-1, conversion_band: Low/Medium/High}

Trained directly on the real historical label Policy_Bind (Yes/No),
so this one is a genuine supervised model on ground truth, not a proxy.
"""
import sys, os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import build_dataset

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "agent2_conversion_predictor.joblib")


def _add_risk_tier_feature(feats: pd.DataFrame, risk_label: pd.Series) -> pd.DataFrame:
    """Fold Agent 1's output in as a feature, since Agent 2 consumes it downstream."""
    feats = feats.copy()
    dummies = pd.get_dummies(risk_label, prefix="RiskTier")
    return pd.concat([feats, dummies], axis=1)


def train(save: bool = True):
    df, feats, risk_label, bind_label = build_dataset()
    feats_plus = _add_risk_tier_feature(feats, risk_label)

    X_train, X_test, y_train, y_test = train_test_split(
        feats_plus, bind_label, test_size=0.2, random_state=42, stratify=bind_label
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("=== Agent 2: Conversion Predictor -- holdout performance ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, preds))

    if save:
        joblib.dump({"model": clf, "feature_columns": list(feats_plus.columns)}, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH}")

    return clf, list(feats_plus.columns)


def load():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_columns"]


def run(feature_row: pd.DataFrame, risk_tier: str, model=None, feature_columns=None) -> dict:
    if model is None:
        model, feature_columns = load()

    feature_row = feature_row.copy()
    for tier in ["Low", "Medium", "High"]:
        feature_row[f"RiskTier_{tier}"] = 1 if tier == risk_tier else 0
    feature_row = feature_row.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(feature_row)[:, 1]
    results = []
    for p in proba:
        band = "High" if p >= 0.5 else ("Medium" if p >= 0.25 else "Low")
        results.append({"bind_probability": round(float(p), 4), "conversion_band": band})
    return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    train()
