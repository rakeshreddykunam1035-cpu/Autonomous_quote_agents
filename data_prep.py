"""
data_prep.py
Loads the raw quote data and builds the feature set shared by all agents.
Also engineers a business-rule Risk_Label (Low/Medium/High) since the raw
data has no explicit risk tier -- Agent 1 will later learn to predict this
label from signals a real quote has *before* underwriting review, so the
label logic below intentionally leans on accident/citation/experience
history rather than anything only known post-bind.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/Autonomous_QUOTE_AGENTS.csv"

ORDINAL_MAPS = {
    "Sal_Range": {
        "<= $ 25 K": 0,
        "> $ 25 K <= $ 40 K": 1,
        "> $ 40 K <= $ 60 K": 2,
        "> $ 60 K <= $ 90 K": 3,
        "> $ 90 K": 4,
    },
    "Annual_Miles_Range": {
        "<= 7.5 K": 0,
        "> 7.5 K & <= 15 K": 1,
        "> 15 K & <= 25 K": 2,
        "> 25 K & <= 35 K": 3,
        "> 35 K & <= 45 K": 4,
        "> 45 K & <= 55 K": 5,
        "> 55 K": 6,
    },
    "Vehicl_Cost_Range": {
        "<= $ 10 K": 0,
        "> $ 10 K <= $ 20 K": 1,
        "> $ 20 K <= $ 30 K": 2,
        "> $ 30 K <= $ 40 K": 3,
        "> $ 40 K": 4,
    },
    "Education": {
        "High School": 0,
        "College": 1,
        "Bachelors": 2,
        "Masters": 3,
        "Ph.D": 4,
    },
    "Coverage": {"Basic": 0, "Balanced": 1, "Enhanced": 2},
}


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ORDINAL_MAPS:
        df[col] = _clean_str(df[col])
    df["Marital_Status"] = df["Marital_Status"].replace({"Dirvorced": "Divorced"})
    return df


def engineer_risk_label(df: pd.DataFrame) -> pd.Series:
    """
    Business-rule ground truth for training the Risk Profiler (Agent 1).
    Points-based scorecard -> Low / Medium / High.
    """
    score = np.zeros(len(df))
    score += df["Prev_Accidents"] * 3
    score += df["Prev_Citations"] * 2
    score += (df["Driving_Exp"] <= 3).astype(int) * 2          # new driver
    score += (df["Driver_Age"] < 25).astype(int) * 1           # young driver
    score += (df["Veh_Usage"] == "Business").astype(int) * 1
    score += (df["Annual_Miles_Range"].map(ORDINAL_MAPS["Annual_Miles_Range"]) >= 5).astype(int) * 1
    score += (df["HH_Drivers"] > df["HH_Vehicles"]).astype(int) * 1  # more drivers than cars

    labels = pd.cut(
        score,
        bins=[-0.1, 1.5, 4.5, 100],
        labels=["Low", "Medium", "High"],
    )
    return labels.astype(str)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot + ordinal encoded feature matrix used by both ML agents."""
    feat = pd.DataFrame(index=df.index)

    # Numeric / behavioural signals
    feat["Driver_Age"] = df["Driver_Age"]
    feat["Driving_Exp"] = df["Driving_Exp"]
    feat["HH_Vehicles"] = df["HH_Vehicles"]
    feat["HH_Drivers"] = df["HH_Drivers"]
    feat["Prev_Accidents"] = df["Prev_Accidents"]
    feat["Prev_Citations"] = df["Prev_Citations"]
    feat["Quoted_Premium"] = df["Quoted_Premium"]

    # Ordinal encodings
    for col, mapping in ORDINAL_MAPS.items():
        feat[col + "_ord"] = df[col].map(mapping)

    # One-hot for low-cardinality nominal fields
    nominal_cols = ["Agent_Type", "Region", "Policy_Type", "Gender",
                     "Marital_Status", "Veh_Usage", "Re_Quote"]
    dummies = pd.get_dummies(df[nominal_cols], prefix=nominal_cols)
    feat = pd.concat([feat, dummies], axis=1)

    return feat


def build_dataset(path: str = RAW_PATH):
    """Convenience entry point: returns (raw_df, feature_df, risk_label, bind_label)."""
    df = load_raw(path)
    risk_label = engineer_risk_label(df)
    bind_label = (df["Policy_Bind"] == "Yes").astype(int)
    features = build_feature_frame(df)
    return df, features, risk_label, bind_label


if __name__ == "__main__":
    df, feats, risk, bind = build_dataset()
    print("Rows:", len(df))
    print("Feature columns:", feats.shape[1])
    print("\nRisk label distribution:\n", risk.value_counts())
    print("\nBind rate:", bind.mean().round(4))
