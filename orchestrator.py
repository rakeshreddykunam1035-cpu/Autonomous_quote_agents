"""
orchestrator.py -- Runs a single raw quote row through all 4 agents in
sequence, passing each agent's output into the next, matching the pipeline
diagram: Risk Profiler -> Conversion Predictor -> Premium Advisor -> Decision Router.

This is a plain Python chain rather than LangGraph/CrewAI/AutoGen. The
brief mentions those frameworks as options; a hand-rolled sequential chain
is functionally identical for 4 agents with a fixed linear order, and
keeps the project dependency-free. If you want to swap in LangGraph later,
each `agents/agentN_*.py::run()` function is already a clean tool/node you
can drop into a graph unchanged.
"""
import sys, os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_feature_frame, ORDINAL_MAPS
from agents import agent1_risk_profiler as agent1
from agents import agent2_conversion_predictor as agent2
from agents import agent3_premium_advisor as agent3
from agents import agent4_decision_router as agent4

_a1_model, _a1_cols = None, None
_a2_model, _a2_cols = None, None


def _load_models():
    global _a1_model, _a1_cols, _a2_model, _a2_cols
    if _a1_model is None:
        _a1_model, _a1_cols = agent1.load()
    if _a2_model is None:
        _a2_model, _a2_cols = agent2.load()


def run_quote(raw_row: pd.Series) -> dict:
    """
    raw_row: a single row (pandas Series) with the same raw columns as
    Autonomous_QUOTE_AGENTS.csv (i.e. one row of df from data_prep.load_raw()).
    """
    _load_models()
    row_df = pd.DataFrame([raw_row])
    feats = build_feature_frame(row_df)

    # --- Agent 1: Risk Profiler ---
    a1_out = agent1.run(feats, model=_a1_model, feature_columns=_a1_cols)

    # --- Agent 2: Conversion Predictor (consumes Agent 1's risk_tier) ---
    a2_out = agent2.run(feats, risk_tier=a1_out["risk_tier"],
                         model=_a2_model, feature_columns=_a2_cols)

    # --- Agent 3: Premium Advisor (consumes Agent 1 + Agent 2 outputs) ---
    a3_out = agent3.run(
        risk_tier=a1_out["risk_tier"],
        risk_score=a1_out["risk_score"],
        bind_probability=a2_out["bind_probability"],
        coverage=raw_row["Coverage"],
        quoted_premium=float(raw_row["Quoted_Premium"]),
        re_quote=raw_row["Re_Quote"],
    )

    # --- Agent 4: Decision Router (consumes Agent 1, 2, 3 + raw fields) ---
    a4_out = agent4.run(
        risk_tier=a1_out["risk_tier"],
        confidence=a1_out["confidence"],
        bind_probability=a2_out["bind_probability"],
        re_quote=raw_row["Re_Quote"],
        hh_drivers=int(raw_row["HH_Drivers"]),
        hh_vehicles=int(raw_row["HH_Vehicles"]),
    )

    return {
        "quote_num": raw_row.get("Quote_Num", None),
        "agent1_risk_profiler": a1_out,
        "agent2_conversion_predictor": a2_out,
        "agent3_premium_advisor": a3_out,
        "agent4_decision_router": a4_out,
    }


if __name__ == "__main__":
    from data_prep import load_raw
    df = load_raw()
    sample = df.sample(3, random_state=7)
    for _, row in sample.iterrows():
        result = run_quote(row)
        print("=" * 70)
        print("Quote:", result["quote_num"])
        print("Agent 1 (Risk):      ", result["agent1_risk_profiler"])
        print("Agent 2 (Conversion):", result["agent2_conversion_predictor"])
        print("Agent 3 (Premium):   ", result["agent3_premium_advisor"])
        print("Agent 4 (Route):     ", result["agent4_decision_router"])
