# Autonomous Quote Agents -- Starter Project

## Setup
```
pip install pandas numpy scikit-learn joblib streamlit
```

## Train the ML agents (run once)
```
python3 agents/agent1_risk_profiler.py
python3 agents/agent2_conversion_predictor.py
```
This saves trained models into `models/`.

## Run the full pipeline on sample quotes (CLI)
```
python3 orchestrator.py
```

## Launch the dashboard
```
streamlit run app.py
```

## What's built
- `data_prep.py` -- cleaning, ordinal/one-hot encoding, engineered Risk_Label
- `agents/agent1_risk_profiler.py` -- GradientBoosting classifier -> Low/Medium/High risk tier
- `agents/agent2_conversion_predictor.py` -- GradientBoosting classifier -> bind probability
- `agents/agent3_premium_advisor.py` -- rule engine + RAG-grounded justification
- `agents/agent4_decision_router.py` -- rule-based Auto-Approve / Follow-Up / Escalate routing
- `rag.py` + `rag_kb/underwriting_guidelines.md` -- TF-IDF retrieval over a small underwriting
  guidelines doc, used by Agents 3 & 4 to ground their justification text in an actual clause
- `orchestrator.py` -- chains all 4 agents for one quote
- `app.py` -- Streamlit ops dashboard

## Important known limitation
On this dataset, `Policy_Bind` (the conversion outcome) has ~0 correlation with every other
column -- confirmed via per-feature correlation and per-category bind-rate checks (~22.2% in
every slice). Agent 2's model is real code and trains without error, but it has no genuine
signal to learn from in this data, so its ROC-AUC lands at ~0.50 (random). Treat it as a
working pipeline stage to demonstrate the architecture, not a validated predictor -- it would
need richer features (e.g. price competitiveness vs. market, contact timing, agent follow-up
count) or a different dataset to actually predict conversion.

Agent 1's risk label is engineered from a business-rule scorecard (see `engineer_risk_label`
in `data_prep.py`) since the raw data has no risk column -- so its near-perfect accuracy
reflects the model recovering a deterministic rule, not real-world generalization. Worth
recalibrating the scorecard weights, or getting real risk-tier labels, before relying on it.
