"""
Agent 4 -- Decision Router (ESCALATE-ONLY)

Input:  risk_tier, risk_score, confidence (Agent 1); bind_probability (Agent 2);
         re_quote, HH_Drivers, HH_Vehicles
Output: {route: Auto-Approve / Agent Follow-Up / Escalate-to-Underwriter, reason}

Pure rule-based per the brief ("ESCALATE-ONLY" tag on the diagram) -- this
agent does not touch the ML models directly, it only reads their outputs.
Thresholds mirror Section 3 of the underwriting guidelines KB.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import generate_justification

CONFIDENCE_ESCALATE_THRESHOLD = 0.70


def run(risk_tier: str, confidence: float, bind_probability: float,
        re_quote: str, hh_drivers: int, hh_vehicles: int) -> dict:

    reasons = []

    # 3.1(a) High risk + model uncertain about the premium fit
    if risk_tier == "High":
        reasons.append("high_risk_tier")

    # 3.1(b) Low confidence in the risk prediction itself
    if confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        reasons.append("low_model_confidence")

    # 3.1(c) more drivers than vehicles -- potential undisclosed risk
    if hh_drivers - hh_vehicles >= 2:
        reasons.append("household_driver_vehicle_mismatch")

    if reasons:
        rag = generate_justification("escalate underwriter high risk low confidence household drivers")
        return {
            "route": "Escalate-to-Underwriter",
            "reasons": reasons,
            "justification": rag["justification"],
        }

    # 3.2 Not escalated, but not a clean auto-approve either
    if re_quote == "Yes" or risk_tier == "Medium":
        rag = generate_justification("route to agent follow-up re-quoted not auto-approved")
        return {
            "route": "Agent Follow-Up",
            "reasons": ["re_quote_or_medium_risk_needs_review"],
            "justification": rag["justification"],
        }

    # 3.3 Clean Low-risk, high-confidence, no re-quote history
    rag = generate_justification("auto-approval low risk high confidence no re-quote")
    return {
        "route": "Auto-Approve",
        "reasons": ["low_risk_high_confidence_clean_history"],
        "justification": rag["justification"],
    }


if __name__ == "__main__":
    examples = [
        dict(risk_tier="Low", confidence=0.95, bind_probability=0.3,
             re_quote="No", hh_drivers=2, hh_vehicles=2),
        dict(risk_tier="High", confidence=0.9, bind_probability=0.5,
             re_quote="No", hh_drivers=2, hh_vehicles=2),
        dict(risk_tier="Medium", confidence=0.6, bind_probability=0.4,
             re_quote="No", hh_drivers=2, hh_vehicles=2),
        dict(risk_tier="Low", confidence=0.9, bind_probability=0.3,
             re_quote="No", hh_drivers=4, hh_vehicles=2),
    ]
    for ex in examples:
        print(ex)
        print(" ->", run(**ex))
        print()
