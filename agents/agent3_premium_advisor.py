"""
Agent 3 -- Premium Advisor (HYBRID)

Input:  risk_tier + risk_score (Agent 1), bind_probability (Agent 2),
         Coverage tier, Quoted_Premium, Re_Quote flag
Output: {recommended_action, adjusted_premium, discount_pct, justification, citations}

The "hybrid" part: the discount decision itself is deterministic rules
(auditable, matches Section 2 of the guidelines KB), but the plain-English
justification is retrieved via RAG against the same guideline document so
the reasoning an agent/underwriter sees is always grounded in an actual
clause rather than free-floating text.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import generate_justification


def run(risk_tier: str, risk_score: float, bind_probability: float,
        coverage: str, quoted_premium: float, re_quote: str) -> dict:

    discount_pct = 0.0
    action = "hold_premium"
    # rag_query mirrors the actual clause wording for the rule that fires,
    # rather than a loose paraphrase -- see module docstring for why this
    # matters for retrieval accuracy.
    rag_query = "high risk never eligible autonomous discounting underwriter sign-off"

    if risk_tier == "High":
        action = "hold_premium"
        discount_pct = 0.0
        rag_query = "high risk never eligible autonomous discounting underwriter sign-off"

    elif risk_tier == "Low":
        if bind_probability < 0.40:
            discount_pct = 0.10
            action = "offer_discount"
            rag_query = "discounts up to 10 percent low risk accounts conversion likelihood below 40"
        elif bind_probability > 0.60:
            discount_pct = 0.0
            action = "hold_premium_already_converting"
            rag_query = "no discount accounts conversion likelihood already above 60"
        else:
            discount_pct = 0.05
            action = "offer_small_discount"
            rag_query = "discounts up to 10 percent low risk accounts conversion likelihood below 40"

    elif risk_tier == "Medium":
        if coverage in ("Basic", "Balanced") and bind_probability < 0.40:
            discount_pct = 0.05
            action = "offer_small_discount"
            rag_query = "smaller discount band up to 5 percent medium risk basic balanced coverage"
        else:
            discount_pct = 0.0
            action = "hold_premium"
            rag_query = "enhanced coverage medium risk accounts should not be discounted"

    # Re-quoted / price-shopped policies: slightly more willing to discount
    # within an already-approved band, per Section 3.2 guidance
    if re_quote == "Yes" and action == "hold_premium" and risk_tier != "High":
        discount_pct = 0.03
        action = "offer_small_discount_requote"
        rag_query = "re-quoted policies customer shopped price multiple times agent follow-up"

    adjusted_premium = round(quoted_premium * (1 - discount_pct), 2)
    rag_result = generate_justification(rag_query)

    return {
        "recommended_action": action,
        "discount_pct": discount_pct,
        "original_premium": quoted_premium,
        "adjusted_premium": adjusted_premium,
        "justification": rag_result["justification"],
        "guideline_citations": rag_result["citations"],
    }


if __name__ == "__main__":
    examples = [
        dict(risk_tier="Low", risk_score=0.05, bind_probability=0.30,
             coverage="Balanced", quoted_premium=700.0, re_quote="No"),
        dict(risk_tier="High", risk_score=0.85, bind_probability=0.55,
             coverage="Basic", quoted_premium=780.0, re_quote="No"),
        dict(risk_tier="Medium", risk_score=0.45, bind_probability=0.35,
             coverage="Enhanced", quoted_premium=740.0, re_quote="Yes"),
    ]
    for ex in examples:
        result = run(**ex)
        print(ex)
        print(" ->", result)
        print()
