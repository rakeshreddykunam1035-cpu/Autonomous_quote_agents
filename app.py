"""
app.py -- Agent Operations Dashboard (Streamlit)
 
Run with:  streamlit run app.py
 
Lets an ops user pick a real historical quote (or hand-enter one) and watch
it flow through all 4 agents live, with each agent's output and RAG-grounded
justification shown as its own pipeline stage.
"""
import sys, os
import streamlit as st
import pandas as pd
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_raw, ORDINAL_MAPS
from orchestrator import run_quote
 
st.set_page_config(page_title="Quote Agents | Operations", layout="wide", page_icon="◆")
 
# ---------------------------------------------------------------------------
# Design tokens -- reuses the automation-tier colors from the case study
# brief itself (teal = Fully Auto, amber = Hybrid, red = Escalate-only)
# ---------------------------------------------------------------------------
TEAL = "#0F8B7E"
TEAL_BG = "#E7F5F3"
AMBER = "#B8860B"
AMBER_BG = "#FBF3DE"
RED = "#C0392B"
RED_BG = "#FBEAE8"
INK = "#1C2430"
MUTED = "#6B7480"
BORDER = "#E4E7EC"
CANVAS = "#F7F8FA"
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
 
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
.stApp {{
    background-color: {CANVAS};
}}
h1, h2, h3, .brand-title {{
    font-family: 'Manrope', sans-serif;
}}
section[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid {BORDER};
}}
div.stButton > button[kind="primary"] {{
    background-color: {TEAL};
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}}
div.stButton > button[kind="primary"]:hover {{
    background-color: #0C6F64;
}}
 
.brand-row {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
}}
.brand-mark {{
    width: 30px; height: 30px; border-radius: 8px; background: {TEAL};
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-family: 'Manrope', sans-serif;
}}
.brand-title {{ font-weight: 800; font-size: 1.05rem; margin: 0; }}
.brand-sub {{ color: {MUTED}; font-size: 0.78rem; margin: 0; }}
 
.page-header {{ margin-bottom: 1.4rem; }}
.page-header h1 {{ font-size: 1.7rem; font-weight: 800; margin-bottom: 2px; }}
.page-header p {{ color: {MUTED}; font-size: 0.95rem; margin: 0; }}
 
.metric-row {{ display: flex; gap: 14px; margin-bottom: 1.6rem; flex-wrap: wrap; }}
.metric-card {{
    background: white; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 12px 18px; min-width: 150px; box-shadow: 0 1px 3px rgba(16,24,40,0.05);
}}
.metric-card.accent-full {{ border-left: 3px solid {TEAL}; }}
.metric-card.accent-hybrid {{ border-left: 3px solid {AMBER}; }}
.metric-card.accent-escalate {{ border-left: 3px solid {RED}; }}
.metric-card .label {{ color: {MUTED}; font-size: 0.75rem; font-weight: 500; }}
.metric-card .value {{ font-size: 1.3rem; font-weight: 700; margin-top: 2px; }}
 
.agent-card {{
    background: white; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 20px 22px; height: 100%; border-left: 4px solid {BORDER};
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}}
.agent-card.tier-full {{ border-left-color: {TEAL}; }}
.agent-card.tier-hybrid {{ border-left-color: {AMBER}; }}
.agent-card.tier-escalate {{ border-left-color: {RED}; }}
 
.tag {{
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    padding: 3px 10px; border-radius: 999px; letter-spacing: 0.04em;
    margin-bottom: 10px;
}}
.tag-full {{ background: {TEAL_BG}; color: {TEAL}; }}
.tag-hybrid {{ background: {AMBER_BG}; color: {AMBER}; }}
.tag-escalate {{ background: {RED_BG}; color: {RED}; }}
 
.agent-name {{ font-weight: 700; font-size: 1.05rem; margin-bottom: 14px; }}
.stat-block {{ margin-bottom: 12px; }}
.stat-label {{ color: {MUTED}; font-size: 0.72rem; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 2px; }}
.stat-value {{ font-weight: 600; font-size: 0.95rem; word-break: break-word;
               overflow-wrap: anywhere; }}
.reason-list {{ margin: 4px 0 0 0; padding-left: 18px; font-size: 0.9rem; }}
.reason-list li {{ margin-bottom: 3px; }}
.why-box {{ margin-top: 14px; padding-top: 12px; border-top: 1px dashed {BORDER}; }}
.why-label {{ color: {MUTED}; font-size: 0.72rem; font-weight: 700;
              text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 4px; }}
.why-text {{ font-size: 0.85rem; color: #45505E; line-height: 1.5; }}
 
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_data
def get_data():
    return load_raw()
 
 
df = get_data()
 
# ---------------------------------------------------------------------------
# Presentation-layer translations: agents return machine-readable codes
# (e.g. "offer_small_discount"); underwriters should see plain English.
# The raw codes are still visible in the JSON expander for audit purposes.
# ---------------------------------------------------------------------------
ACTION_LABELS = {
    "offer_discount": "Offer 10% discount",
    "offer_small_discount": "Offer small discount (5%)",
    "offer_small_discount_requote": "Offer small discount (3%) — repeat quote",
    "hold_premium": "Hold at quoted premium",
    "hold_premium_already_converting": "Hold premium — likely to convert without one",
}
 
REASON_LABELS = {
    "high_risk_tier": "Flagged as high risk",
    "low_model_confidence": "Risk model confidence below 70% threshold",
    "household_driver_vehicle_mismatch": "More drivers than vehicles in the household",
    "re_quote_or_medium_risk_needs_review": "Repeat quote or medium risk — needs a look",
    "low_risk_high_confidence_clean_history": "Low risk, high confidence, clean quote history",
}
 
ROUTE_STYLE = {
    "Auto-Approve": ("tier-full", "tag-full", "FULLY AUTO"),
    "Agent Follow-Up": ("tier-hybrid", "tag-hybrid", "NEEDS REVIEW"),
    "Escalate-to-Underwriter": ("tier-escalate", "tag-escalate", "ESCALATE"),
}
 
 
def humanize_action(code: str) -> str:
    return ACTION_LABELS.get(code, code.replace("_", " ").capitalize())
 
 
def humanize_reason(code: str) -> str:
    return REASON_LABELS.get(code, code.replace("_", " ").capitalize())
 
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">Q</div>'
        '<div><p class="brand-title">Quote Agents</p>'
        '<p class="brand-sub">operations console</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Select a quote**")
    mode = st.radio("Source", ["Sample from dataset", "Enter manually"], label_visibility="collapsed")
 
    if mode == "Sample from dataset":
        quote_num = st.selectbox("Quote number", df["Quote_Num"].sample(200, random_state=1).tolist())
        row = df[df["Quote_Num"] == quote_num].iloc[0]
    else:
        row = pd.Series({
            "Quote_Num": "MANUAL-ENTRY",
            "Agent_Type": {"Exclusive agent (sells one carrier)": "EA",
                           "Independent agent (sells multiple carriers)": "IA"}[
                st.selectbox("Agent type", ["Exclusive agent (sells one carrier)",
                                            "Independent agent (sells multiple carriers)"])
            ],
            "Region": st.selectbox("Region", list("ABCDEFGH")),
            "Policy_Type": st.selectbox("Vehicle type", ["Car", "Van", "Truck"]),
            "HH_Vehicles": st.slider("Vehicles in household", 1, 6, 2),
            "HH_Drivers": st.slider("Licensed drivers in household", 1, 6, 2),
            "Driver_Age": st.slider("Driver's age", 16, 85, 35),
            "Driving_Exp": st.slider("Years of driving experience", 0, 60, 10),
            "Prev_Accidents": 1 if st.selectbox("Any prior at-fault accident?", ["No", "Yes"]) == "Yes" else 0,
            "Prev_Citations": 1 if st.selectbox("Any prior traffic citation?", ["No", "Yes"]) == "Yes" else 0,
            "Gender": {"Male": "M", "Female": "F"}[st.selectbox("Gender", ["Male", "Female"])],
            "Marital_Status": st.selectbox("Marital status", list(df["Marital_Status"].unique())),
            "Education": st.selectbox("Highest education level", list(ORDINAL_MAPS["Education"].keys())),
            "Sal_Range": st.selectbox("Household income range", list(ORDINAL_MAPS["Sal_Range"].keys())),
            "Coverage": st.selectbox("Coverage tier", list(ORDINAL_MAPS["Coverage"].keys())),
            "Veh_Usage": st.selectbox("Primary vehicle use", ["Commute", "Pleasure", "Business"]),
            "Annual_Miles_Range": st.selectbox("Annual mileage", list(ORDINAL_MAPS["Annual_Miles_Range"].keys())),
            "Vehicl_Cost_Range": st.selectbox("Vehicle value", list(ORDINAL_MAPS["Vehicl_Cost_Range"].keys())),
            "Re_Quote": st.selectbox("Has the customer requoted this policy before?", ["No", "Yes"]),
            "Quoted_Premium": st.number_input("Quoted premium ($)", 100.0, 5000.0, 750.0),
        })
 
    st.markdown("")
    run_btn = st.button("Run through pipeline", type="primary", use_container_width=True)
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="page-header"><h1>Quote pipeline</h1>'
    '<p>Risk Profiler &rarr; Conversion Predictor &rarr; Premium Advisor &rarr; Decision Router</p></div>',
    unsafe_allow_html=True,
)
 
with st.expander("Selected quote -- details", expanded=False):
    DISPLAY_NAMES = {
        "Quote_Num": "Quote number", "Agent_Type": "Agent type", "Region": "Region",
        "Policy_Type": "Vehicle type", "HH_Vehicles": "Vehicles in household",
        "HH_Drivers": "Drivers in household", "Driver_Age": "Driver's age",
        "Driving_Exp": "Years of driving experience", "Prev_Accidents": "Prior at-fault accident",
        "Prev_Citations": "Prior traffic citation", "Gender": "Gender",
        "Marital_Status": "Marital status", "Education": "Education level",
        "Sal_Range": "Household income range", "Coverage": "Coverage tier",
        "Veh_Usage": "Primary vehicle use", "Annual_Miles_Range": "Annual mileage",
        "Vehicl_Cost_Range": "Vehicle value", "Re_Quote": "Repeat quote",
        "Quoted_Premium": "Quoted premium ($)", "Policy_Bind": "Bound? (historical)",
        "Q_Creation_DT": "Quote created", "Q_Valid_DT": "Quote expires",
        "Policy_Bind_DT": "Bind date (historical)", "Agent_Num": "Agent ID",
    }
    display_row = row.rename(index=DISPLAY_NAMES)
    for code, label in [("Prior at-fault accident", None), ("Prior traffic citation", None)]:
        if code in display_row.index:
            display_row[code] = "Yes" if display_row[code] == 1 else "No"
    st.dataframe(display_row.to_frame().T, use_container_width=True, hide_index=True)
 
if run_btn:
    result = run_quote(row)
    a1 = result["agent1_risk_profiler"]
    a2 = result["agent2_conversion_predictor"]
    a3 = result["agent3_premium_advisor"]
    a4 = result["agent4_decision_router"]
 
    route_accent = {"Auto-Approve": "accent-full", "Agent Follow-Up": "accent-hybrid",
                     "Escalate-to-Underwriter": "accent-escalate"}.get(a4["route"], "")
 
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card"><div class="label">RISK TIER</div><div class="value">{a1['risk_tier']}</div></div>
          <div class="metric-card"><div class="label">BIND PROBABILITY</div><div class="value">{a2['bind_probability']*100:.1f}%</div></div>
          <div class="metric-card"><div class="label">ADJUSTED PREMIUM</div><div class="value">${a3['adjusted_premium']:,.2f}</div></div>
          <div class="metric-card {route_accent}"><div class="label">ROUTE</div><div class="value">{a4['route']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    c1, c2, c3, c4 = st.columns(4)
 
    with c1:
        st.markdown(f"""
        <div class="agent-card tier-full">
          <span class="tag tag-full">FULLY AUTO</span>
          <div class="agent-name">Risk Profiler</div>
          <div class="stat-block"><div class="stat-label">Risk tier</div><div class="stat-value">{a1['risk_tier']}</div></div>
          <div class="stat-block"><div class="stat-label">Risk score</div><div class="stat-value">{a1['risk_score']:.3f}</div></div>
          <div class="stat-block"><div class="stat-label">Confidence</div><div class="stat-value">{a1['confidence']:.1%}</div></div>
        </div>
        """, unsafe_allow_html=True)
 
    with c2:
        st.markdown(f"""
        <div class="agent-card tier-full">
          <span class="tag tag-full">FULLY AUTO</span>
          <div class="agent-name">Conversion Predictor</div>
          <div class="stat-block"><div class="stat-label">Bind probability</div><div class="stat-value">{a2['bind_probability']:.1%}</div></div>
          <div class="stat-block"><div class="stat-label">Conversion band</div><div class="stat-value">{a2['conversion_band']}</div></div>
          <div class="why-box">
            <div class="why-label">Data caveat</div>
            <div class="why-text">This dataset's conversion outcome showed no measurable link to any input feature in testing. Treat this score as a pipeline demo, not a validated prediction, until retrained on data with real signal.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    with c3:
        st.markdown(f"""
        <div class="agent-card tier-hybrid">
          <span class="tag tag-hybrid">HYBRID</span>
          <div class="agent-name">Premium Advisor</div>
          <div class="stat-block"><div class="stat-label">Recommended action</div><div class="stat-value">{humanize_action(a3['recommended_action'])}</div></div>
          <div class="stat-block"><div class="stat-label">Original premium</div><div class="stat-value">${a3['original_premium']:,.2f}</div></div>
          <div class="stat-block"><div class="stat-label">Adjusted premium</div><div class="stat-value">${a3['adjusted_premium']:,.2f}</div></div>
          <div class="why-box">
            <div class="why-label">Why</div>
            <div class="why-text">{a3['justification']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    with c4:
        tier_class, tag_class, tag_label = ROUTE_STYLE.get(a4["route"], ("", "", "RULE-BASED"))
        reason_items = "".join(f"<li>{humanize_reason(r)}</li>" for r in a4["reasons"])
        st.markdown(f"""
        <div class="agent-card {tier_class}">
          <span class="tag {tag_class}">{tag_label}</span>
          <div class="agent-name">Decision Router</div>
          <div class="stat-block"><div class="stat-label">Route</div><div class="stat-value">{a4['route']}</div></div>
          <div class="stat-block"><div class="stat-label">Why this route</div>
            <ul class="reason-list">{reason_items}</ul>
          </div>
          <div class="why-box">
            <div class="why-label">Guideline reference</div>
            <div class="why-text">{a4['justification']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    with st.expander("Raw pipeline output (JSON)"):
        st.json(result)
else:
    st.info("Pick a quote in the sidebar, then click **Run through pipeline**.")