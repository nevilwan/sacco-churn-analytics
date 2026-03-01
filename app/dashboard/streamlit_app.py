"""
app/dashboard/streamlit_app.py
───────────────────────────────
Streamlit dashboard for the SACCO Retention Analysis System.

Displays:
  - Experiment overview and status
  - KPI results with visualizations
  - Statistical test results
  - Sample size calculator
  - Exportable reports

Run: streamlit run app/dashboard/streamlit_app.py
"""
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.analysis.statistical_tests import (
    simulate_experiment_data,
    test_binary_kpi,
    test_continuous_kpi,
    apply_multiple_comparison_correction,
)
from app.experiments.ab_engine import SampleSizeCalculator

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SACCO Retention Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: bold;
        color: #0D3B6E; margin-bottom: 0.2rem;
    }
    .sub-header { color: #555; margin-bottom: 1.5rem; font-size: 0.9rem; }
    .metric-card {
        background: #EEF5FC; border-radius: 8px;
        padding: 1rem; border-left: 4px solid #1A5EA8; color: #0D3B6E;
    }
    .guardrail-pass { border-left: 4px solid #1A6B3C; background: #D4EDE0; color: #0D3B6E; }
    .guardrail-fail { border-left: 4px solid #B22222; background: #FDECEA; color: #0D3B6E; }
    .warning-box {
        background: #FFF0DC; border-left: 4px solid #B85C00;
        padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0; color: #5C2E00;
    }
    .compliance-note {
        background: #D6E8F7; border-left: 4px solid #0D3B6E;
        padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;
        font-size: 0.85rem; color: #0D3B6E;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.markdown("## 🏦 SACCO Analytics")
st.sidebar.markdown("**Experimental Retention System**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Experiment Overview",
        "🧪 Run A/B Analysis",
        "📐 Sample Size Calculator",
        "📈 KPI Dashboard",
        "📋 Compliance & Audit",
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    "<div class='compliance-note'>⚖️ <strong>Regulatory Note</strong><br>"
    "All results show aggregate data only. "
    "Minimum cohort size: 30 members.<br><br>"
    "Compliant with: SASRA, CBK, Kenya Data Protection Act.</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: Experiment Overview
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Experiment Overview":
    st.markdown("<div class='main-header'>📊 Experiment Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Prototype Mode — Using simulated data. Connect to API for live data.</div>", unsafe_allow_html=True)

    # Simulated experiment registry
    experiments_data = [
        {
            "ID": "EXP-001", "Name": "Mobile Savings Reminders",
            "Status": "🟢 Running", "Primary KPI": "90d Retention",
            "Start": "2024-09-01", "End": "2024-12-01",
            "Control N": 183, "Treatment N": 187,
            "Guardrail": "✅ Passing",
        },
        {
            "ID": "EXP-002", "Name": "Dividend Visibility Dashboard",
            "Status": "⏳ Approved", "Primary KPI": "Share Capital Growth",
            "Start": "2024-10-15", "End": "2025-01-15",
            "Control N": "-", "Treatment N": "-",
            "Guardrail": "-",
        },
        {
            "ID": "EXP-003", "Name": "Loan Repeat Nudge via USSD",
            "Status": "✅ Completed", "Primary KPI": "Loan Repeat Rate",
            "Start": "2024-06-01", "End": "2024-09-01",
            "Control N": 210, "Treatment N": 214,
            "Guardrail": "✅ Passed",
        },
    ]

    df = pd.DataFrame(experiments_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", "3", "+1 this quarter")
    with col2:
        st.metric("Currently Running", "1", "Max allowed: 3")
    with col3:
        st.metric("Members Enrolled", "794", "Across all experiments")
    with col4:
        st.metric("Guardrail Violations", "0", "✅ Clean")

    st.divider()
    st.markdown("### 🔒 Concurrent Experiment Limit")
    progress_val = 1 / 3
    st.progress(progress_val, text=f"1 of 3 maximum concurrent experiments in use")

    st.markdown(
        "<div class='compliance-note'>📌 <strong>SASRA Compliance</strong>: "
        "Maximum 3 concurrent experiments enforced. Members are enrolled in only one "
        "experiment at a time. All experiment configurations require Data Steward approval "
        "before enrollment.</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: Run A/B Analysis
# ═══════════════════════════════════════════════════════════════════
elif page == "🧪 Run A/B Analysis":
    st.markdown("<div class='main-header'>🧪 A/B Experiment Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Statistical analysis of experiment results with regulatory-grade rigour.</div>", unsafe_allow_html=True)

    st.markdown("### Experiment Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        n_control = st.number_input("Control Group Size (N)", 50, 5000, 185, 10)
        n_treatment = st.number_input("Treatment Group Size (N)", 50, 5000, 190, 10)

    with col2:
        control_rate = st.slider("Control Retention Rate (baseline)", 0.30, 0.90, 0.62, 0.01,
                                 format="%.2f")
        true_effect = st.slider("Simulated True Effect (pp)", -0.10, 0.20, 0.06, 0.01,
                                format="%.2f")

    with col3:
        alpha = st.select_slider("Significance Level (α)", [0.001, 0.01, 0.05, 0.10], value=0.05)
        window_days = st.selectbox("Analysis Window", [30, 60, 90], index=2)
        seed = st.number_input("Random Seed", 1, 9999, 42)

    if st.button("▶️ Run Analysis", type="primary"):
        # Generate simulated data
        control_df, treatment_df = simulate_experiment_data(
            n_control=n_control,
            n_treatment=n_treatment,
            control_retention_rate=control_rate,
            treatment_effect=true_effect,
            seed=seed,
        )

        # Primary KPI: retention
        c_succ = int(control_df["retained_90d"].sum())
        t_succ = int(treatment_df["retained_90d"].sum())

        result = test_binary_kpi(
            control_successes=c_succ,
            control_total=n_control,
            treatment_successes=t_succ,
            treatment_total=n_treatment,
            alpha=alpha,
            kpi_name=f"retention_{window_days}d",
        )

        # ── Result display ────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Results")

        # Verdict banner
        if result.test_result.null_rejected and result.absolute_effect > 0:
            st.success(f"✅ **SIGNIFICANT IMPROVEMENT DETECTED** — Feature shows statistically significant lift in {window_days}-day retention.")
        elif result.test_result.null_rejected and result.absolute_effect < 0:
            st.error(f"❌ **SIGNIFICANT HARM DETECTED** — Feature significantly reduced retention. Do NOT launch.")
        else:
            st.warning(f"⚠️ **NO SIGNIFICANT DIFFERENCE** — Cannot conclude the feature improves retention at α={alpha}.")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Control Rate", f"{result.control_rate:.1%}",
                    f"n={result.control_n}")
        col2.metric("Treatment Rate", f"{result.treatment_rate:.1%}",
                    f"n={result.treatment_n}")
        col3.metric("Absolute Effect", f"{result.absolute_effect:+.1%}",
                    "Treatment vs Control")
        col4.metric("p-value", f"{result.test_result.p_value:.4f}",
                    f"α={alpha} threshold")

        # Confidence interval chart
        st.markdown("#### 95% Confidence Interval on Effect Size")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[result.test_result.ci_lower, result.test_result.ci_upper],
            y=["Effect"], mode="lines",
            line=dict(color="#1A5EA8", width=4),
            name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=[result.absolute_effect], y=["Effect"], mode="markers",
            marker=dict(color="#0D3B6E", size=14, symbol="diamond"),
            name="Point Estimate",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#B22222",
                      annotation_text="No effect", annotation_position="top right")
        fig.update_layout(
            height=200, showlegend=True,
            xaxis_title="Absolute Difference in Retention Rate",
            xaxis_tickformat=".1%",
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Retention comparison bar chart
        st.markdown("#### Retention Rate by Group")
        fig2 = px.bar(
            x=["Control", "Treatment"],
            y=[result.control_rate, result.treatment_rate],
            color=["Control", "Treatment"],
            color_discrete_map={"Control": "#7FB3D3", "Treatment": "#0D3B6E"},
            text=[f"{result.control_rate:.1%}", f"{result.treatment_rate:.1%}"],
            labels={"x": "Group", "y": "Retention Rate"},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            height=350, showlegend=False,
            yaxis_tickformat=".0%",
            yaxis_range=[0, min(1.0, max(result.treatment_rate, result.control_rate) * 1.2)],
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed stats table
        st.markdown("#### Statistical Test Details")
        stats_data = {
            "Parameter": ["Test Used", "Test Statistic", "p-value", "α Threshold",
                          "95% CI Lower", "95% CI Upper", "Cohen's h", "Null Rejected"],
            "Value": [
                result.test_result.test_name,
                f"{result.test_result.statistic:.4f}",
                f"{result.test_result.p_value:.6f}",
                f"{result.test_result.alpha}",
                f"{result.test_result.ci_lower:.4f}",
                f"{result.test_result.ci_upper:.4f}",
                f"{result.test_result.effect_size:.4f}",
                "✅ Yes" if result.test_result.null_rejected else "❌ No",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Warnings
        if result.test_result.warnings:
            for w in result.test_result.warnings:
                st.markdown(f"<div class='warning-box'>⚠️ {w}</div>", unsafe_allow_html=True)

        # Guardrail check (simulated)
        st.divider()
        st.markdown("### 🛡️ Guardrail KPI Check — On-Time Repayment")
        guard_control = float(np.random.default_rng(seed + 1).binomial(n_control, 0.80) / n_control)
        guard_treatment = float(np.random.default_rng(seed + 2).binomial(n_treatment, 0.79) / n_treatment)
        drop = guard_control - guard_treatment

        guard_result = test_binary_kpi(
            control_successes=int(guard_control * n_control),
            control_total=n_control,
            treatment_successes=int(guard_treatment * n_treatment),
            treatment_total=n_treatment,
            alpha=0.05,
            kpi_name="on_time_repayment",
        )

        colg1, colg2, colg3 = st.columns(3)
        colg1.metric("Control Repayment Rate", f"{guard_control:.1%}")
        colg2.metric("Treatment Repayment Rate", f"{guard_treatment:.1%}",
                     f"{guard_treatment - guard_control:+.1%}")
        colg3.metric("Rate Drop", f"{drop:.1%}",
                     "⚠️ Watch" if drop > 0.01 else "✅ OK")

        if drop > 0.02:
            st.error("🚨 **GUARDRAIL VIOLATION**: Repayment rate drop exceeds 2pp threshold. Experiment would be AUTO-SUSPENDED.")
        elif drop > 0.01:
            st.warning("⚠️ **GUARDRAIL WARNING**: Repayment rate shows downward trend. Monitor closely.")
        else:
            st.success("✅ **GUARDRAIL PASSING**: On-time repayment rate not significantly harmed.")

        # Export report
        st.divider()
        st.markdown("### 📥 Export Report")
        report_data = {
            "experiment_analysis": {
                "generated_at": str(date.today()),
                "window_days": window_days,
                "primary_kpi": f"retention_{window_days}d",
                "control_n": result.control_n,
                "treatment_n": result.treatment_n,
                "control_rate": result.control_rate,
                "treatment_rate": result.treatment_rate,
                "absolute_effect": result.absolute_effect,
                "p_value": result.test_result.p_value,
                "ci_95": [result.test_result.ci_lower, result.test_result.ci_upper],
                "null_rejected": bool(result.test_result.null_rejected),
                "test_used": result.test_result.test_name,
                "conclusion": result.test_result.conclusion,
                "warnings": result.test_result.warnings,
            }
        }
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download JSON Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"experiment_analysis_{date.today()}.json",
                mime="application/json",
            )
        with col_dl2:
            csv_df = pd.DataFrame([{
                "KPI": f"retention_{window_days}d",
                "Control N": result.control_n,
                "Treatment N": result.treatment_n,
                "Control Rate": f"{result.control_rate:.4f}",
                "Treatment Rate": f"{result.treatment_rate:.4f}",
                "Absolute Effect": f"{result.absolute_effect:.4f}",
                "p-value": f"{result.test_result.p_value:.6f}",
                "CI Lower": f"{result.test_result.ci_lower:.4f}",
                "CI Upper": f"{result.test_result.ci_upper:.4f}",
                "Test": result.test_result.test_name,
                "Null Rejected": result.test_result.null_rejected,
            }])
            st.download_button(
                "⬇️ Download CSV Report",
                data=csv_df.to_csv(index=False),
                file_name=f"experiment_analysis_{date.today()}.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: Sample Size Calculator
# ═══════════════════════════════════════════════════════════════════
elif page == "📐 Sample Size Calculator":
    st.markdown("<div class='main-header'>📐 Sample Size Calculator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Estimate required members before designing your experiment.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Parameters")
        baseline = st.slider("Baseline Retention Rate", 0.30, 0.90, 0.62, 0.01, format="%.2f")
        mde = st.slider("Minimum Detectable Effect (pp)", 0.02, 0.20, 0.06, 0.01, format="%.2f")
        alpha = st.select_slider("Significance Level (α)", [0.001, 0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Target Power (1-β)", [0.70, 0.80, 0.90], value=0.80)
        eligible_count = st.number_input("Your Eligible Member Count", 100, 50000, 1200, 100)

    with col2:
        calc = SampleSizeCalculator()

        try:
            result = calc.calculate(
                baseline_rate=baseline,
                minimum_detectable_effect=mde,
                alpha=alpha,
                power=power,
            )
            feasibility = calc.check_feasibility(result.n_per_arm, eligible_count)

            st.markdown("### Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Per Arm Required", f"{result.n_per_arm:,}")
            m2.metric("Total Required", f"{result.total_n:,}")
            m3.metric("Treatment Rate", f"{result.treatment_rate:.1%}")

            # Feasibility indicator
            if feasibility["feasible"]:
                st.success(f"✅ **FEASIBLE** — You have {eligible_count:,} eligible members and need {result.total_n:,}.")
            else:
                st.error(f"❌ **UNDERPOWERED** — You have {eligible_count:,} members but need {result.total_n:,}.")
                st.markdown(f"<div class='warning-box'>{feasibility['recommendation']}</div>", unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Invalid parameters: {e}")
            st.stop()

    # Power curve
    st.divider()
    st.markdown("### Power Curve — Effect Size vs Required Sample Size")

    mde_range = np.arange(0.02, 0.21, 0.01)
    ns = []
    for m in mde_range:
        try:
            r = calc.calculate(baseline, m, alpha, power)
            ns.append(r.n_per_arm)
        except:
            ns.append(None)

    fig = px.line(
        x=mde_range, y=ns,
        labels={"x": "Minimum Detectable Effect (percentage points)", "y": "Required N per arm"},
        title=f"Baseline={baseline:.0%}, α={alpha}, Power={power:.0%}",
    )
    fig.add_vline(x=mde, line_dash="dash", line_color="#B22222",
                  annotation_text=f"Current MDE={mde:.0%}")
    fig.add_hline(y=eligible_count // 2, line_dash="dot", line_color="#1A6B3C",
                  annotation_text="Available per arm")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Notes
    for note in result.notes:
        st.markdown(f"<div class='warning-box'>📌 {note}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: KPI Dashboard
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 KPI Dashboard":
    st.markdown("<div class='main-header'>📈 KPI Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Retention and engagement KPIs across member cohorts. All data aggregated (min cohort: 30).</div>", unsafe_allow_html=True)

    # Simulated KPI data
    rng = np.random.default_rng(100)

    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    retention_30d = [0.71, 0.73, 0.74, 0.72, 0.75, 0.76]
    retention_90d = [0.61, 0.63, 0.64, 0.62, 0.65, 0.67]
    repayment_rate = [0.81, 0.82, 0.80, 0.83, 0.84, 0.85]
    savings_consistency = [0.58, 0.60, 0.61, 0.63, 0.64, 0.66]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("30-Day Retention", "76.0%", "+1.0pp vs last month")
    col2.metric("90-Day Retention", "67.0%", "+2.0pp vs last month")
    col3.metric("On-Time Repayment", "85.0%", "+1.0pp vs last month")
    col4.metric("Savings Consistency", "66.0%", "+2.0pp vs last month")

    st.divider()

    # Trend charts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=retention_30d, mode="lines+markers",
                             name="30-Day Retention", line=dict(color="#1A5EA8", width=2)))
    fig.add_trace(go.Scatter(x=months, y=retention_90d, mode="lines+markers",
                             name="90-Day Retention", line=dict(color="#0D3B6E", width=2)))
    fig.add_trace(go.Scatter(x=months, y=repayment_rate, mode="lines+markers",
                             name="On-Time Repayment", line=dict(color="#1A6B3C", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=months, y=savings_consistency, mode="lines+markers",
                             name="Savings Consistency", line=dict(color="#C8920A", width=2, dash="dot")))
    fig.update_layout(
        title="Key Retention KPIs — 6-Month Trend",
        yaxis_tickformat=".0%", yaxis_range=[0.4, 1.0],
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Zone breakdown
    st.markdown("### Retention by Geographic Zone")
    zones = ["Urban Nairobi", "Urban Other", "Peri-Urban", "Rural"]
    rates_30d = [0.82, 0.78, 0.72, 0.64]
    rates_90d = [0.74, 0.69, 0.62, 0.54]

    fig2 = go.Figure(data=[
        go.Bar(name="30-Day", x=zones, y=rates_30d,
               marker_color="#4A7EC7", text=[f"{r:.0%}" for r in rates_30d], textposition="outside"),
        go.Bar(name="90-Day", x=zones, y=rates_90d,
               marker_color="#0D3B6E", text=[f"{r:.0%}" for r in rates_90d], textposition="outside"),
    ])
    fig2.update_layout(
        barmode="group", yaxis_tickformat=".0%",
        yaxis_range=[0, 1.0], height=380,
        title="⚠️ Rural retention is significantly lower — design experiments specifically for this segment",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<div class='compliance-note'>📊 <strong>Fairness Note</strong>: "
        "Rural members show 20pp lower retention than urban Nairobi members. "
        "Any experiment showing aggregate improvement must be evaluated for "
        "differential impact on rural segments before launch.</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 5: Compliance & Audit
# ═══════════════════════════════════════════════════════════════════
elif page == "📋 Compliance & Audit":
    st.markdown("<div class='main-header'>📋 Compliance & Audit Log</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Board-level experiment governance and regulatory audit trail.</div>", unsafe_allow_html=True)

    # Simulated audit log
    audit_entries = [
        {"Timestamp": "2024-10-01 08:32:11", "Actor": "steward_****_001", "Role": "data_steward",
         "Action": "EXPERIMENT_APPROVED", "Resource": "EXP-001", "Status": "✅"},
        {"Timestamp": "2024-10-01 08:35:44", "Actor": "designer_****_001", "Role": "experiment_designer",
         "Action": "EXPERIMENT_ENROLLMENT_COMPLETED", "Resource": "EXP-001", "Status": "✅"},
        {"Timestamp": "2024-10-08 10:12:03", "Actor": "analyst_****_001", "Role": "data_analyst",
         "Action": "KPI_RESULTS_QUERIED", "Resource": "EXP-001", "Status": "✅"},
        {"Timestamp": "2024-10-15 14:22:55", "Actor": "steward_****_001", "Role": "data_steward",
         "Action": "EXPERIMENT_APPROVED", "Resource": "EXP-002", "Status": "✅"},
        {"Timestamp": "2024-09-01 09:00:00", "Actor": "analyst_****_001", "Role": "data_analyst",
         "Action": "EXPERIMENT_RESULT_EXPORTED", "Resource": "EXP-003", "Status": "⚠️ Review"},
    ]

    st.markdown("### Recent Audit Events")
    audit_df = pd.DataFrame(audit_entries)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### ✅ Compliance Checklist — Pre-Deployment")

    checklist = {
        "DPIA completed and filed with DPO": True,
        "Privacy Notice updated to reflect experiment processing": True,
        "Board resolution approving experiment programme": True,
        "AGM or member notice issued": False,
        "SASRA informal guidance sought for experiment framework": False,
        "Tokenization master key stored in HSM (not config file)": True,
        "A/A validation passed before live experiment": True,
        "Credit risk officer sign-off on guardrail KPIs": True,
        "Minimum experiment duration ≥ 60 days enforced": True,
        "Rural member fairness analysis documented": False,
        "Data retention policy aligned with 7-year requirement": True,
        "Incident response runbook tested": False,
    }

    col1, col2 = st.columns(2)
    items = list(checklist.items())
    half = len(items) // 2

    for col, chunk in [(col1, items[:half]), (col2, items[half:])]:
        with col:
            for item, done in chunk:
                icon = "✅" if done else "❌"
                color = "#1A6B3C" if done else "#B22222"
                st.markdown(
                    f"<div style='color:{color}; padding: 4px 0;'>{icon} {item}</div>",
                    unsafe_allow_html=True,
                )

    total = len(checklist)
    done_count = sum(checklist.values())
    pct = done_count / total
    st.divider()
    st.progress(pct, text=f"Compliance readiness: {done_count}/{total} items complete ({pct:.0%})")

    if pct < 1.0:
        st.markdown(
            "<div class='warning-box'>⚠️ <strong>Not production-ready</strong>: "
            "Complete all checklist items before processing live member data. "
            "Outstanding items require board, regulatory, or legal action.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("🎉 All compliance items complete. System is production-ready.")
