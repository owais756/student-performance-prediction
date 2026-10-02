"""
Student Performance Prediction — Streamlit Application
Pages:
  1. 🏠 Home
  2. 🔮 Predict Score
  3. 📊 EDA & Analytics
  4. 🤖 Model Insights
"""
import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Path resolution ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "StudentPerformanceFactors.xlsx")
sys.path.insert(0, SRC_DIR)

from preprocess import (
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL,
    encode_and_scale, load_raw_data
)


# ── Auto-train if models are missing (e.g. first run on Streamlit Cloud) ──────
if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    import subprocess
    st.info("First run detected — training models. This takes about 60 seconds...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "train.py")], check=True)
    st.rerun()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem;}
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .value {font-size: 1.9rem; font-weight: 700; color: #1e3a8a;}
    .metric-card .label {font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;}
    .predict-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 2px solid #3b82f6; border-radius: 16px;
        padding: 1.5rem 2rem; text-align: center; margin: 1rem 0;
    }
    .predict-score {font-size: 3.5rem; font-weight: 800; color: #1e3a8a;}
    .grade-badge {
        display: inline-block; padding: 0.3rem 1rem;
        border-radius: 999px; font-weight: 700;
        font-size: 1rem; margin-top: 0.5rem;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600; color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    encoders   = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        summary = json.load(f)
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return best_model, all_models, encoders, scaler, summary, feature_names

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return load_raw_data()

best_model, all_models, encoders, scaler, summary, feature_names = load_artifacts()
df_raw = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎓 Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔮 Predict Score", "📊 EDA & Analytics", "🤖 Model Insights"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df_raw):,} students · 19 features")
st.sidebar.markdown(f"**Best Model:** {summary['best_model']}")
best_r2 = summary['metrics'][summary['best_model']]['R2']
st.sidebar.markdown(f"**R² Score:** {best_r2:.4f}")

# ── Helper ────────────────────────────────────────────────────────────────────
def score_to_grade(score):
    if score >= 90: return "A+", "#059669"
    if score >= 80: return "A",  "#10b981"
    if score >= 70: return "B",  "#3b82f6"
    if score >= 60: return "C",  "#f59e0b"
    if score >= 50: return "D",  "#f97316"
    return "F", "#ef4444"


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎓 Student Performance Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict exam scores using machine learning on 6,607 student profiles across 19 features.</p>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(df_raw):,}</div><div class="label">Students</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">19</div><div class="label">Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="value">{best_r2:.2%}</div><div class="label">Best R² Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="value">{df_raw[TARGET_COL].mean():.1f}</div><div class="label">Avg Exam Score</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📈 Exam Score Distribution")
        fig = px.histogram(
            df_raw, x=TARGET_COL, nbins=40,
            color_discrete_sequence=["#3b82f6"],
            labels={TARGET_COL: "Exam Score", "count": "# Students"},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=20, r=20, t=30, b=20), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📋 Feature Overview")
        feature_info = {
            "Numerical": ", ".join(NUMERICAL_COLS),
            "Categorical": ", ".join(CATEGORICAL_COLS),
            "Target": TARGET_COL,
        }
        for k, v in feature_info.items():
            st.markdown(f"**{k}:** {v}")
        st.markdown("---")
        st.subheader("🏆 Models Trained")
        metrics_df = pd.DataFrame(summary["metrics"]).T.reset_index()
        metrics_df.columns = ["Model", "MAE", "RMSE", "R2"]
        metrics_df = metrics_df.sort_values("R2", ascending=False)
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Sample Data")
    st.dataframe(df_raw.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT SCORE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Score":
    st.markdown('<p class="main-header">🔮 Predict Exam Score</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fill in student details below to get an AI-powered exam score prediction.</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("📚 Academic & Study Habits")
        c1, c2, c3 = st.columns(3)
        hours_studied     = c1.slider("Hours Studied (per week)", 1, 44, 20)
        attendance        = c2.slider("Attendance (%)", 60, 100, 85)
        previous_scores   = c3.slider("Previous Scores", 50, 100, 75)

        c4, c5, c6 = st.columns(3)
        tutoring_sessions = c4.slider("Tutoring Sessions (per month)", 0, 8, 1)
        sleep_hours       = c5.slider("Sleep Hours (per night)", 4, 10, 7)
        physical_activity = c6.slider("Physical Activity (hrs/week)", 0, 6, 3)

        st.subheader("🏫 School & Environment")
        c7, c8, c9 = st.columns(3)
        parental_involvement   = c7.selectbox("Parental Involvement",    ["Low", "Medium", "High"])
        access_to_resources    = c8.selectbox("Access to Resources",     ["Low", "Medium", "High"])
        motivation_level       = c9.selectbox("Motivation Level",        ["Low", "Medium", "High"])

        c10, c11, c12 = st.columns(3)
        family_income          = c10.selectbox("Family Income",          ["Low", "Medium", "High"])
        teacher_quality        = c11.selectbox("Teacher Quality",        ["Low", "Medium", "High"])
        school_type            = c12.selectbox("School Type",            ["Public", "Private"])

        st.subheader("👤 Personal & Social")
        c13, c14, c15 = st.columns(3)
        peer_influence         = c13.selectbox("Peer Influence",         ["Positive", "Neutral", "Negative"])
        parental_education     = c14.selectbox("Parental Education",     ["High School", "College", "Postgraduate"])
        distance_from_home     = c15.selectbox("Distance from Home",     ["Near", "Moderate", "Far"])

        c16, c17, c18 = st.columns(3)
        internet_access        = c16.selectbox("Internet Access",        ["Yes", "No"])
        extracurricular        = c17.selectbox("Extracurricular Activities", ["Yes", "No"])
        learning_disabilities  = c18.selectbox("Learning Disabilities",  ["No", "Yes"])
        gender                 = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("🚀 Predict Exam Score", use_container_width=True, type="primary")

    if submitted:
        input_dict = {
            "Hours_Studied":             hours_studied,
            "Attendance":                attendance,
            "Sleep_Hours":               sleep_hours,
            "Previous_Scores":           previous_scores,
            "Tutoring_Sessions":         tutoring_sessions,
            "Physical_Activity":         physical_activity,
            "Parental_Involvement":      parental_involvement,
            "Access_to_Resources":       access_to_resources,
            "Extracurricular_Activities": extracurricular,
            "Motivation_Level":          motivation_level,
            "Internet_Access":           internet_access,
            "Family_Income":             family_income,
            "Teacher_Quality":           teacher_quality,
            "School_Type":               school_type,
            "Peer_Influence":            peer_influence,
            "Learning_Disabilities":     learning_disabilities,
            "Parental_Education_Level":  parental_education,
            "Distance_from_Home":        distance_from_home,
            "Gender":                    gender,
        }

        # Encode
        input_df = pd.DataFrame([input_dict])
        for col in CATEGORICAL_COLS:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))
        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])
        X_input = input_df[feature_names]

        predicted_score = float(best_model.predict(X_input)[0])
        predicted_score = np.clip(predicted_score, 0, 100)
        grade, color = score_to_grade(predicted_score)

        col_pred, col_info = st.columns([1, 1])
        with col_pred:
            st.markdown(f"""
            <div class="predict-box">
                <div style="font-size:1.1rem;color:#374151;font-weight:600;">Predicted Exam Score</div>
                <div class="predict-score">{predicted_score:.1f}</div>
                <span class="grade-badge" style="background:{color};color:white;">Grade {grade}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            st.subheader("📊 Score Context")
            avg = df_raw[TARGET_COL].mean()
            pct = (df_raw[TARGET_COL] < predicted_score).mean() * 100
            st.metric("Your Score", f"{predicted_score:.1f}")
            st.metric("Dataset Average", f"{avg:.1f}", delta=f"{predicted_score - avg:+.1f}")
            st.metric("Percentile Rank", f"{pct:.0f}th percentile")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Exam Score Gauge"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0,  50], "color": "#fee2e2"},
                    {"range": [50, 60], "color": "#ffedd5"},
                    {"range": [60, 70], "color": "#fef3c7"},
                    {"range": [70, 80], "color": "#dbeafe"},
                    {"range": [80, 90], "color": "#d1fae5"},
                    {"range": [90, 100],"color": "#a7f3d0"},
                ],
                "threshold": {"line": {"color": "#1e3a8a", "width": 3}, "value": avg},
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # All-models comparison
        st.subheader("🤖 All Model Predictions")
        model_preds = {}
        for name, mdl in all_models.items():
            p = float(mdl.predict(X_input)[0])
            p = np.clip(p, 0, 100)
            model_preds[name] = round(p, 2)

        pred_df = pd.DataFrame(
            {"Model": list(model_preds.keys()), "Predicted Score": list(model_preds.values())}
        ).sort_values("Predicted Score", ascending=False)

        fig_bar = px.bar(
            pred_df, x="Predicted Score", y="Model", orientation="h",
            color="Predicted Score", color_continuous_scale="Blues",
            text="Predicted Score",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=260, margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA & ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Analytics":
    st.markdown('<p class="main-header">📊 EDA & Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore patterns and relationships in the student dataset.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Category Analysis", "Scatter Explorer"])

    # ── Tab 1: Distributions ──
    with tab1:
        st.subheader("Exam Score Distribution")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df_raw, x=TARGET_COL, nbins=40, color_discrete_sequence=["#3b82f6"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=350, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.box(df_raw, y=TARGET_COL, color_discrete_sequence=["#3b82f6"])
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=350, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Numerical Feature Distributions")
        num_feat = st.selectbox("Select feature", NUMERICAL_COLS, key="num_dist")
        col_c, col_d = st.columns(2)
        with col_c:
            fig3 = px.histogram(df_raw, x=num_feat, nbins=30, color_discrete_sequence=["#6366f1"])
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig3, use_container_width=True)
        with col_d:
            fig4 = px.scatter(df_raw, x=num_feat, y=TARGET_COL, opacity=0.4,
                              color_discrete_sequence=["#6366f1"],
                              trendline="lowess")
            fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2: Correlations ──
    with tab2:
        st.subheader("Correlation of Numerical Features with Exam Score")
        num_df = df_raw[NUMERICAL_COLS + [TARGET_COL]]
        corr = num_df.corr()[TARGET_COL].drop(TARGET_COL).sort_values()
        colors = ["#ef4444" if v < 0 else "#3b82f6" for v in corr.values]
        fig_corr = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=colors, text=[f"{v:.3f}" for v in corr.values],
            textposition="outside",
        ))
        fig_corr.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Pearson Correlation"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Full Numerical Correlation Matrix")
        corr_matrix = num_df.corr()
        fig_heat = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r", aspect="auto",
        )
        fig_heat.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Category Analysis ──
    with tab3:
        cat_feat = st.selectbox("Select categorical feature", CATEGORICAL_COLS, key="cat_feat")
        group_df = df_raw.groupby(cat_feat)[TARGET_COL].agg(["mean", "median", "count"]).reset_index()
        group_df.columns = [cat_feat, "Mean Score", "Median Score", "Count"]

        col_e, col_f = st.columns(2)
        with col_e:
            fig_cat = px.bar(
                group_df, x=cat_feat, y="Mean Score",
                color="Mean Score", color_continuous_scale="Blues",
                text="Mean Score",
            )
            fig_cat.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_cat.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_f:
            fig_box = px.box(df_raw, x=cat_feat, y=TARGET_COL,
                             color=cat_feat,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_box.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.dataframe(group_df.set_index(cat_feat), use_container_width=True)

    # ── Tab 4: Scatter Explorer ──
    with tab4:
        col_s1, col_s2, col_s3 = st.columns(3)
        x_feat = col_s1.selectbox("X-axis", NUMERICAL_COLS, index=0, key="sc_x")
        y_feat = col_s2.selectbox("Y-axis", NUMERICAL_COLS + [TARGET_COL], index=len(NUMERICAL_COLS), key="sc_y")
        color_by = col_s3.selectbox("Color by", CATEGORICAL_COLS, index=0, key="sc_c")

        fig_sc = px.scatter(
            df_raw, x=x_feat, y=y_feat, color=color_by,
            opacity=0.55, hover_data=[TARGET_COL],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sc.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=500, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.markdown('<p class="main-header">🤖 Model Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Model performance metrics, feature importance, and prediction analysis.</p>', unsafe_allow_html=True)

    tab_m1, tab_m2, tab_m3 = st.tabs(["Performance Comparison", "Feature Importance", "Prediction Analysis"])

    with tab_m1:
        st.subheader("📊 Model Comparison")
        metrics_df = pd.DataFrame(summary["metrics"]).T.reset_index()
        metrics_df.columns = ["Model", "MAE", "RMSE", "R2"]
        metrics_df = metrics_df.sort_values("R2", ascending=False)

        col_m1, col_m2, col_m3 = st.columns(3)
        for col, metric in zip([col_m1, col_m2, col_m3], ["R2", "MAE", "RMSE"]):
            with col:
                best_val = metrics_df.loc[metrics_df["R2"].idxmax(), metric]
                fig_m = px.bar(
                    metrics_df, x="Model", y=metric,
                    color=metric, color_continuous_scale="Blues" if metric=="R2" else "Reds_r",
                    text=metric,
                    title=metric,
                )
                fig_m.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_m.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=360, margin=dict(l=10, r=10, t=30, b=40),
                    coloraxis_showscale=False, showlegend=False,
                    xaxis_tickangle=-25,
                )
                st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")
        st.dataframe(
            metrics_df.set_index("Model").style.highlight_max(["R2"], color="#bbf7d0").highlight_min(["MAE", "RMSE"], color="#bbf7d0"),
            use_container_width=True,
        )

    with tab_m2:
        st.subheader(f"🌟 Feature Importance — {summary['best_model']}")
        if hasattr(best_model, "feature_importances_"):
            imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color="#3b82f6", text=[f"{v:.4f}" for v in imp.values],
                textposition="outside",
            ))
            fig_fi.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=550, margin=dict(l=20, r=60, t=10, b=20),
                xaxis_title="Importance Score",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(best_model, "coef_"):
            coef = pd.Series(best_model.coef_, index=feature_names).sort_values(ascending=True)
            fig_coef = go.Figure(go.Bar(
                x=coef.values, y=coef.index, orientation="h",
                marker_color=["#ef4444" if v < 0 else "#3b82f6" for v in coef.values],
            ))
            fig_coef.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=500, margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Coefficient",
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with tab_m3:
        st.subheader("📈 Actual vs Predicted (Test Set)")
        test_data_path = os.path.join(MODELS_DIR, "test_data.csv")
        test_df = pd.read_csv(test_data_path)
        y_test = test_df["Exam_Score"].values
        X_test = test_df.drop(columns=["Exam_Score"])
        y_pred = best_model.predict(X_test)

        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#3b82f6", opacity=0.4, size=5),
            name="Predictions",
        ))
        fig_ap.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines", line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect Prediction",
        ))
        fig_ap.update_layout(
            xaxis_title="Actual Score", yaxis_title="Predicted Score",
            plot_bgcolor="white", paper_bgcolor="white",
            height=450, margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig_ap, use_container_width=True)

        residuals = y_test - y_pred
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.subheader("Residual Distribution")
            fig_res = px.histogram(
                x=residuals, nbins=50,
                color_discrete_sequence=["#6366f1"],
                labels={"x": "Residual (Actual − Predicted)", "count": "Frequency"},
            )
            fig_res.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=330, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_res, use_container_width=True)
        with col_r2:
            st.subheader("Residuals vs Predicted")
            fig_res2 = go.Figure(go.Scatter(
                x=y_pred, y=residuals, mode="markers",
                marker=dict(color="#6366f1", opacity=0.4, size=5),
            ))
            fig_res2.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res2.update_layout(
                xaxis_title="Predicted Score", yaxis_title="Residual",
                plot_bgcolor="white", paper_bgcolor="white",
                height=330, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_res2, use_container_width=True)

        # Final metrics summary
        st.markdown("---")
        st.subheader("📋 Test Set Metrics")
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score",  f"{r2:.4f}")
        m2.metric("MAE",       f"{mae:.4f}")
        m3.metric("RMSE",      f"{rmse:.4f}")
