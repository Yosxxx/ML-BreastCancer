# app.py – Streamlit breast‑cancer predictor with rich history & charts

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Breast‑Cancer Predictor",
    page_icon="🔬",
    layout="centered",
)

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    # each record: dict(Timestamp, Patient, Diagnosis, Probability, Features)
    st.session_state["history"] = []

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return load("models/logreg_pipeline.joblib")

model = load_model()
feature_names = list(model.feature_names_in_)

# ---------- HEADER ----------
st.title("🔬 Breast‑Cancer Predictor")
st.caption("Estimate malignancy probability from tumor measurements.")

# ---------- TABS LAYOUT ----------
pred_tab, hist_tab = st.tabs(["🔍 Prediction", "📜 History"])

# ====================================================================
# PREDICTION TAB
# ====================================================================
with pred_tab:
    patient_name: str = st.text_input(
        "Patients Name",
        placeholder="Enter patient name here",
        key="patient_name",
    )

    st.subheader("Enter tumor prediction")

    cols = st.columns(3)
    user_input: list[float] = []

    for i, feat in enumerate(feature_names):
        label = feat.replace("_", " ").title()
        with cols[i % 3]:
            value = st.number_input(
                label,
                min_value=0.0,
                format="%.4f",
                key=f"val_{feat}"
            )
        user_input.append(value)

    st.divider()
    if st.button("Predict", type="primary", key="predict_btn"):
        if not patient_name.strip():
            st.warning("Please enter the patient's name above.")
        elif any(v == 0.0 for v in user_input):
            st.warning("All feature values should be filled.")
        else:
            X = np.asarray(user_input).reshape(1, -1)
            pred = model.predict(X)[0]
            prob = float(model.predict_proba(X)[0, 1])

            diagnosis = "Malignant" if pred == 1 else "Benign"
            st.success(f"**{patient_name}** • {diagnosis}")
            st.caption(f"Probability of malignancy: {prob:.2%}")

            # --- save record in history ---
            st.session_state.history.append(
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Patient": patient_name,
                    "Diagnosis": diagnosis,
                    "Probability": prob,
                    "Features": user_input,
                }
            )

# ====================================================================
# HISTORY TAB
# ====================================================================
with hist_tab:
    st.subheader("Prediction History")

    if st.session_state.history:
        # build DataFrame for overview (exclude feature vectors)
        hist_df = pd.DataFrame(
            [
                {
                    "Timestamp": rec["Timestamp"],
                    "Patient": rec["Patient"],
                    "Diagnosis": rec["Diagnosis"],
                    "Probability (%)": f"{rec['Probability']*100:.2f}"
                }
                for rec in st.session_state.history
            ]
        )

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        st.markdown("### Inspect a previous prediction")
        selected_idx = st.selectbox(
            "Select by timestamp", options=list(range(len(hist_df)))[::-1],
            format_func=lambda i: f"{hist_df.loc[i, 'Timestamp']} • {hist_df.loc[i, 'Patient']}"
        )

        record = st.session_state.history[selected_idx]
        st.write(f"**Patient:** {record['Patient']}")
        st.write(f"**Diagnosis:** {record['Diagnosis']}  |  **Probability:** {record['Probability']:.2%}")

        # --- bar chart of feature values vs dataset mean (optional)
        with st.expander("Show feature values vs dataset mean"):
            feat_vals = np.array(record["Features"])
            mean_vals = np.mean([r["Features"] for r in st.session_state.history], axis=0)
            df_chart = pd.DataFrame({
                "Feature": feature_names,
                "This Record": feat_vals,
                "History Mean": mean_vals,
            })
            df_chart = df_chart.set_index("Feature")

            fig, ax = plt.subplots(figsize=(6, 4))
            df_chart[["This Record", "History Mean"]].plot.bar(ax=ax)
            ax.set_ylabel("Value")
            ax.set_title("Feature comparison")
            ax.tick_params(axis='x', labelrotation=90)
            st.pyplot(fig)

        # --- probability trend line ---
        st.markdown("### Probability trend over time")
        trend_df = pd.DataFrame({
            "Timestamp": pd.to_datetime(hist_df["Timestamp"]),
            "Probability": [rec["Probability"] for rec in st.session_state.history]
        }).set_index("Timestamp")
        st.line_chart(trend_df)

        # --- clear history button ---
        if st.button("🗑️ Clear history", key="clear_hist"):
            st.session_state.history.clear()
            st.success("History cleared ✅")
            st.experimental_rerun()
    else:
        st.info("No predictions yet. Run one in the *Prediction* tab ✨")

# ---------- FOOTER / EXTRA CSS ----------
st.markdown(
    """
    <style>
    .stNumberInput > div > div > input {text-align: right;}
    @media (max-width: 600px) {
        .stColumn {flex: 1 1 100% !important; max-width: 100% !important;}
    }
    div[data-testid="stTextInput"] > div > input {width: 70vw !important;}
    </style>
    """,
    unsafe_allow_html=True,
)
