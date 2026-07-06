"""
app.py
------
Streamlit web application for the Student Placement Prediction project.

Run with:
    streamlit run app.py

Features:
    * Sidebar form for entering a single student's details
    * Instant prediction with confidence score & recommendation
    * Batch prediction via CSV upload, with a downloadable results file
    * Feature importance & model comparison charts (from training run)
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.predict import predict_placement, predict_batch

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "placement_model.pkl")

# ----------------------------------------------------------------------
# Minimal custom CSS for a more polished, dark-mode-friendly look
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.6rem;
    }
    .result-box {
        padding: 1.4rem 1.6rem;
        border-radius: 0.9rem;
        margin-top: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .placed-box {
        background: rgba(46, 204, 113, 0.12);
        border-color: rgba(46, 204, 113, 0.45);
    }
    .not-placed-box {
        background: rgba(231, 76, 60, 0.10);
        border-color: rgba(231, 76, 60, 0.4);
    }
    .footer {
        text-align: center;
        opacity: 0.6;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Placement Prediction</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether a student is likely to be '
    'placed, using academic performance and skill-related attributes.</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "⚠️ No trained model found. Please run `python src/train.py` "
        "first to train and save the model, then restart this app."
    )
    st.stop()

# ----------------------------------------------------------------------
# Sidebar: student details input
# ----------------------------------------------------------------------
st.sidebar.header("📋 Student Details")
st.sidebar.caption("Fill in the details below, then click **Predict**.")

cgpa = st.sidebar.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
iq = st.sidebar.slider("IQ", 70, 145, 100, 1)
prev_sem_percentage = st.sidebar.slider("Previous Semester Percentage", 40.0, 100.0, 70.0, 0.5)
communication_skills = st.sidebar.slider("Communication Skills (1-10)", 1.0, 10.0, 6.5, 0.1)
aptitude_score = st.sidebar.slider("Aptitude Score", 20.0, 100.0, 65.0, 0.5)
technical_skills_score = st.sidebar.slider("Technical Skills Score", 20.0, 100.0, 65.0, 0.5)
projects_completed = st.sidebar.number_input("Projects Completed", 0, 15, 2, 1)
internships = st.sidebar.number_input("Internships", 0, 10, 1, 1)
certifications = st.sidebar.number_input("Certifications", 0, 15, 1, 1)
attendance_percentage = st.sidebar.slider("Attendance Percentage", 40.0, 100.0, 82.0, 0.5)
backlogs = st.sidebar.number_input("Backlogs", 0, 15, 0, 1)

predict_clicked = st.sidebar.button("🔮 Predict Placement", use_container_width=True)

student_input = {
    "CGPA": cgpa,
    "IQ": iq,
    "Previous_Semester_Percentage": prev_sem_percentage,
    "Communication_Skills": communication_skills,
    "Aptitude_Score": aptitude_score,
    "Technical_Skills_Score": technical_skills_score,
    "Projects_Completed": projects_completed,
    "Internships": internships,
    "Certifications": certifications,
    "Attendance_Percentage": attendance_percentage,
    "Backlogs": backlogs,
}

# ----------------------------------------------------------------------
# Main content - tabs
# ----------------------------------------------------------------------
tab_predict, tab_batch, tab_insights = st.tabs(
    ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Insights"]
)

# ---------------------- Tab 1: Single Prediction -----------------------
with tab_predict:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Student Profile Summary")
        st.dataframe(
            pd.DataFrame(student_input.items(), columns=["Attribute", "Value"]),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.subheader("Prediction Result")

        if predict_clicked:
            try:
                result = predict_placement(student_input)
                is_placed = result["prediction"] == "Placed"
                box_class = "placed-box" if is_placed else "not-placed-box"
                icon = "✅" if is_placed else "❌"

                st.markdown(
                    f"""
                    <div class="result-box {box_class}">
                        <h3>{icon} Prediction: {result['prediction']}</h3>
                        <p style="font-size:1.1rem;"><b>Confidence:</b> {result['confidence']}%</p>
                        <p><b>Recommendation:</b> {result['recommendation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(int(result["confidence"]), 100))

                if is_placed:
                    st.success("Great! This profile shows strong placement potential.")
                else:
                    st.warning("This profile could be strengthened before placement season.")

            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
        else:
            st.info("👈 Fill in the student details in the sidebar and click "
                     "**Predict Placement** to see the result here.")

# ---------------------- Tab 2: Batch Prediction -------------------------
with tab_batch:
    st.subheader("Batch Prediction via CSV Upload")
    st.write(
        "Upload a CSV file containing the following columns "
        "(same as the training dataset, without the `Placement` column):"
    )
    st.code(
        "CGPA, IQ, Previous_Semester_Percentage, Communication_Skills, "
        "Aptitude_Score, Technical_Skills_Score, Projects_Completed, "
        "Internships, Certifications, Attendance_Percentage, Backlogs",
        language="text",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            with st.spinner("Running batch predictions..."):
                result_df = predict_batch(input_df)

            st.success(f"✅ Predictions generated for {len(result_df)} students.")
            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_bytes,
                file_name="placement_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except ValueError as ve:
            st.error(f"Column error: {ve}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process the uploaded file: {e}")

# ---------------------- Tab 3: Model Insights ---------------------------
with tab_insights:
    st.subheader("Model Performance & Feature Insights")
    st.caption(
        "These charts are generated automatically by `src/train.py` and "
        "saved to the `screenshots/` folder."
    )

    comparison_path = os.path.join(SCREENSHOTS_DIR, "model_comparison.png")
    importance_path = os.path.join(SCREENSHOTS_DIR, "feature_importance.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(comparison_path):
            st.image(comparison_path, caption="Model Accuracy Comparison", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate this chart.")
    with c2:
        if os.path.exists(importance_path):
            st.image(importance_path, caption="Feature Importance (Best Model)", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate this chart.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    '<div class="footer">Created by <b>Your Name</b> · '
    'Built with Streamlit &amp; Scikit-learn</div>',
    unsafe_allow_html=True,
)
