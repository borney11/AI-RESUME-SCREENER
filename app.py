import streamlit as st
import pandas as pd
from resume_match import extract_text_from_pdf, match_resume_to_jobs

st.title("📄 AI-Powered Resume Matcher")
st.write("Upload your resume and get top matching job roles with scores.")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract resume text
    resume_text = extract_text_from_pdf("uploaded_resume.pdf")

    # Match with jobs
    results = match_resume_to_jobs(resume_text, top_n=3)

    # Display Results
    st.subheader("✅ Top Matching Jobs")
    st.dataframe(pd.DataFrame(results))

    # Show extracted text for debugging
    with st.expander("🔍 View Extracted Resume Text"):
        st.text(resume_text)
