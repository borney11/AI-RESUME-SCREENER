import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# 📌 Utility Functions
# --------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from uploaded PDF"""
    text = ""
    with fitz.open(stream=pdf_path.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def match_resume_to_jobs(resume_text, jobs_df, top_n=3):
    """Match resume text to job descriptions using TF-IDF + cosine similarity"""
    documents = jobs_df["Job_Description"].tolist() + [resume_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vec = tfidf_matrix[-1]
    job_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vec, job_vecs).flatten()
    jobs_df["Score"] = similarities * 100
    top_matches = jobs_df.sort_values(by="Score", ascending=False).head(top_n)

    return top_matches[["Job_Title", "Score"]]

# --------------------------
# 📌 Streamlit UI
# --------------------------

st.title("📄 AI-Powered Resume Matcher")
st.write("Upload your resume PDF and get top matching job roles with scores.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    st.info("✅ Resume uploaded successfully!")

    # Load jobs dataset
    jobs_df = pd.read_csv("jobs.csv")

    # Extract resume text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Match and show results
    st.subheader("🔍 Top Job Matches")
    results = match_resume_to_jobs(resume_text, jobs_df, top_n=3)
    st.dataframe(results)

    # Optional: Show extracted resume text
    with st.expander("📜 View Extracted Resume Text"):
        st.write(resume_text)
