import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF
import docx2txt  # For DOCX
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# 📌 Utility Functions
# --------------------------

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def clean_text(text):
    """Clean and normalize text for better matching."""
    # Fix formatting issues: join broken words like 'MachineLearning' -> 'Machine Learning'
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def prepare_job_descriptions(df):
    combined_texts = []
    for _, row in df.iterrows():
        combined = f"{row['Job_Description']} {row.get('Skills','')} {row.get('Experience','')}"
        combined_texts.append(clean_text(combined))
    df["Combined_Text"] = combined_texts
    return df

def match_resume_to_jobs(resume_text, jobs_df, top_n=5):
    documents = jobs_df["Combined_Text"].tolist() + [resume_text]
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,3), max_features=15000)
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

st.set_page_config(page_title="AI-Powered Resume Screener", layout="wide")

st.title("📄 AI-Powered Resume Screener")
st.write("Upload your resume in PDF, DOCX, or TXT format and get the top matching job roles with detailed similarity scores.")

uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.info("✅ Resume uploaded successfully!")

    jobs_df = pd.read_csv("jobs.csv")

    if "Skills" not in jobs_df.columns:
        jobs_df["Skills"] = ""
    if "Experience" not in jobs_df.columns:
        jobs_df["Experience"] = ""

    jobs_df = prepare_job_descriptions(jobs_df)

    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    elif ext == "txt":
        resume_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type!")
        st.stop()

    resume_text = clean_text(resume_text)

    st.subheader("🔍 Top Job Matches")
    results = match_resume_to_jobs(resume_text, jobs_df, top_n=5)
    st.dataframe(results.style.format({"Score": "{:.2f}"}))

    with st.expander("📜 View Extracted Resume Text"):
        st.write(resume_text)

    st.subheader("📊 Match Scores Visualization")
    st.bar_chart(results.set_index("Job_Title")["Score"])

else:
    st.info("Please upload your resume to get started.")
