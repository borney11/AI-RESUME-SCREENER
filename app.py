import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --------------------------
# Utility Functions
# --------------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(stream=pdf_path.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_section(text, section_name):
    pattern = rf"{section_name}(.+?)(?=\n[A-Z ]{{3,}}|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_skills(text, skill_set):
    text_lower = text.lower()
    found_skills = [skill for skill in skill_set if skill.lower() in text_lower]
    return list(set(found_skills))

def extract_tools(text, tool_set):
    text_lower = text.lower()
    found_tools = [tool for tool in tool_set if tool.lower() in text_lower]
    return list(set(found_tools))

def extract_experience(text):
    return extract_section(text, "EXPERIENCE")

def build_resume_text(resume_text, skills_list, tools_list, experience_text):
    # Combine resume raw text + extracted skills + tools + experience for richer representation
    combined = " ".join([
        resume_text,
        " ".join(skills_list),
        " ".join(tools_list),
        experience_text
    ])
    return combined

def match_resume_to_jobs(resume_text, jobs_df, top_n=5):
    # Combine relevant job columns for matching
    combined_jobs_texts = (
        jobs_df["Job_Description"].fillna("") + " " +
        jobs_df["Skills"].fillna("") + " " +
        jobs_df["Experience"].fillna("") + " " +
        jobs_df["Tools"].fillna("")
    ).tolist()

    documents = combined_jobs_texts + [resume_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vec = tfidf_matrix[-1]
    job_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vec, job_vecs).flatten()
    jobs_df["Score"] = similarities * 100
    top_matches = jobs_df.sort_values(by="Score", ascending=False).head(top_n)

    return top_matches[["Job_Title", "Score"]]

# --------------------------
# Streamlit UI
# --------------------------

st.title("AI-Powered Resume Screener")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    st.info("Resume uploaded successfully!")

    jobs_df = pd.read_csv("jobs.csv", on_bad_lines='skip', engine='python')


    # Predefined skill & tool sets based on jobs.csv columns
    # Extract unique skills and tools from jobs_df columns
    all_skills = set()
    all_tools = set()
    for s in jobs_df["Skills"].dropna():
        all_skills.update([skill.strip() for skill in s.split(",")])
    for t in jobs_df["Tools"].dropna():
        all_tools.update([tool.strip() for tool in t.split(",")])

    # Extract resume text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Extract skills, tools, experience from resume text
    skills_found = extract_skills(resume_text, all_skills)
    tools_found = extract_tools(resume_text, all_tools)
    experience_text = extract_experience(resume_text)

    # st.subheader("🛠Extracted Skills")
    # st.write(skills_found)

    # st.subheader("Extracted Tools")
    # st.write(tools_found)

    # st.subheader("Extracted Experience")
    # st.write(experience_text[:1000])

    combined_resume_text = build_resume_text(resume_text, skills_found, tools_found, experience_text)

    st.subheader("Top Job Matches")
    results = match_resume_to_jobs(combined_resume_text, jobs_df, top_n=5)
    st.dataframe(results.style.format({"Score": "{:.2f}"}))
