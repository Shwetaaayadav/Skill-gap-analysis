import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Skill Gap Analysis", layout="wide")

st.title("📊 Skill Gap Analysis Dashboard")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("Upload Files")

job_file = st.sidebar.file_uploader("Upload Job Dataset (CSV)", type=["csv"])
fy_pdf = st.sidebar.file_uploader("Upload FY Syllabus PDF", type=["pdf"])
sy_pdf = st.sidebar.file_uploader("Upload SY Syllabus PDF", type=["pdf"])
ty_pdf = st.sidebar.file_uploader("Upload TY Syllabus PDF", type=["pdf"])

# ---------------- FUNCTIONS ----------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text.lower()

def normalize(skill):
    return skill.replace("-", " ").replace("_", " ").strip().lower()

# ---------------- MAIN LOGIC ----------------
if job_file and fy_pdf and sy_pdf and ty_pdf:

    with st.spinner("Processing data... ⏳"):

        df = pd.read_csv(job_file)

        df['combined_skills'] = (df['Skills'] + ";" + df['Keywords']).fillna("").str.lower()
        df['skill_list'] = df['combined_skills'].apply(
            lambda x: [s.strip() for s in x.split(";") if s.strip()]
        )

        # Extract skills
        all_skills = set()
        for skills in df['skill_list']:
            for skill in skills:
                all_skills.add(normalize(skill))

        cleaned_skills = set(s for s in all_skills if len(s) > 2)

        # Extract syllabus text
        fy_text = extract_text(fy_pdf)
        sy_text = extract_text(sy_pdf)
        ty_text = extract_text(ty_pdf)

        all_text = fy_text + " " + sy_text + " " + ty_text

        curriculum_skills = set()
        for skill in cleaned_skills:
            if skill in all_text:
                curriculum_skills.add(skill)

        curriculum_skill_list = list(curriculum_skills)

        # Role skills mapping
        df['clean_role'] = df['Title'].astype(str).str.lower().str.strip()
        role_skills = {}

        for _, row in df.iterrows():
            role = row['clean_role']
            skills = set(row['skill_list'])
            role_skills.setdefault(role, set()).update(skills)

        # Graph creation
        G = nx.Graph()
        for skills in df['skill_list']:
            skills = list(set(skills))
            for i in range(len(skills)):
                for j in range(i+1, len(skills)):
                    u, v = skills[i], skills[j]
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)

        # Model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        curriculum_embeddings = model.encode(curriculum_skill_list)

        def is_skill_covered(skill, threshold=0.6):
            skill_vec = model.encode([skill])
            similarities = cosine_similarity(skill_vec, curriculum_embeddings)
            return np.max(similarities) > threshold

        def graph_similarity(skill):
            for cs in curriculum_skill_list:
                if G.has_edge(skill, cs):
                    return True
            return False

        def is_skill_covered_final(skill):
            return is_skill_covered(skill) or graph_similarity(skill)

        # Compute gaps
        output_data = []

        for role, skills in role_skills.items():
            missing = [s for s in skills if not is_skill_covered_final(s)]
            total = len(skills)
            gap_percent = (len(missing) / total) * 100 if total > 0 else 0

            output_data.append({
                "Role": role,
                "Total Skills": total,
                "Missing Count": len(missing),
                "Gap %": round(gap_percent, 2),
                "Missing Skills": ", ".join(missing)
            })

        final_df = pd.DataFrame(output_data).sort_values(by="Gap %", ascending=False)

    st.success("Analysis Completed ✅")

    # ---------------- DASHBOARD ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Roles", len(final_df))

    with col2:
        st.metric("Avg Skill Gap (%)", round(final_df["Gap %"].mean(), 2))

    st.subheader("📈 Skill Gap Table")
    st.dataframe(final_df, use_container_width=True)

    # ---------------- ROLE FILTER ----------------
    st.subheader("🔍 Role-wise Analysis")

    selected_role = st.selectbox("Select Role", final_df["Role"])

    role_data = final_df[final_df["Role"] == selected_role].iloc[0]

    st.write(f"### {selected_role}")
    st.write(f"**Gap %:** {role_data['Gap %']}%")
    st.write(f"**Missing Skills:**")
    st.write(role_data["Missing Skills"])

else:
    st.info("👈 Please upload all required files to begin analysis.")