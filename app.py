import io
from pathlib import Path
import pandas as pd
import streamlit as st

from parser import extract_jd, extract_text_from_pdf, extract_resumes
from ranker import rank_candidates, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
from explain import generate_top3_explanations
from resume_quality import check_resume_completeness

# Page configuration
st.set_page_config(
    page_title="Resume Ranker",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Ranker")
st.markdown(
    f"""
    Rank and evaluate candidate resumes against a Job Description using hybrid
    **Semantic Embeddings ({SEMANTIC_WEIGHT * 100:.0f}%)**, **Keyword Matching ({KEYWORD_WEIGHT * 100:.0f}%)**,
    and **Resume Structural Completeness (X/6 sections)**.
    """
)

st.divider()

# ==============================================================================
# 1. File Uploaders
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    jd_file = st.file_uploader(
        "Upload Job Description (PDF)",
        type=["pdf"],
        help="Upload the target job description PDF",
    )
    if jd_file:
        st.success(f"Loaded JD: {jd_file.name}")

with col2:
    st.subheader("2. Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload Resumes (PDF, up to 20)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload up to 20 candidate resume PDFs",
    )
    if resume_files:
        if len(resume_files) > 20:
            st.warning(f"Uploaded {len(resume_files)} resumes. Only the first 20 will be evaluated.")
            resume_files = resume_files[:20]
        else:
            st.success(f"Loaded {len(resume_files)} resume(s)")

# Quick Demo Sample Data Option
with st.expander("💡 Or test with pre-loaded demo files"):
    use_sample = st.checkbox("Load demo Job Description and 3 sample resumes", value=False)
    if use_sample:
        st.info("Using sample ML Engineer JD and 3 sample resumes (Alice, Bob, Carol).")

st.divider()

# ==============================================================================
# 2. Ranking Execution
# ==============================================================================
run_button = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

if run_button:
    sample_dir = Path(__file__).resolve().parent / "sample_data"
    jd_text = ""
    resumes_dict = {}

    # Check input sources
    if jd_file and resume_files:
        with st.spinner("Extracting text from uploaded PDFs..."):
            try:
                jd_text = extract_text_from_pdf(jd_file)
            except Exception as e:
                st.error(f"Error parsing JD PDF: {e}")
                st.stop()

            for rf in resume_files:
                try:
                    text = extract_text_from_pdf(rf)
                    resumes_dict[rf.name] = text
                except Exception as e:
                    st.warning(f"Could not parse '{rf.name}': {e}")

    elif use_sample and sample_dir.exists():
        with st.spinner("Loading demo sample files..."):
            jd_path = sample_dir / "job_description.txt"
            resumes_folder = sample_dir / "resumes"

            jd_text = extract_jd(jd_path)
            resumes_dict = extract_resumes(resumes_folder)

    else:
        st.error("Please upload a Job Description (PDF) and at least one resume (PDF), or check the demo data option.")
        st.stop()

    if not jd_text.strip():
        st.error("The Job Description contains no extractable text. Please upload a valid text-based PDF.")
        st.stop()

    if not resumes_dict:
        st.error("No valid resumes were parsed. Please check your uploaded files.")
        st.stop()

    # Run ranking
    with st.spinner("Evaluating semantic scores, keyword matches, and structural completeness..."):
        ranked_df = rank_candidates(jd_text, resumes_dict)

        # Compute completeness_score (X/6) from resume_quality.py
        completeness_col = []
        for cand_name in ranked_df["candidate"]:
            raw_resume = resumes_dict.get(cand_name, "")
            comp_res = check_resume_completeness(raw_resume)
            completeness_col.append(f"{comp_res.found_count}/6")

        ranked_df["completeness_score"] = completeness_col
        st.session_state["ranked_df"] = ranked_df

# ==============================================================================
# 3. Results Display
# ==============================================================================
if "ranked_df" in st.session_state and not st.session_state["ranked_df"].empty:
    ranked_df: pd.DataFrame = st.session_state["ranked_df"]

    st.subheader("📊 Ranked Candidates")

    # Format lists as strings for clean table viewing
    display_df = ranked_df.copy()
    display_df["matched_required"] = display_df["matched_required"].apply(
        lambda lst: ", ".join(lst) if lst else "None"
    )
    display_df["missing_required"] = display_df["missing_required"].apply(
        lambda lst: ", ".join(lst) if lst else "None"
    )
    display_df["matched_preferred"] = display_df["matched_preferred"].apply(
        lambda lst: ", ".join(lst) if lst else "None"
    )

    # Reorder columns to include completeness_score
    display_cols = [
        "candidate",
        "final_score",
        "semantic_score",
        "keyword_score",
        "completeness_score",
        "matched_required",
        "missing_required",
        "matched_preferred",
    ]
    display_df = display_df[display_cols]

    # Show interactive sortable table
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "final_score": st.column_config.ProgressColumn(
                "Final Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "semantic_score": st.column_config.NumberColumn("Semantic Score", format="%.4f"),
            "keyword_score": st.column_config.NumberColumn("Keyword Score", format="%.4f"),
            "completeness_score": st.column_config.TextColumn("Completeness (X/6)"),
        },
    )

    st.divider()

    # ==============================================================================
    # 4. Score Distribution Bar Chart
    # ==============================================================================
    st.subheader("📈 Final Score Comparison Across Candidates")
    chart_data = ranked_df[["candidate", "final_score"]].set_index("candidate")
    st.bar_chart(chart_data)

    st.divider()

    # ==============================================================================
    # 5. Top 3 Explanations
    # ==============================================================================
    st.subheader("💡 Top 3 Candidate Explanations")
    st.markdown("Deterministic, rule-based audit trail detailing skill matches, gaps, and semantic alignment:")

    top3_explanations = generate_top3_explanations(ranked_df)
    top3_rows = ranked_df.head(3)

    for idx, (_, row) in enumerate(top3_rows.iterrows()):
        candidate_name = row["candidate"]
        final_score = row["final_score"]
        completeness = row.get("completeness_score", "N/A")
        explanation = top3_explanations[idx]

        with st.expander(f"🏆 Rank #{idx + 1}: {candidate_name} (Score: {final_score:.2f})", expanded=(idx == 0)):
            st.markdown(f"**Explanation:** {explanation}")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Final Score", f"{final_score:.2%}")
            with col_m2:
                st.metric("Semantic Score", f"{row['semantic_score']:.2%}")
            with col_m3:
                st.metric("Keyword Score", f"{row['keyword_score']:.2%}")
            with col_m4:
                st.metric("Completeness", completeness)
