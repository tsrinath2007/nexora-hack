import io
import sys
import re
import importlib
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import streamlit as st

import ranker
try:
    importlib.reload(ranker)
except Exception:
    pass

from parser import extract_jd, extract_text_from_pdf, extract_resumes, extract_text_any, dedup_files
from ranker import rank_candidates, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
COMPLETENESS_WEIGHT = getattr(ranker, "COMPLETENESS_WEIGHT", 0.10)
from explain import generate_top3_explanations
from resume_quality import check_resume_completeness, extract_email

# Page configuration
st.set_page_config(
    page_title="Smart Shortlisting Engine",
    page_icon="🌳",
    layout="wide",
)

# Custom Brand Theme CSS (Dark Navy #1a2332, Card Navy #232f42, Teal Accent #2dd4a7)
st.markdown(
    """
    <style>
    /* Global Container Adjustments */
    .stApp {
        background-color: #1a2332;
        color: #e8ecf1;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        letter-spacing: -0.015em;
    }

    /* Header Section */
    .app-header {
        padding-top: 0.5rem;
        margin-bottom: 1.25rem;
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }
    .main-title .teal-accent {
        color: #2dd4a7;
    }
    .main-subtitle {
        font-size: 1.12rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .hackathon-caption {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 0.15rem;
    }
    .team-caption {
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }

    /* Info Badge / System Pill */
    .system-pill {
        background-color: #232f42;
        border: 1px solid #2e3e56;
        border-radius: 12px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.75rem;
        font-size: 0.92rem;
        color: #cbd5e1;
    }

    /* File Uploader Cards */
    [data-testid="stFileUploader"] {
        background-color: #232f42;
        border: 1px solid #2e3e56;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a2332 !important;
        border: 1.5px dashed #2dd4a7 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #94a3b8 !important;
    }

    /* Primary Buttons (Run Ranking) */
    .stButton > button[kind="primary"], div.stButton > button:first-child {
        background: linear-gradient(135deg, #2dd4a7, #1eb88e) !important;
        color: #0b201a !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(45, 212, 167, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="primary"]:hover, div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #37e4b7, #24c69b) !important;
        color: #051410 !important;
        box-shadow: 0 6px 22px rgba(45, 212, 167, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* Action Links (Send Mail) */
    .send-mail-link {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 6px 14px;
        background: #2dd4a7;
        color: #0b201a !important;
        text-decoration: none !important;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13.5px;
        box-shadow: 0 2px 8px rgba(45, 212, 167, 0.2);
        transition: all 0.15s ease;
    }
    .send-mail-link:hover {
        background: #39e6b8;
        color: #051410 !important;
        box-shadow: 0 4px 14px rgba(45, 212, 167, 0.35);
        transform: translateY(-1px);
    }
    .no-email-badge {
        display: inline-block;
        color: #64748b;
        font-size: 13px;
        font-style: italic;
    }

    /* Expanders & Cards */
    div[data-testid="stExpander"] {
        background-color: #232f42 !important;
        border: 1px solid #2e3e56 !important;
        border-radius: 12px !important;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600 !important;
        color: #e8ecf1 !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #1a2332;
        border: 1px solid #2e3e56;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #2dd4a7 !important;
        font-weight: 700 !important;
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2e3e56;
    }

    /* Horizontal Divider */
    hr {
        border-color: #2e3e56 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="app-header">
        <div class="main-title">🌳 <span class="teal-accent">Smart Shortlisting Engine</span></div>
        <div class="main-subtitle">Upload a Job Description and Candidate Resumes to find the perfect fit.</div>
        <div class="hackathon-caption">Built for the Nexora Hackathon @ Manipal Institute of Technology, Bengaluru</div>
        <div class="team-caption">Team: i dont know</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="system-pill">
        ⚡ <strong>Hybrid Multi-Criteria Evaluation</strong>: 
        <strong style="color: #2dd4a7;">Semantic Embeddings ({SEMANTIC_WEIGHT * 100:.0f}%)</strong> + 
        <strong style="color: #2dd4a7;">Keyword Matching ({KEYWORD_WEIGHT * 100:.0f}%)</strong> + 
        <strong style="color: #2dd4a7;">Resume Completeness ({COMPLETENESS_WEIGHT * 100:.0f}%)</strong>. 
        Supports <strong>PDF</strong>, <strong>DOCX</strong>, <strong>TXT</strong>, and <strong>XML</strong> formats with automated candidate deduplication.
    </div>
    """,
    unsafe_allow_html=True,
)

def get_jd_title(jd_name: str, jd_text: str) -> str:
    """Extracts a clean Job Description title from JD text or filename."""
    if jd_text:
        for line in jd_text.splitlines()[:6]:
            clean = line.strip()
            m = re.match(r"^(?:job\s+title|title|role|position)\s*[:\-]\s*(.+)$", clean, re.IGNORECASE)
            if m:
                return m.group(1).strip()
    if jd_name:
        stem = Path(jd_name).stem.replace("_", " ").replace("-", " ")
        stem = re.sub(r"(?i)\s*(?:job\s+description|jd|job)\s*$", "", stem).strip()
        if stem:
            return stem.title()
    return "Target Role"


# ==============================================================================
# 1. File Uploaders
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    st.caption("Upload the target job requirements and qualifications document.")
    jd_file = st.file_uploader(
        "Upload Job Description (PDF, DOCX, TXT, XML)",
        type=["pdf", "docx", "txt", "xml"],
        help="Upload the target job description file",
    )
    if jd_file:
        st.success(f"Loaded JD: {jd_file.name}")

with col2:
    st.subheader("2. Candidate Resumes")
    st.caption("Upload candidate resumes (up to 20 files) to evaluate and rank.")
    resume_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, TXT, XML - up to 20)",
        type=["pdf", "docx", "txt", "xml"],
        accept_multiple_files=True,
        help="Upload up to 20 candidate resume files",
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
        # 1. Deduplicate uploaded resumes before extraction using shared parser.dedup_files
        file_pairs = [(rf.name, rf) for rf in resume_files]
        deduped_resumes = dedup_files(file_pairs)

        # 2. Display deduplication details in UI and console
        if deduped_resumes.dropped:
            print(f"[DEDUP] Streamlit Upload: Resolved {len(deduped_resumes.dropped)} duplicate groups ({deduped_resumes.total_dropped} redundant files dropped).")
            for stem, info in deduped_resumes.dropped.items():
                print(f"  - Candidate '{stem}': KEPT '{info['kept']}', DROPPED {info['dropped']}")

            st.info(
                f"📋 **Resume Deduplication**: Detected **{len(resume_files)}** uploaded file(s) across "
                f"**{len(deduped_resumes)}** unique candidate(s). "
                f"Retained highest priority format (`.pdf > .docx > .txt > .xml`), dropping **{deduped_resumes.total_dropped}** redundant duplicate(s)."
            )
            with st.expander("🔍 View Deduplicated Candidates (Files Kept vs Dropped)", expanded=True):
                for stem, info in deduped_resumes.dropped.items():
                    st.markdown(f"**Candidate:** `{stem}`")
                    st.write(f"- ✅ **Kept:** `{info['kept']}`")
                    st.write(f"- ❌ **Dropped duplicate(s):** `{', '.join(info['dropped'])}`")

        with st.spinner("Extracting text from deduplicated candidate files..."):
            try:
                jd_text = extract_jd(jd_file)
            except Exception as e:
                st.error(f"Error parsing JD file: {e}")
                st.stop()

            for fname, fobj in deduped_resumes:
                try:
                    text = extract_text_any(fobj)
                    resumes_dict[fname] = text
                except Exception as e:
                    st.warning(f"Could not parse '{fname}': {e}")

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

        # Detect and store JD title for shortlist notifications
        jd_display_name = jd_file.name if jd_file else ("Demo Job Description" if use_sample else "")
        st.session_state["jd_title"] = get_jd_title(jd_display_name, jd_text)

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

    # Reorder columns to include email and completeness_score
    display_cols = [
        "candidate",
        "email",
        "final_score",
        "semantic_score",
        "keyword_score",
        "completeness_score",
        "matched_required",
        "missing_required",
        "matched_preferred",
    ]
    # Filter only available columns
    display_cols = [col for col in display_cols if col in display_df.columns]
    display_df = display_df[display_cols]

    # Show interactive sortable table
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "email": st.column_config.TextColumn("Email Address"),
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

    # Quick Candidate Outreach (Shortlist Notifications)
    st.markdown("##### ✉️ Quick Candidate Outreach")
    st.caption("Click below to draft an automated shortlist invitation in your default mail app:")
    outreach_df = ranked_df.head(3)
    jd_title = st.session_state.get("jd_title", "Position")

    outreach_header = st.columns([1, 4, 4, 3])
    outreach_header[0].markdown("**Rank**")
    outreach_header[1].markdown("**Candidate**")
    outreach_header[2].markdown("**Email**")
    outreach_header[3].markdown("**Action**")

    for idx, (_, row) in enumerate(outreach_df.iterrows()):
        r_cols = st.columns([1, 4, 4, 3])
        r_cols[0].write(f"#{idx + 1}")
        r_cols[1].write(f"**{row['candidate']}**")
        cand_email = row.get("email")
        if cand_email and pd.notna(cand_email) and str(cand_email).strip():
            r_cols[2].write(f"`{cand_email}`")
            subj = quote(f"You've been shortlisted - {jd_title}")
            c_name = Path(str(row["candidate"])).stem.replace("_", " ")
            body = quote(
                f"Hi {c_name},\n\n"
                f"Congratulations! You have been shortlisted for the {jd_title} position "
                f"(Rank #{idx + 1}, Score: {row['final_score']:.2%}).\n\n"
                f"We would like to invite you for an initial interview to discuss next steps.\n\n"
                f"Best regards,\n"
                f"The Hiring Team"
            )
            r_cols[3].markdown(
                f'<a href="mailto:{cand_email}?subject={subj}&body={body}" target="_blank" class="send-mail-link">'
                f'📧 Send Mail</a>',
                unsafe_allow_html=True,
            )
        else:
            r_cols[2].write("—")
            r_cols[3].markdown('<span class="no-email-badge">📧 No email found</span>', unsafe_allow_html=True)

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
        email = row.get("email")

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

            st.markdown("---")
            col_act1, col_act2 = st.columns([2, 5])
            with col_act1:
                if email and pd.notna(email) and str(email).strip():
                    subj = quote(f"You've been shortlisted - {jd_title}")
                    c_name = Path(str(candidate_name)).stem.replace("_", " ")
                    body = quote(
                        f"Hi {c_name},\n\n"
                        f"Congratulations! After reviewing your resume against our {jd_title} role, "
                        f"we are pleased to inform you that you have been ranked #{idx + 1} "
                        f"with an overall score of {final_score:.2%}.\n\n"
                        f"We would love to schedule an interview to discuss next steps.\n\n"
                        f"Best regards,\n"
                        f"The Hiring Team"
                    )
                    st.markdown(
                        f'<a href="mailto:{email}?subject={subj}&body={body}" target="_blank" class="send-mail-link">'
                        f'📧 Send Mail</a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<span class="no-email-badge">📧 No email found</span>', unsafe_allow_html=True)
            with col_act2:
                if email and pd.notna(email) and str(email).strip():
                    st.caption(f"Candidate Contact: `{email}`")
                else:
                    st.caption("No contact email detected in resume.")
