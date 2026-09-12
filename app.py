import io
import sys
import re
import importlib
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import ranker
try:
    importlib.reload(ranker)
except Exception:
    pass

from parser import extract_jd, extract_text_from_pdf, extract_resumes, extract_text_any, dedup_files
from ranker import rank_candidates, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
COMPLETENESS_WEIGHT = getattr(ranker, "COMPLETENESS_WEIGHT", 0.10)
from explain import (
    generate_top3_explanations,
    recommend_best_fit,
    compare_candidates,
    compare_multiple,
    extract_two_candidates_from_query,
    _clean_candidate_label,
)
from resume_quality import check_resume_completeness, extract_email
from jd_bias_check import flag_jd_bias

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

    /* Explanation Card & Structured Points */
    .explanation-box {
        background-color: #1a2332;
        border: 1px solid #2e3e56;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
        font-size: 0.93rem;
        line-height: 1.65;
        color: #e2e8f0;
    }
    .explanation-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    .explanation-item:last-child {
        margin-bottom: 0;
    }
    .explanation-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        height: 24px;
        background-color: rgba(45, 212, 167, 0.15);
        color: #2dd4a7;
        font-weight: 700;
        font-size: 0.82rem;
        border-radius: 6px;
        border: 1px solid rgba(45, 212, 167, 0.35);
        flex-shrink: 0;
        margin-top: 1px;
    }
    .explanation-text {
        flex: 1;
    }

    /* Best Fit Recommendation Banner */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(45, 212, 167, 0.08), rgba(35, 47, 66, 0.95));
        border: 1.5px solid #2dd4a7;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(45, 212, 167, 0.12);
    }
    .recommendation-title {
        color: #2dd4a7;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .recommendation-body {
        color: #e2e8f0;
        font-size: 0.96rem;
        line-height: 1.65;
        margin-bottom: 0.85rem;
    }
    .recommendation-closing {
        background-color: rgba(45, 212, 167, 0.12);
        border-left: 3.5px solid #2dd4a7;
        padding: 0.6rem 0.95rem;
        border-radius: 0 8px 8px 0;
        color: #5eead4;
        font-weight: 600;
        font-size: 0.94rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* Recruiter Q&A Chat Styling */
    .chat-qa-card {
        background-color: #232f42;
        border: 1px solid #2e3e56;
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .chat-question-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: rgba(45, 212, 167, 0.12);
        color: #2dd4a7;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(45, 212, 167, 0.3);
    }
    .chat-bubble-response {
        background-color: #1a2332;
        border-left: 3.5px solid #2dd4a7;
        border-radius: 4px 10px 10px 4px;
        padding: 1.15rem 1.35rem;
        color: #e2e8f0;
        font-size: 0.94rem;
        line-height: 1.65;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
logo_file = Path(__file__).parent / "assets" / "internloom_logo.png"
if not logo_file.exists():
    logo_file = Path("assets/internloom_logo.png")

if logo_file.exists():
    header_col1, header_col2 = st.columns([1.6, 8.4], vertical_alignment="center")
    with header_col1:
        st.image(str(logo_file), width=180)
    with header_col2:
        st.markdown(
            """
            <div class="main-title" style="margin-bottom: 0; line-height: 1.2;">
                <span class="teal-accent">Smart Shortlisting Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="app-header" style="padding-top: 0.25rem; margin-top: -0.25rem;">
            <div class="main-subtitle">Upload a Job Description and Candidate Resumes to find the perfect fit.</div>
            <div class="hackathon-caption">Built for the Nexora Hackathon @ Manipal Institute of Technology, Bengaluru</div>
            <div class="team-caption">Team: i dont know</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
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


def format_chart_candidate_label(filename: str, max_chars: int = 30) -> str:
    """
    Cleans and shortens candidate filenames for horizontal chart axis display:
    - Strips file extensions (.pdf, .docx, .txt, .xml)
    - Removes common redundant terms ('resume', 'cv')
    - Converts underscores and hyphens to spaces
    - Capitalizes if lowercase
    - Truncates long names gracefully with ellipsis
    """
    stem = Path(filename).stem
    cleaned = re.sub(r"(?i)[_\-\s]*(?:resume|cv)\b", "", stem)
    cleaned = re.sub(r"(?i)\b(?:resume|cv)[_\-\s]*", "", cleaned)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = Path(filename).stem
    if cleaned.islower():
        cleaned = cleaned.title()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1] + "…"
    return cleaned


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

# ==============================================================================
# JD Bias & Inclusivity Audit
# ==============================================================================
preview_jd_text = ""
if jd_file:
    try:
        preview_jd_text = extract_jd(jd_file)
    except Exception:
        pass
elif use_sample:
    sample_jd_path = Path(__file__).resolve().parent / "sample_data" / "job_description.txt"
    if sample_jd_path.exists():
        preview_jd_text = sample_jd_path.read_text(encoding="utf-8")

if jd_file or use_sample:
    bias_flags = flag_jd_bias(preview_jd_text) if preview_jd_text else []
    with st.expander("⚠️ JD Bias Check", expanded=(len(bias_flags) > 0)):
        st.markdown("Automated scan for overly narrow tool requirements, seniority-experience mismatches, exclusionary/age-coded phrasing, gendered terms, and degree gatekeeping:")
        if bias_flags:
            st.warning(f"Detected **{len(bias_flags)}** potential narrow-phrasing or bias concern(s):")
            for flag in bias_flags:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ **No major bias concerns detected** in this Job Description.")

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

    # Best Fit Recommendation Banner
    best_fit = recommend_best_fit(ranked_df)
    if best_fit:
        st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-title">🏆 Our Recommendation</div>
                <div class="recommendation-body">{best_fit.paragraph}</div>
                <div class="recommendation-closing">✅ {best_fit.closing}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
    # 4. Score Distribution Bar Chart (Horizontal Plotly)
    # ==============================================================================
    st.subheader("📈 Final Score Comparison Across Candidates")
    st.caption("Visual ranking comparison across all evaluated candidates (ordered from highest to lowest score):")

    # Prepare data for horizontal bar chart
    plot_df = ranked_df.copy()
    plot_df["rank"] = range(1, len(plot_df) + 1)
    plot_df["display_name"] = plot_df["candidate"].apply(format_chart_candidate_label)

    # Reverse order so highest-ranked candidate (#1) appears at the top
    chart_data = plot_df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_data["final_score"],
            y=chart_data["display_name"],
            orientation="h",
            marker=dict(
                color="#2dd4a7",
                line=dict(color="#24c69b", width=1),
            ),
            text=[f" {score:.1%}" for score in chart_data["final_score"]],
            textposition="outside",
            textfont=dict(
                color="#e8ecf1",
                size=12,
                family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            ),
            customdata=list(
                zip(
                    chart_data["candidate"],
                    chart_data["rank"],
                    chart_data["semantic_score"],
                    chart_data["keyword_score"],
                    chart_data["completeness_score"],
                )
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Rank: #%{customdata[1]}<br>"
                "Final Score: <b>%{x:.2%}</b><br>"
                "Semantic Score: %{customdata[2]:.2%}<br>"
                "Keyword Score: %{customdata[3]:.2%}<br>"
                "Completeness: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
        )
    )

    chart_height = max(280, len(chart_data) * 45 + 70)
    fig.update_layout(
        paper_bgcolor="#232f42",
        plot_bgcolor="#1a2332",
        margin=dict(l=15, r=60, t=25, b=25),
        height=chart_height,
        xaxis=dict(
            title=dict(text="Final Score", font=dict(color="#94a3b8", size=12)),
            tickformat=".0%",
            range=[0, 1.08],
            gridcolor="#2e3e56",
            zerolinecolor="#2e3e56",
            tickfont=dict(color="#94a3b8", size=11),
        ),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=chart_data["display_name"].tolist(),
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#e8ecf1", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#2dd4a7",
            font=dict(color="#e8ecf1", size=12),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

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
            st.markdown("##### 📋 Detailed Candidate Evaluation")
            if hasattr(explanation, "sentences") and explanation.sentences:
                items_html = "".join(
                    f'<div class="explanation-item">'
                    f'<span class="explanation-badge">{i}</span>'
                    f'<span class="explanation-text">{s}</span>'
                    f'</div>'
                    for i, s in enumerate(explanation.sentences, 1)
                )
                st.markdown(f'<div class="explanation-box">{items_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"{explanation}")
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

    st.divider()

    # ==============================================================================
    # 6. Recruiter Q&A & Head-to-Head Candidate Comparison
    # ==============================================================================
    st.subheader("💬 Ask About the Ranking & Compare Candidates")
    st.markdown(
        "Ask a natural language question comparing any two candidates (e.g., *'Why is CAND_001 ranked above CAND_003?'*), "
        "or select candidates directly using the dropdowns below for an instant comparative breakdown."
    )

    all_candidate_files = ranked_df["candidate"].tolist()

    # Recruiter Chat-style Q&A Input
    user_query = st.text_input(
        "Ask about the ranking",
        placeholder="e.g. Why is CAND_001 ranked above CAND_003?",
        key="recruiter_ranking_qa",
    )

    if user_query and user_query.strip():
        cand_a_name, cand_b_name = extract_two_candidates_from_query(user_query, all_candidate_files)
        if cand_a_name and cand_b_name:
            row_a = ranked_df[ranked_df["candidate"] == cand_a_name].iloc[0]
            row_b = ranked_df[ranked_df["candidate"] == cand_b_name].iloc[0]
            comparison_answer = compare_candidates(row_a, row_b)

            clean_a = _clean_candidate_label(cand_a_name)
            clean_b = _clean_candidate_label(cand_b_name)
            rank_a = all_candidate_files.index(cand_a_name) + 1
            rank_b = all_candidate_files.index(cand_b_name) + 1

            p_list = [p.strip() for p in comparison_answer.split("\n\n") if p.strip()]
            p_html_list = "".join(
                f'<div style="margin-bottom: 0.75rem;">{re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", p)}</div>'
                for p in p_list
            )

            st.markdown(
                f"""
                <div class="chat-qa-card">
                    <div class="chat-question-pill">💬 Recruiter Question: "{user_query.strip()}"</div>
                    <div style="font-size: 0.86rem; color: #94a3b8; margin-bottom: 0.85rem;">
                        Comparing <strong>{clean_a}</strong> (Rank #{rank_a}) vs <strong>{clean_b}</strong> (Rank #{rank_b}):
                    </div>
                    <div class="chat-bubble-response">
                        {p_html_list}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "💡 I couldn't identify two candidates in that question — try naming them directly, "
                "e.g. *'Why is CAND_001 ranked above CAND_003?'*"
            )

    st.markdown("##### 🔍 Or Select Candidates Directly to Compare")

    default_selected = all_candidate_files[:2] if len(all_candidate_files) >= 2 else all_candidate_files

    selected_candidates = st.multiselect(
        "Compare Candidates",
        options=all_candidate_files,
        default=default_selected,
        format_func=lambda c: f"#{all_candidate_files.index(c) + 1}: {_clean_candidate_label(c)}",
        max_selections=5,
        help="Select between 2 and 5 candidates for a comparative breakdown.",
        key="compare_candidates_multiselect",
    )

    if len(selected_candidates) < 2:
        st.info("Select at least 2 candidates to compare.")
    else:
        selected_rows = [ranked_df[ranked_df["candidate"] == c].iloc[0] for c in selected_candidates]
        comparison_result = compare_multiple(selected_rows)

        p_list_mult = [p.strip() for p in comparison_result.split("\n\n") if p.strip()]
        p_html_mult = "".join(
            f'<div style="margin-bottom: 0.75rem;">{re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", p)}</div>'
            for p in p_list_mult
        )
        st.markdown(
            f'<div class="chat-bubble-response">{p_html_mult}</div>',
            unsafe_allow_html=True,
        )
