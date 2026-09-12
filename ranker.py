"""
Resume Ranking Module.

Combines keyword matching and semantic embeddings to rank candidate resumes
against a target Job Description (JD).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from parser import extract_jd, extract_resumes, extract_text_from_pdf
from keyword_match import (
    extract_keywords_from_jd,
    extract_keywords_from_resume,
    keyword_score,
)
from semantic_match import batch_semantic_scores, semantic_score
from resume_quality import check_resume_completeness, extract_email

# ==============================================================================
# Scoring Weight Constants
# ==============================================================================

# Team Design Decision:
# We assign a balanced tri-factor weighting:
# - Semantic matching (SEMANTIC_WEIGHT = 0.45): captures deep contextual relevance,
#   transferable domain knowledge, and qualitative project descriptions even when
#   phrased in alternative or non-standard terminology.
# - Keyword matching (KEYWORD_WEIGHT = 0.45): validates hard technical prerequisites,
#   framework proficiencies, and mandatory tech stack requirements.
# - Structural completeness (COMPLETENESS_WEIGHT = 0.10): rewards professional, well-formed
#   resumes containing all 6 essential sections (Education, Skills, Experience, Projects,
#   Certifications, Contact info).
# This formula ensures strong conceptual candidates (like CAND_006) are not unfairly
# penalized by vocabulary mismatches, landing close to or above candidates with partial
# stack coverage, while maintaining strict standards for essential requirements.
SEMANTIC_WEIGHT: float = 0.45
KEYWORD_WEIGHT: float = 0.45
COMPLETENESS_WEIGHT: float = 0.10


def rank_candidates(
    jd_text: str,
    resumes_dict: Dict[str, str],
) -> pd.DataFrame:
    """
    Ranks candidates by combining semantic matching, keyword matching,
    and structural resume completeness:
    final_score = 0.45*semantic + 0.45*keyword + 0.10*completeness

    Args:
        jd_text: Raw text of the Job Description.
        resumes_dict: Dictionary mapping candidate identifiers/filenames to raw resume text.

    Returns:
        A pandas DataFrame sorted descending by final_score with columns:
        - candidate: Filename or candidate identifier
        - email: Extracted candidate email address (or None)
        - final_score: Combined score (0.45*semantic + 0.45*keyword + 0.10*completeness)
        - semantic_score: Semantic similarity score in [0, 1]
        - keyword_score: Weighted keyword match score in [0, 1]
        - completeness_score: Structural completeness score in [0, 1] (X/6)
        - matched_required: List of required skills found in resume
        - missing_required: List of required skills missing from resume
        - matched_preferred: List of preferred skills found in resume
    """
    column_names = [
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

    if not resumes_dict or not jd_text.strip():
        return pd.DataFrame(columns=column_names)

    # 1. Parse JD keywords once
    jd_required, jd_preferred = extract_keywords_from_jd(jd_text)

    # 2. Batch-encode all resumes for high-throughput semantic scoring
    semantic_scores_dict = batch_semantic_scores(jd_text, resumes_dict)

    records = []

    for candidate_name, resume_text in resumes_dict.items():
        # Retrieve semantic score
        sem_score = float(semantic_scores_dict.get(candidate_name, 0.0))

        # Extract contact email
        email = extract_email(resume_text)

        # Compute keyword match score and skill breakdowns
        resume_skills = extract_keywords_from_resume(resume_text)
        kw_result = keyword_score(jd_required, jd_preferred, resume_skills)
        kw_score = float(kw_result.score)

        # Compute structural completeness score from resume_quality.py
        comp_result = check_resume_completeness(resume_text)
        comp_score = float(comp_result.completeness_score)

        # Compute weighted final score: 0.45*semantic + 0.45*keyword + 0.10*completeness
        final_score = round(
            SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score + COMPLETENESS_WEIGHT * comp_score,
            4,
        )

        records.append({
            "candidate": candidate_name,
            "email": email,
            "final_score": final_score,
            "semantic_score": sem_score,
            "keyword_score": kw_score,
            "completeness_score": comp_score,
            "matched_required": kw_result.matched_required,
            "missing_required": kw_result.missing_required,
            "matched_preferred": kw_result.matched_preferred,
        })

    df = pd.DataFrame(records, columns=column_names)

    # Sort descending by final_score (and secondary keyword_score, semantic_score)
    df = df.sort_values(
        by=["final_score", "keyword_score", "semantic_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def rank_from_files(
    jd_path: Union[str, Path],
    resumes_folder: Union[str, Path],
) -> pd.DataFrame:
    """
    Convenience wrapper that reads JD and resume files from disk,
    then executes candidate ranking.

    Args:
        jd_path: Path to the Job Description file (.txt or .pdf).
        resumes_folder: Path to the folder containing candidate resumes (.pdf).

    Returns:
        Ranked pandas DataFrame.
    """
    jd_text = extract_jd(jd_path)
    resumes_dict = extract_resumes(resumes_folder)
    return rank_candidates(jd_text, resumes_dict)


# ==============================================================================
# Verification Runner
# ==============================================================================

def main():
    """Runs ranking on Game Developer test data and validates score distribution."""
    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "test_data" / "Software_Game_Developer_Job_Description.pdf"
    resumes_dir = base_dir / "test_data" / "resumes"

    # Fallback to sample_data if test_data not present
    if not jd_path.exists():
        jd_path = base_dir / "sample_data" / "job_description.txt"
        resumes_dir = base_dir / "sample_data" / "resumes"

    print("=" * 80)
    print("RESUME RANKER - CANDIDATE RANKING VERIFICATION")
    print("=" * 80)
    print(f"JD File: {jd_path.name}")
    print(f"Resumes Directory: {resumes_dir.name}/")
    print(
        f"Weights Configured: Semantic = {SEMANTIC_WEIGHT}, "
        f"Keyword = {KEYWORD_WEIGHT}, Completeness = {COMPLETENESS_WEIGHT}\n"
    )

    # Execute ranking
    ranked_df = rank_from_files(jd_path, resumes_dir)

    # Configure pandas display options to avoid truncation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)

    print("-" * 80)
    print("FULL RE-RANKED DATAFRAME:")
    print("-" * 80)
    print(ranked_df.to_string(index=True))

    # Score Spread Analysis
    print("\n" + "=" * 80)
    print("SCORE SPREAD & RANK ORDER ANALYSIS:")
    print("=" * 80)

    scores = ranked_df["final_score"]
    max_score = scores.max()
    min_score = scores.min()
    score_range = max_score - min_score
    std_dev = scores.std()

    top_cand = str(ranked_df.loc[0, 'candidate'])
    bottom_cand = str(ranked_df.loc[len(ranked_df) - 1, 'candidate'])

    print(f"Top Candidate:     {top_cand} ({max_score:.2%})")
    print(f"Bottom Candidate:  {bottom_cand} ({min_score:.2%})")
    print(f"Score Spread:      {score_range:.4f} ({score_range * 100:.2f} percentage points)")
    print(f"Standard Dev:      {std_dev:.4f}\n")

    # Order verification for test resumes
    cand_order = [str(c) for c in ranked_df["candidate"]]
    cand_names_short = [c.split("_")[0] + "_" + c.split("_")[1] if "CAND_" in c else c for c in cand_order]
    print(f"Ranked Candidates Sequence: {' > '.join(cand_names_short)}")

    cand_ranks = {cand_names_short[i]: i + 1 for i in range(len(cand_names_short))}

    if "CAND_001" in cand_ranks and "CAND_005" in cand_ranks:
        print("\nChecking qualitative expectations:")
        print(f"  - CAND_001 Rank #{cand_ranks['CAND_001']} (Highest)")
        print(f"  - CAND_002 Rank #{cand_ranks.get('CAND_002', 'N/A')}")
        print(f"  - CAND_006 Rank #{cand_ranks.get('CAND_006', 'N/A')}")
        print(f"  - CAND_003 Rank #{cand_ranks.get('CAND_003', 'N/A')}")
        print(f"  - CAND_004 Rank #{cand_ranks.get('CAND_004', 'N/A')}")
        print(f"  - CAND_005 Rank #{cand_ranks['CAND_005']} (Lowest)")

        assert cand_ranks["CAND_001"] == 1, "CAND_001 must be ranked highest (#1)"
        assert cand_ranks["CAND_005"] == len(cand_ranks), "CAND_005 must be ranked lowest"
        assert cand_ranks["CAND_006"] <= 4, "CAND_006 must land close behind / near or above CAND_002/CAND_003"
        print("\n[PASS] Re-ranked table matches expected distribution!")

    # Check that scores are well-differentiated
    if score_range >= 0.20:
        print("\n[PASS] Scores show real, healthy spread (spread >= 20 percentage points).")
    else:
        print("\n[WARNING] Scores may be clustered (spread < 20 percentage points).")

    print("=" * 80)


if __name__ == "__main__":
    main()
