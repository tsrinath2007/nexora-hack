"""
Resume Ranking Module.

Combines keyword matching and semantic embeddings to rank candidate resumes
against a target Job Description (JD).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from parser import extract_jd, extract_resumes
from keyword_match import (
    extract_keywords_from_jd,
    extract_keywords_from_resume,
    keyword_score,
)
from semantic_match import batch_semantic_scores, semantic_score

# ==============================================================================
# Scoring Weight Constants
# ==============================================================================

# Team Design Decision:
# We assign equal 50/50 weighting to semantic similarity and keyword matching.
# - Semantic matching (SEMANTIC_WEIGHT = 0.5) captures conceptual context,
#   transferable domain knowledge, and qualitative project descriptions even
#   when phrased differently from the JD.
# - Keyword matching (KEYWORD_WEIGHT = 0.5) enforces strict technical competency
#   by validating core tools, languages, and hard framework requirements.
# This prevents candidates with keyword-stuffed resumes lacking context from
# dominating, while ensuring candidates missing mandatory technical stack
# prerequisites are appropriately penalized.
SEMANTIC_WEIGHT: float = 0.5
KEYWORD_WEIGHT: float = 0.5


def rank_candidates(
    jd_text: str,
    resumes_dict: Dict[str, str],
) -> pd.DataFrame:
    """
    Ranks candidates by combining semantic matching and keyword matching.

    Args:
        jd_text: Raw text of the Job Description.
        resumes_dict: Dictionary mapping candidate identifiers/filenames to raw resume text.

    Returns:
        A pandas DataFrame sorted descending by final_score with columns:
        - candidate: Filename or candidate identifier
        - semantic_score: Semantic similarity score in [0, 1]
        - keyword_score: Weighted keyword match score in [0, 1]
        - final_score: Combined score (0.5 * semantic + 0.5 * keyword)
        - matched_required: List of required skills found in resume
        - missing_required: List of required skills missing from resume
        - matched_preferred: List of preferred skills found in resume
    """
    column_names = [
        "candidate",
        "semantic_score",
        "keyword_score",
        "final_score",
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

        # Compute keyword match score and skill breakdowns
        resume_skills = extract_keywords_from_resume(resume_text)
        kw_result = keyword_score(jd_required, jd_preferred, resume_skills)
        kw_score = float(kw_result.score)

        # Compute weighted final score
        final_score = round(
            SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score,
            4,
        )

        records.append({
            "candidate": candidate_name,
            "semantic_score": sem_score,
            "keyword_score": kw_score,
            "final_score": final_score,
            "matched_required": kw_result.matched_required,
            "missing_required": kw_result.missing_required,
            "matched_preferred": kw_result.matched_preferred,
        })

    df = pd.DataFrame(records, columns=column_names)

    # Sort descending by final_score (and secondary keyword_score)
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
    """Runs ranking on sample data and validates score distribution."""
    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "sample_data" / "job_description.txt"
    resumes_dir = base_dir / "sample_data" / "resumes"

    print("=" * 80)
    print("RESUME RANKER - CANDIDATE RANKING VERIFICATION")
    print("=" * 80)
    print(f"JD File: {jd_path.name}")
    print(f"Resumes Directory: {resumes_dir.name}/")
    print(f"Weights Configured: Semantic = {SEMANTIC_WEIGHT}, Keyword = {KEYWORD_WEIGHT}\n")

    # Execute ranking
    ranked_df = rank_from_files(jd_path, resumes_dir)

    # Configure pandas display options to avoid truncation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)

    print("-" * 80)
    print("FULL RANKED DATAFRAME:")
    print("-" * 80)
    print(ranked_df.to_string(index=True))

    # Score Spread Analysis
    print("\n" + "=" * 80)
    print("SCORE SPREAD ANALYSIS:")
    print("=" * 80)

    scores = ranked_df["final_score"]
    max_score = scores.max()
    min_score = scores.min()
    score_range = max_score - min_score
    std_dev = scores.std()

    print(f"Top Candidate:     {ranked_df.loc[0, 'candidate']} ({max_score:.2%})")
    print(f"Bottom Candidate:  {ranked_df.loc[len(ranked_df) - 1, 'candidate']} ({min_score:.2%})")
    print(f"Score Spread:      {score_range:.4f} ({score_range * 100:.2f} percentage points)")
    print(f"Standard Dev:      {std_dev:.4f}")

    # Check that scores are well-differentiated
    if score_range >= 0.20:
        print("\n[PASS] Scores show real, healthy spread (spread >= 20 percentage points).")
    else:
        print("\n[WARNING] Scores may be clustered (spread < 20 percentage points).")

    print("=" * 80)


if __name__ == "__main__":
    main()
