"""
Rule-based Candidate Explanation Module.

Generates concise, human-readable 2-3 sentence explanations for ranked candidates
based on deterministic rules, skill matches, missing requirements, and semantic alignment.
Fully rule-based without LLM calls for complete auditability.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import pandas as pd


# Display name overrides for clean presentation
_SKILL_DISPLAY_NAMES: Dict[str, str] = {
    "ci/cd": "CI/CD",
    "sql": "SQL",
    "html": "HTML",
    "css": "CSS",
    "aws": "AWS",
    "gcp": "GCP",
    "api": "API",
    "rest api": "REST API",
    "node.js": "Node.js",
    "next.js": "Next.js",
    "vue.js": "Vue.js",
    "scikit-learn": "Scikit-Learn",
    "pytorch": "PyTorch",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "mongodb": "MongoDB",
    "postgresql": "PostgreSQL",
    "fastapi": "FastAPI",
    "sentence-transformers": "Sentence-Transformers",
    "natural language processing": "Natural Language Processing (NLP)",
    "large language models": "Large Language Models (LLMs)",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "vector search": "Vector Search",
    "docker": "Docker",
    "git": "Git",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "python": "Python",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "kubernetes": "Kubernetes",
}


def _format_skill_name(skill: str) -> str:
    """Formats a skill string into standard canonical casing."""
    lower = skill.strip().lower()
    if lower in _SKILL_DISPLAY_NAMES:
        return _SKILL_DISPLAY_NAMES[lower]
    # Default to Title Case
    return skill.title()


def _format_skill_list(skills: List[str], max_items: int = 5) -> str:
    """Formats a list of skill strings into a human-readable comma list."""
    if not skills:
        return ""
    formatted = [_format_skill_name(s) for s in skills]
    if len(formatted) > max_items:
        truncated_count = len(formatted) - max_items
        formatted = formatted[:max_items] + [f"+{truncated_count} more"]
    return ", ".join(formatted)


def generate_explanation(
    row: Union[pd.Series, Dict[str, Any]],
    rank: Optional[int] = None,
) -> str:
    """
    Generates a clear 2-3 sentence rule-based explanation for a ranked candidate.

    Args:
        row: A pandas Series or dictionary containing candidate ranking data:
             - candidate (str)
             - final_score (float)
             - semantic_score (float, optional)
             - keyword_score (float, optional)
             - matched_required (list[str])
             - missing_required (list[str])
             - matched_preferred (list[str])
        rank: Optional 1-based rank. If None, derived from row index if available.

    Returns:
        A 2-3 sentence explanatory string.
    """
    # 1. Determine rank
    if rank is None:
        if isinstance(row, pd.Series) and isinstance(row.name, int):
            rank = row.name + 1
        elif "rank" in row:
            rank = int(row["rank"])
        else:
            rank = 1

    final_score = float(row.get("final_score", 0.0))
    matched_req = list(row.get("matched_required", []))
    missing_req = list(row.get("missing_required", []))
    matched_pref = list(row.get("matched_preferred", []))
    semantic_score = float(row.get("semantic_score", 0.0))

    # Sentence 1: Rank and overall score summary
    sentence_1 = f"Ranked #{rank} with score {final_score:.2f}."

    # Sentence 2: Required skills matching and gaps
    if matched_req and missing_req:
        matched_str = _format_skill_list(matched_req)
        missing_str = _format_skill_list(missing_req)
        sentence_2 = f"Matches required skills: {matched_str}. Missing: {missing_str}."
    elif matched_req and not missing_req:
        matched_str = _format_skill_list(matched_req)
        sentence_2 = f"Matches all required skills: {matched_str} with zero missing prerequisites."
    elif not matched_req and missing_req:
        missing_str = _format_skill_list(missing_req)
        sentence_2 = f"Matches none of the required skills. Missing: {missing_str}."
    else:
        sentence_2 = "No specific required skills were identified for comparison."

    # Sentence 3: Preferred skills and semantic domain alignment
    if matched_pref:
        pref_str = _format_skill_list(matched_pref)
        sentence_3 = f"Also shows relevant experience in {pref_str} that aligns semantically with the JD."
    elif semantic_score >= 0.70:
        sentence_3 = "Demonstrates strong contextual and qualitative domain alignment with the JD requirements."
    elif semantic_score >= 0.50:
        sentence_3 = "Displays moderate general domain overlap with the target role."
    else:
        sentence_3 = "Shows limited technical and contextual alignment with this role's focus areas."

    return f"{sentence_1} {sentence_2} {sentence_3}"


class Top3Explanations(list):
    """
    List of top 3 candidate explanations that also provides dictionary
    mapping by candidate name and pretty printing.
    """
    def __init__(self, explanations: List[str], mapping: Dict[str, str]):
        super().__init__(explanations)
        self.mapping = mapping

    def to_dict(self) -> Dict[str, str]:
        """Returns candidate -> explanation dictionary."""
        return self.mapping

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.mapping[key]
        return super().__getitem__(key)


def generate_top3_explanations(ranked_df: pd.DataFrame) -> Top3Explanations:
    """
    Generates rule-based explanations for the top 3 candidates in a ranked DataFrame.

    Args:
        ranked_df: pandas DataFrame returned by ranker.rank_candidates.

    Returns:
        Top3Explanations object (behaves as a list of strings, with .to_dict() support).
    """
    if ranked_df.empty:
        return Top3Explanations([], {})

    top3_df = ranked_df.head(3)
    explanations: List[str] = []
    mapping: Dict[str, str] = {}

    for idx, (_, row) in enumerate(top3_df.iterrows(), start=1):
        explanation = generate_explanation(row, rank=idx)
        candidate_name = str(row.get("candidate", f"Candidate #{idx}"))
        explanations.append(explanation)
        mapping[candidate_name] = explanation

    return Top3Explanations(explanations, mapping)


# ==============================================================================
# Verification Runner
# ==============================================================================

def main():
    """Runs ranking pipeline and prints explanations for the top candidates."""
    from pathlib import Path
    from ranker import rank_from_files

    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "sample_data" / "job_description.txt"
    resumes_dir = base_dir / "sample_data" / "resumes"

    print("=" * 80)
    print("EXPLAINABILITY MODULE - TOP 3 CANDIDATE EXPLANATIONS")
    print("=" * 80)

    # 1. Run ranking
    ranked_df = rank_from_files(jd_path, resumes_dir)

    # 2. Generate explanations
    top3 = generate_top3_explanations(ranked_df)

    # 3. Print explanations
    for rank_idx, (candidate, exp) in enumerate(top3.to_dict().items(), start=1):
        print(f"\n[RANK #{rank_idx}] {candidate}")
        print("-" * 80)
        print(exp)

    print("\n" + "=" * 80)
    print("Explanation verification completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
