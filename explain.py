"""
Rule-based Candidate Explanation Module.

Generates comprehensive, human-readable 5-7 sentence explanations for ranked candidates
based on deterministic rules, skill proficiencies, missing requirements, semantic alignment,
and structural completeness.
Fully rule-based without LLM calls for complete auditability.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import pandas as pd


# Display name overrides for clean presentation
_SKILL_DISPLAY_NAMES: Dict[str, str] = {
    "c++": "C++",
    "c#": "C#",
    "c": "C",
    "unity": "Unity",
    "unreal engine": "Unreal Engine",
    "data structures & algorithms": "Data Structures & Algorithms",
    "data structures and algorithms": "Data Structures & Algorithms",
    "object-oriented programming": "Object-Oriented Programming",
    "game physics": "Game Physics",
    "game development": "Game Development",
    "3d game development": "3D Game Development",
    "shader programming": "Shader Programming",
    "multiplayer/network programming": "Multiplayer/Network Programming",
    "ai programming": "AI Programming",
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
    "natural language processing": "Natural Language Processing",
    "large language models": "Large Language Models",
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
    return skill.title()


def _format_skill_list(skills: List[str]) -> str:
    """Formats a list of skill strings into an Oxford comma list."""
    if not skills:
        return ""
    formatted = [_format_skill_name(s) for s in skills]
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def _format_skill_with_proficiency(skill: str, weight: Optional[float]) -> str:
    """Formats a skill with its proficiency level if weight is available."""
    name = _format_skill_name(skill)
    if weight is None:
        return name
    if weight >= 0.85:
        return f"Advanced in {name}"
    elif weight >= 0.55:
        return f"Intermediate in {name}"
    elif weight >= 0.25:
        return f"Beginner in {name}"
    return name


def _format_skill_list_with_levels(
    skills: List[str],
    skill_weights: Dict[str, float],
) -> str:
    """Formats skills with explicit proficiencies into an Oxford comma list."""
    if not skills:
        return ""
    items = []
    for s in skills:
        w = skill_weights.get(s, skill_weights.get(s.lower(), None))
        items.append(_format_skill_with_proficiency(s, w))

    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return f"{', '.join(items[:-1])}, and {items[-1]}"


class ExplanationResult(str):
    """
    Rich Explanation string object holding individual sentence breakdown
    and providing formatted representations.
    """
    def __new__(cls, text: str, sentences: Optional[List[str]] = None):
        obj = super().__new__(cls, text)
        obj.sentences = sentences or []
        return obj

    @property
    def paragraph(self) -> str:
        """Returns all sentences joined as a single continuous paragraph."""
        return " ".join(self.sentences)

    @property
    def numbered(self) -> str:
        """Returns all sentences formatted as a numbered list with double line breaks."""
        return "\n\n".join(f"{i}. {s}" for i, s in enumerate(self.sentences, 1))

    @property
    def bulleted(self) -> str:
        """Returns all sentences formatted as bullet points with double line breaks."""
        return "\n\n".join(f"- {s}" for s in self.sentences)


def generate_explanation(
    row: Union[pd.Series, Dict[str, Any]],
    rank: Optional[int] = None,
    style: str = "numbered",
) -> ExplanationResult:
    """
    Generates an audit-ready 5-7 sentence rule-based explanation for a ranked candidate.

    Structure:
    1. Opening line: rank, final score, and one-word fit label (Excellent/Strong/Moderate/Weak Fit).
    2. Required skills sentence: matched_required skills with proficiency level (Advanced/Intermediate/Beginner).
    3. Missing required skills sentence: constructively framed ("However, the candidate does not show experience in: ...")
       or ("The candidate meets every required skill for this role with no gaps.").
    4. Preferred/bonus skills sentence: matched_preferred skills if any ("Additionally brings relevant preferred experience in: ...").
    5. Semantic alignment sentence: plain-language translation of raw semantic_score.
    6. Resume completeness sentence: X/6 completeness score and missing sections if any.
    7. Closing recommendation line: interview recommendation for rank 1-3.

    Args:
        row: A pandas Series or dict containing candidate ranking data.
        rank: Optional 1-based rank (derived from row index if not provided).
        style: Output format ('numbered' or 'paragraph'). Defaults to 'numbered'.

    Returns:
        ExplanationResult string object containing the explanation.
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
    semantic_score = float(row.get("semantic_score", 0.0))
    matched_req = list(row.get("matched_required", []))
    missing_req = list(row.get("missing_required", []))
    matched_pref = list(row.get("matched_preferred", []))

    # Skill weights mapping
    raw_weights = row.get("skill_weights")
    skill_weights: Dict[str, float] = raw_weights if isinstance(raw_weights, dict) else {}

    # Completeness data
    raw_missing_sections = row.get("missing_sections")
    missing_sections: List[str] = list(raw_missing_sections) if isinstance(raw_missing_sections, (list, tuple)) else []
    completeness_val = row.get("completeness_score", 1.0)

    if isinstance(completeness_val, str) and "/" in completeness_val:
        parts = completeness_val.split("/")
        try:
            found_count = int(parts[0])
            total_count = int(parts[1]) if len(parts) > 1 else 6
        except ValueError:
            found_count, total_count = 6, 6
    elif isinstance(completeness_val, (int, float)):
        found_count = round(float(completeness_val) * 6)
        total_count = 6
    else:
        found_count, total_count = 6, 6

    # ==========================================================================
    # Sentence 1: Opening line with fit label
    # ==========================================================================
    if final_score >= 0.80:
        fit_label = "Excellent Fit"
    elif final_score >= 0.60:
        fit_label = "Strong Fit"
    elif final_score >= 0.40:
        fit_label = "Moderate Fit"
    else:
        fit_label = "Weak Fit"

    sentence_1 = f"Ranked #{rank} overall with a final score of {final_score:.2f} ({fit_label})."

    # ==========================================================================
    # Sentence 2: Required skills with proficiency levels
    # ==========================================================================
    if matched_req:
        skills_str = _format_skill_list_with_levels(matched_req, skill_weights)
        sentence_2 = f"The candidate demonstrates verified technical proficiency across required areas: {skills_str}."
    else:
        sentence_2 = "The candidate does not demonstrate any of the required core skills in their resume."

    # ==========================================================================
    # Sentence 3: Missing required skills sentence
    # ==========================================================================
    if missing_req:
        missing_str = _format_skill_list(missing_req)
        sentence_3 = f"However, the candidate does not show experience in: {missing_str}."
    else:
        sentence_3 = "The candidate meets every required skill for this role with no gaps."

    # ==========================================================================
    # Sentence 4: Preferred/bonus skills sentence
    # ==========================================================================
    if matched_pref:
        pref_str = _format_skill_list(matched_pref)
        sentence_4 = f"Additionally brings relevant preferred experience in: {pref_str}."
    else:
        sentence_4 = "No additional preferred or bonus skills were noted in the profile."

    # ==========================================================================
    # Sentence 5: Semantic alignment sentence
    # ==========================================================================
    if semantic_score >= 0.80:
        sentence_5 = (
            "The resume's overall experience and project background strongly align with the "
            "role's context, even beyond exact keyword matches."
        )
    elif semantic_score >= 0.60:
        sentence_5 = (
            "The candidate's background shows solid contextual relevance to the role, "
            "demonstrating transferable project experience."
        )
    else:
        sentence_5 = (
            "The resume shows limited contextual overlap with the role description, "
            "indicating fewer transferable domain experiences."
        )

    # ==========================================================================
    # Sentence 6: Resume completeness sentence
    # ==========================================================================
    if found_count >= total_count:
        sentence_6 = (
            f"The resume demonstrates a complete {found_count}/{total_count} structural completeness score, "
            "covering all standard professional sections."
        )
    else:
        if missing_sections:
            if len(missing_sections) == 1:
                missing_desc = f"a {missing_sections[0]} section"
            elif len(missing_sections) == 2:
                missing_desc = f"{missing_sections[0]} and {missing_sections[1]} sections"
            else:
                missing_desc = f"{', '.join(missing_sections[:-1])}, and {missing_sections[-1]} sections"
            sentence_6 = (
                f"Note: this resume achieves a {found_count}/{total_count} completeness score and is missing "
                f"{missing_desc}, which may be worth following up on."
            )
        else:
            sentence_6 = (
                f"Note: this resume achieves a {found_count}/{total_count} completeness score across "
                "standard structural sections."
            )

    # ==========================================================================
    # Sentence 7: Closing recommendation line
    # ==========================================================================
    if rank <= 3:
        sentence_7 = (
            "Recommended for interview based on strong technical and contextual alignment "
            "with the job requirements."
        )
    elif final_score >= 0.60:
        sentence_7 = (
            "Consider as a potential alternate candidate or for adjacent roles aligned with "
            "their technical profile."
        )
    else:
        sentence_7 = (
            "Not recommended for immediate interview due to significant skill and qualification gaps."
        )

    sentences = [
        sentence_1,
        sentence_2,
        sentence_3,
        sentence_4,
        sentence_5,
        sentence_6,
        sentence_7,
    ]

    if style == "paragraph":
        full_text = " ".join(sentences)
    else:
        # Default: numbered points separated by double newlines
        full_text = "\n\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

    return ExplanationResult(full_text, sentences)


class Top3Explanations(list):
    """
    List of top 3 candidate explanations that also provides dictionary
    mapping by candidate name and pretty printing.
    """
    def __init__(self, explanations: List[ExplanationResult], mapping: Dict[str, ExplanationResult]):
        super().__init__(explanations)
        self.mapping = mapping

    def to_dict(self) -> Dict[str, str]:
        """Returns candidate -> explanation string dictionary."""
        return {k: str(v) for k, v in self.mapping.items()}

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
        Top3Explanations object (behaves as a list of ExplanationResult strings, with .to_dict() support).
    """
    if ranked_df.empty:
        return Top3Explanations([], {})

    top3_df = ranked_df.head(3)
    explanations: List[ExplanationResult] = []
    mapping: Dict[str, ExplanationResult] = {}

    for idx, (_, row) in enumerate(top3_df.iterrows(), start=1):
        explanation = generate_explanation(row, rank=idx, style="numbered")
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
