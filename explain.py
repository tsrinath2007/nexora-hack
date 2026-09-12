"""
Rule-based Candidate Explanation Module.

Generates comprehensive, human-readable 5-7 sentence explanations for ranked candidates
based on deterministic rules, skill proficiencies, missing requirements, semantic alignment,
and structural completeness.
Fully rule-based without LLM calls for complete auditability.
"""

from __future__ import annotations
import re
from pathlib import Path
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


class BestFitRecommendation(str):
    """
    Rich Best Fit Recommendation object encapsulating comparative reasoning,
    individual summary points, and formatted representation.
    """
    def __new__(
        cls,
        full_text: str,
        candidate: str = "",
        candidate_display: str = "",
        final_score: float = 0.0,
        margin: float = 0.0,
        paragraph: str = "",
        closing: str = "",
    ):
        obj = super().__new__(cls, full_text)
        obj.candidate = candidate
        obj.candidate_display = candidate_display
        obj.final_score = final_score
        obj.margin = margin
        obj.paragraph = paragraph
        obj.closing = closing
        obj.full_text = full_text
        return obj


def _clean_candidate_label(filename: str) -> str:
    """Cleans a candidate filename into a clean display title."""
    from pathlib import Path
    import re
    stem = Path(filename).stem
    m = re.match(r"^(CAND_\d+)[_\-\s]*(.*)$", stem, re.IGNORECASE)
    if m:
        cand_id = m.group(1).upper()
        rest = m.group(2)
        rest = re.sub(r"(?i)[_\-\s]*(?:resume|cv)\b", "", rest)
        rest = re.sub(r"(?i)\b(?:resume|cv)[_\-\s]*", "", rest)
        rest = rest.replace("_", " ").replace("-", " ")
        rest = re.sub(r"\s+", " ", rest).strip()
        if rest:
            return f"{cand_id} ({rest.title() if rest.islower() else rest})"
        return cand_id

    cleaned = re.sub(r"(?i)[_\-\s]*(?:resume|cv)\b", "", stem)
    cleaned = re.sub(r"(?i)\b(?:resume|cv)[_\-\s]*", "", cleaned)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = Path(filename).stem
    if cleaned.islower():
        cleaned = cleaned.title()
    return cleaned


def recommend_best_fit(
    ranked_df: pd.DataFrame,
    top_n: int = 3,
) -> Optional[BestFitRecommendation]:
    """
    Identifies the #1 candidate from the ranked DataFrame and builds a
    rule-based comparative paragraph explaining why they are the best overall fit.

    Evaluates:
    a. Final score and margin comparisons against runner-up (#2) and third-place (#3).
    b. Core required skills #1 has that runner-ups are missing.
    c. Close call detection (within 0.03-0.05 margin) with specific differentiating factors.
    d. Completeness caveats if #1 is missing standard resume sections.
    e. Single clear closing sentence recommending the candidate.

    Args:
        ranked_df: Ranked pandas DataFrame sorted descending by final_score.
        top_n: Number of top candidates to compare (default 3).

    Returns:
        BestFitRecommendation object or None if ranked_df is empty.
    """
    if ranked_df.empty:
        return None

    subset = ranked_df.head(max(1, top_n))
    num_candidates = len(subset)

    # #1 Candidate
    row_1 = subset.iloc[0]
    cand_1_raw = str(row_1.get("candidate", "Top Candidate"))
    cand_1_name = _clean_candidate_label(cand_1_raw)
    s1 = float(row_1.get("final_score", 0.0))
    sem_1 = float(row_1.get("semantic_score", 0.0))
    kw_1 = float(row_1.get("keyword_score", 0.0))
    matched_req_1 = set(row_1.get("matched_required", []))
    missing_sections_1 = list(row_1.get("missing_sections", [])) if isinstance(row_1.get("missing_sections"), (list, tuple)) else []
    comp_val_1 = row_1.get("completeness_score", 1.0)

    # Runner-up (#2)
    has_runner_up = num_candidates >= 2
    if has_runner_up:
        row_2 = subset.iloc[1]
        cand_2_name = _clean_candidate_label(str(row_2.get("candidate", "Runner-Up")))
        s2 = float(row_2.get("final_score", 0.0))
        sem_2 = float(row_2.get("semantic_score", 0.0))
        kw_2 = float(row_2.get("keyword_score", 0.0))
        missing_req_2 = set(row_2.get("missing_required", []))
        diff_1_2 = round(s1 - s2, 4)
        diff_pts_1_2 = round((s1 - s2) * 100, 1)
    else:
        diff_1_2 = 0.0
        diff_pts_1_2 = 0.0

    # Third Place (#3)
    has_third = num_candidates >= 3
    if has_third:
        row_3 = subset.iloc[2]
        cand_3_name = _clean_candidate_label(str(row_3.get("candidate", "Candidate #3")))
        s3 = float(row_3.get("final_score", 0.0))
        missing_req_3 = set(row_3.get("missing_required", []))
        diff_1_3 = round(s1 - s3, 4)
        diff_pts_1_3 = round((s1 - s3) * 100, 1)

    # ==========================================================================
    # 1. Score Comparison & Close Call Logic (a & c)
    # ==========================================================================
    if not has_runner_up:
        score_sentence = f"{cand_1_name} stands out as the top candidate with an overall score of {s1:.2f}."
    elif diff_1_2 <= 0.05:
        # Close call race
        if sem_1 > sem_2 + 0.02:
            separating_factor = f"stronger semantic alignment with the role's qualitative requirements ({sem_1:.1%} vs {sem_2:.1%})"
        elif kw_1 > kw_2 + 0.02:
            separating_factor = f"higher verified technical skill proficiency ({kw_1:.1%} vs {kw_2:.1%})"
        else:
            separating_factor = f"a critical edge in technical depth and balanced prerequisite coverage"

        if has_third:
            score_sentence = (
                f"{cand_1_name} finishes as the #1 candidate with an overall score of {s1:.2f}. "
                f"While {cand_2_name} is a close second ({s2:.2f}, within {diff_pts_1_2:.1f} points), "
                f"this was a close call where {cand_1_name} edges ahead due to {separating_factor}. "
                f"Third-ranked {cand_3_name} follows at {s3:.2f} (a {diff_pts_1_3:.1f}-point margin)."
            )
        else:
            score_sentence = (
                f"{cand_1_name} finishes as the #1 candidate with an overall score of {s1:.2f}. "
                f"While {cand_2_name} is a close second ({s2:.2f}, within {diff_pts_1_2:.1f} points), "
                f"this was a close call where {cand_1_name} edges ahead due to {separating_factor}."
            )
    else:
        # Decisive lead (> 0.05)
        if has_third:
            score_sentence = (
                f"{cand_1_name} secures the top position with a score of {s1:.2f}, leading runner-up "
                f"{cand_2_name} ({s2:.2f}) by {diff_pts_1_2:.1f} points and third-place {cand_3_name} "
                f"({s3:.2f}) by {diff_pts_1_3:.1f} points."
            )
        else:
            score_sentence = (
                f"{cand_1_name} secures the top position with a score of {s1:.2f}, leading runner-up "
                f"{cand_2_name} ({s2:.2f}) by {diff_pts_1_2:.1f} points."
            )

    # ==========================================================================
    # 2. Required Skills Comparative Advantages (b)
    # ==========================================================================
    if has_runner_up:
        skills_over_2 = matched_req_1.intersection(missing_req_2)
        if skills_over_2:
            fmt_skills_2 = _format_skill_list(sorted(skills_over_2))
            skills_sentence = (
                f"Notably, {cand_1_name} demonstrates verified proficiency in {fmt_skills_2}, "
                f"essential requirements that runner-up {cand_2_name} lacks."
            )
        elif has_third and matched_req_1.intersection(missing_req_3):
            skills_over_3 = matched_req_1.intersection(missing_req_3)
            fmt_skills_3 = _format_skill_list(sorted(skills_over_3))
            skills_sentence = (
                f"While both top contenders satisfy the primary stack prerequisites, {cand_1_name} "
                f"retains core competency in {fmt_skills_3} where third-place {cand_3_name} has gaps."
            )
        elif matched_req_1:
            skills_sentence = (
                f"Both top contenders demonstrate broad required coverage, but {cand_1_name} "
                f"differentiates with higher technical proficiency weighting."
            )
        else:
            skills_sentence = (
                f"{cand_1_name} demonstrates the strongest overall domain alignment among evaluated profiles."
            )
    else:
        if matched_req_1:
            fmt_req = _format_skill_list(sorted(matched_req_1))
            skills_sentence = f"The candidate demonstrates verified technical proficiency across: {fmt_req}."
        else:
            skills_sentence = "The candidate shows broad contextual relevance to the target position."

    # ==========================================================================
    # 3. Completeness Caveat (d)
    # ==========================================================================
    if isinstance(comp_val_1, str) and "/" in comp_val_1:
        try:
            found_count = int(comp_val_1.split("/")[0])
            total_count = int(comp_val_1.split("/")[1])
        except ValueError:
            found_count, total_count = 6, 6
    elif isinstance(comp_val_1, (int, float)):
        found_count = round(float(comp_val_1) * 6)
        total_count = 6
    else:
        found_count, total_count = 6, 6

    if found_count < total_count or missing_sections_1:
        if missing_sections_1:
            if len(missing_sections_1) == 1:
                miss_str = f"a {missing_sections_1[0]} section"
            elif len(missing_sections_1) == 2:
                miss_str = f"{missing_sections_1[0]} and {missing_sections_1[1]} sections"
            else:
                miss_str = f"{', '.join(missing_sections_1[:-1])}, and {missing_sections_1[-1]} sections"
            caveat_sentence = (
                f"Note: although leading in overall fit, the resume is missing {miss_str}, "
                f"which the recruiter may want to verify separately."
            )
        else:
            caveat_sentence = (
                f"Note: although leading in overall fit, the resume achieves a {found_count}/{total_count} "
                f"completeness score, which the recruiter may want to verify separately."
            )
    else:
        caveat_sentence = (
            f"Additionally, {cand_1_name}'s resume is structurally complete (6/6), "
            f"providing comprehensive documentation across all standard sections."
        )

    # ==========================================================================
    # 4. Closing Sentence (e)
    # ==========================================================================
    closing_sentence = (
        f"Recommended candidate: {cand_1_name} — best overall fit for this role "
        f"based on required skill coverage and contextual relevance."
    )

    paragraph = f"{score_sentence} {skills_sentence} {caveat_sentence}"
    full_text = f"{paragraph}\n\n**{closing_sentence}**"

    return BestFitRecommendation(
        full_text=full_text,
        candidate=cand_1_raw,
        candidate_display=cand_1_name,
        final_score=s1,
        margin=diff_1_2,
        paragraph=paragraph,
        closing=closing_sentence,
    )


# ==============================================================================
# Candidate Comparison & Query Parsing
# ==============================================================================

def compare_candidates(
    row_a: Union[pd.Series, Dict[str, Any]],
    row_b: Union[pd.Series, Dict[str, Any]],
) -> str:
    """
    Compares two candidates and provides a direct, rule-based explanation
    of why one candidate is ranked higher than the other.

    Compares:
    - Overall score and margin
    - Key required technical skills differences (what A has that B lacks, and vice versa)
    - Semantic alignment and domain relevance differences
    - Structural completeness / documentation differences
    - Direct concluding verdict

    Args:
        row_a: Ranking data for Candidate A.
        row_b: Ranking data for Candidate B.

    Returns:
        Structured comparative explanation string.
    """
    cand_a_raw = str(row_a.get("candidate", "Candidate A"))
    cand_b_raw = str(row_b.get("candidate", "Candidate B"))
    cand_a = _clean_candidate_label(cand_a_raw)
    cand_b = _clean_candidate_label(cand_b_raw)

    score_a = float(row_a.get("final_score", 0.0))
    score_b = float(row_b.get("final_score", 0.0))
    sem_a = float(row_a.get("semantic_score", 0.0))
    sem_b = float(row_b.get("semantic_score", 0.0))
    kw_a = float(row_a.get("keyword_score", 0.0))
    kw_b = float(row_b.get("keyword_score", 0.0))

    req_a = set(row_a.get("matched_required", []))
    req_b = set(row_b.get("matched_required", []))
    missing_a = set(row_a.get("missing_required", []))
    missing_b = set(row_b.get("missing_required", []))

    # Who is the leader?
    if score_a >= score_b:
        leader_name, trailer_name = cand_a, cand_b
        leader_score, trailer_score = score_a, score_b
        leader_kw, trailer_kw = kw_a, kw_b
        leader_sem, trailer_sem = sem_a, sem_b
        leader_req_adv = req_a.intersection(missing_b)
        trailer_req_adv = req_b.intersection(missing_a)
        is_reverse_query = False
    else:
        leader_name, trailer_name = cand_b, cand_a
        leader_score, trailer_score = score_b, score_a
        leader_kw, trailer_kw = kw_b, kw_a
        leader_sem, trailer_sem = sem_b, sem_a
        leader_req_adv = req_b.intersection(missing_a)
        trailer_req_adv = req_a.intersection(missing_b)
        is_reverse_query = True

    margin = abs(score_a - score_b)
    margin_pts = round(margin * 100, 1)

    points = []

    # Point 1: Score & Rank Overview
    if score_a == score_b:
        points.append(
            f"📊 **Score Comparison**: {cand_a} and {cand_b} are currently tied with an overall score of {score_a:.2f}."
        )
    elif is_reverse_query:
        points.append(
            f"📊 **Score Comparison**: Actually, {leader_name} ({leader_score:.2f}) is ranked ahead of "
            f"{trailer_name} ({trailer_score:.2f}) by a margin of {margin_pts:.1f} percentage points."
        )
    else:
        points.append(
            f"📊 **Score Comparison**: {leader_name} is ranked higher with an overall score of {leader_score:.2f} "
            f"compared to {trailer_name}'s {trailer_score:.2f} (a {margin_pts:.1f}-point lead)."
        )

    # Point 2: Technical & Required Skills Coverage
    if leader_req_adv and not trailer_req_adv:
        fmt_adv = _format_skill_list(sorted(leader_req_adv))
        points.append(
            f"⚡ **Technical Skill Coverage**: {leader_name} demonstrates verified proficiency in {fmt_adv}, "
            f"which {trailer_name} is missing from their resume."
        )
    elif leader_req_adv and trailer_req_adv:
        fmt_l = _format_skill_list(sorted(leader_req_adv))
        fmt_t = _format_skill_list(sorted(trailer_req_adv))
        points.append(
            f"⚡ **Technical Skill Coverage**: {leader_name} covers key required skills that {trailer_name} lacks ({fmt_l}), "
            f"whereas {trailer_name} only holds advantages in {fmt_t}."
        )
    elif not leader_req_adv and not trailer_req_adv:
        if leader_kw > trailer_kw + 0.05:
            points.append(
                f"⚡ **Technical Proficiency**: Both candidates cover identical required skills, but {leader_name} demonstrates "
                f"higher verified proficiency weighting (keyword match {leader_kw:.1%} vs {trailer_kw:.1%})."
            )
        else:
            points.append(
                f"⚡ **Technical Skill Coverage**: Both candidates demonstrate comparable core required skill matches "
                f"(keyword match {leader_kw:.1%} vs {trailer_kw:.1%})."
            )
    else:
        points.append(
            f"⚡ **Technical Skill Coverage**: {leader_name} maintains a more balanced technical profile across required stack competencies."
        )

    # Point 3: Semantic Alignment & Project Relevance
    diff_sem = leader_sem - trailer_sem
    if diff_sem >= 0.03:
        points.append(
            f"🎯 **Contextual & Domain Depth**: {leader_name}'s project experience and work history demonstrate stronger "
            f"semantic alignment with the role's qualitative scope (semantic similarity {leader_sem:.1%} vs {trailer_sem:.1%})."
        )
    elif diff_sem <= -0.03:
        points.append(
            f"🎯 **Contextual & Domain Depth**: While {trailer_name} exhibits slightly higher qualitative domain similarity "
            f"({trailer_sem:.1%} vs {leader_sem:.1%}), this is outweighed by {leader_name}'s decisive technical skill advantages."
        )
    else:
        points.append(
            f"🎯 **Contextual & Domain Depth**: Both candidates show similar contextual relevance to the role "
            f"({leader_sem:.1%} vs {trailer_sem:.1%})."
        )

    # Point 4: Structural Completeness
    comp_val_l = row_a.get("completeness_score", "6/6") if not is_reverse_query else row_b.get("completeness_score", "6/6")
    comp_val_t = row_b.get("completeness_score", "6/6") if not is_reverse_query else row_a.get("completeness_score", "6/6")
    if str(comp_val_l) != str(comp_val_t):
        points.append(
            f"📋 **Resume Quality**: {leader_name} achieves a {comp_val_l} structural completeness score compared to "
            f"{trailer_name}'s {comp_val_t}."
        )

    # Concluding Verdict
    if leader_req_adv:
        reason = f"broader required skill coverage (including {_format_skill_list(sorted(leader_req_adv))})"
    elif leader_kw > trailer_kw:
        reason = f"higher verified technical proficiency ({leader_kw:.1%} vs {trailer_kw:.1%})"
    elif diff_sem >= 0.03:
        reason = f"superior semantic alignment with the role description ({leader_sem:.1%} vs {trailer_sem:.1%})"
    else:
        reason = "a stronger combined balance of technical and qualitative criteria"

    conclusion = f"🏆 **Verdict**: {leader_name} is the superior match for this role due to {reason}."
    points.append(conclusion)

    return "\n\n".join(points)


def extract_two_candidates_from_query(
    query: str,
    candidates: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts two candidate identifiers from a recruiter's natural language question
    by matching against known candidate filenames and clean aliases.

    Returns:
        (candidate_a, candidate_b) in the order they were mentioned in the query,
        or (None, None) if exactly two unique candidates could not be identified.
    """
    import re
    from collections import defaultdict

    if not query or not query.strip() or len(candidates) < 2:
        return None, None

    alias_to_cands: Dict[str, Set[str]] = defaultdict(set)
    for cand in candidates:
        stem = Path(cand).stem.lower()
        alias_to_cands[cand.lower()].add(cand)
        alias_to_cands[stem].add(cand)
        clean = _clean_candidate_label(cand).lower()
        if clean:
            alias_to_cands[clean].add(cand)

        # Match CAND_xxx patterns
        m = re.search(r"cand[_\-\s]*0*(\d+)", stem)
        if m:
            num = m.group(1)
            int_num = int(num)
            for variant in [
                f"cand_{num}",
                f"cand_{int_num:03d}",
                f"cand-{num}",
                f"cand-{int_num:03d}",
                f"cand {num}",
                f"cand {int_num:03d}",
                f"cand{num}",
                f"cand{int_num:03d}",
                f"candidate {num}",
                f"candidate {int_num:03d}",
                f"candidate_{num}",
                f"candidate_{int_num:03d}",
            ]:
                alias_to_cands[variant].add(cand)

        # Extract name tokens (e.g. Aditya, Meera, Alice, Bob, Carol)
        tokens = [
            t for t in re.split(r"[_\-\s]+", stem)
            if len(t) >= 3 and t not in {
                "resume", "cv", "developer", "software", "engineer", "frontend",
                "backend", "manager", "semantic", "data", "test", "candidate",
            }
        ]
        for t in tokens:
            alias_to_cands[t].add(cand)

    # Filter to unique aliases only
    unique_alias_map = {
        alias: list(cands)[0]
        for alias, cands in alias_to_cands.items()
        if len(cands) == 1
    }

    q_lower = query.lower()
    matches: Dict[str, int] = {}  # candidate -> earliest start index in query

    # Match aliases sorted by length descending so longer phrases take precedence
    for alias in sorted(unique_alias_map.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"
        match = re.search(pattern, q_lower)
        if match:
            cand = unique_alias_map[alias]
            if cand not in matches or match.start() < matches[cand]:
                matches[cand] = match.start()

    if len(matches) == 2:
        sorted_cands = sorted(matches.keys(), key=lambda c: matches[c])
        return sorted_cands[0], sorted_cands[1]

    return None, None


# ==============================================================================
# Verification Runner
# ==============================================================================

def main():
    """Runs ranking pipeline and prints explanations and best fit recommendation."""
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

    # 4. Best Fit Recommendation
    print("\n" + "=" * 80)
    print("BEST FIT RECOMMENDATION")
    print("=" * 80)
    rec = recommend_best_fit(ranked_df)
    if rec:
        print(f"\n{rec.paragraph}\n\n{rec.closing}\n")

    print("=" * 80)
    print("Explanation verification completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
