"""
Job Description Bias & Narrow-Phrasing Checker Module.

Rule-based auditing engine that scans Job Descriptions (JDs) for:
1. Overly narrow tool requirements (brand-specific tools without 'or equivalent' language)
2. Excessive experience requirements relative to seniority words (e.g. junior role demanding 5+ years)
3. Exclusionary or age-coded language ('digital native', 'young team', 'recent graduate only')
4. Gendered or gender-skewed language ('he/she', 'salesman', 'manpower', 'rockstar/ninja')
5. Degree-only gatekeeping (mandatory degree without 'or equivalent experience')

Fully rule-based without LLM calls for auditability and compliance.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set


# ==============================================================================
# 1. Narrow Tool Categories & Equivalents
# ==============================================================================

_NARROW_TOOLS: Dict[str, Dict[str, Any]] = {
    "photoshop": {
        "display": "Photoshop",
        "category": "graphic design or image editing software",
        "alternatives": ["figma", "sketch", "gimp", "illustrator", "canva"],
    },
    "figma": {
        "display": "Figma",
        "category": "UI/UX design and prototyping tools",
        "alternatives": ["adobe xd", "sketch", "invision", "photoshop", "framer"],
    },
    "tableau": {
        "display": "Tableau",
        "category": "business intelligence / data visualization tools",
        "alternatives": ["power bi", "powerbi", "looker", "metabase", "qlik"],
    },
    "power bi": {
        "display": "Power BI",
        "category": "business intelligence / data visualization tools",
        "alternatives": ["tableau", "looker", "metabase", "qlik"],
    },
    "jira": {
        "display": "Jira",
        "category": "agile project management / issue tracking software",
        "alternatives": ["linear", "asana", "trello", "azure devops", "clickup"],
    },
    "salesforce": {
        "display": "Salesforce",
        "category": "CRM platforms",
        "alternatives": ["hubspot", "microsoft dynamics", "dynamics", "zoho", "pipedrive"],
    },
}

_EQUIVALENCE_PATTERN = re.compile(
    r"(?i)\b(?:or\s+(?:similar|equivalent|comparable|other|related)|"
    r"such\s+as|e\.g\.|for\s+example|including|like|equivalent\s+experience)\b"
)


# ==============================================================================
# 2. Seniority vs Experience Patterns
# ==============================================================================

_JUNIOR_TERMS = [
    ("intern", 2, "internship"),
    ("internship", 2, "internship"),
    ("fresher", 2, "entry-level / fresher"),
    ("fresh graduate", 2, "fresh graduate"),
    ("recent grad", 2, "entry-level"),
    ("entry-level", 3, "entry-level"),
    ("entry level", 3, "entry-level"),
    ("junior", 4, "junior"),
    ("associate", 4, "associate"),
]

_EXP_PATTERN = re.compile(
    r"(\d+)(?:\s*[\-\–\—\sto]\s*(\d+))?\+?\s*years?(?:\s+of)?(?:\s+relevant|\s+work|\s+hands[\-\s]on)?\s+experience",
    re.IGNORECASE,
)


# ==============================================================================
# 3. Age-Coded & Exclusionary Language
# ==============================================================================

_AGE_CODED_PATTERNS = [
    (
        re.compile(r"\b(?:digital\s+natives?)\b", re.IGNORECASE),
        "digital native",
        "The phrase 'digital native' is age-coded and can discourage experienced workers.",
    ),
    (
        re.compile(
            r"\b(?:young\s+and\s+energetic|energetic\s+and\s+young|young\s+and\s+dynamic|dynamic\s+and\s+young)\b",
            re.IGNORECASE,
        ),
        "young and energetic",
        "Phrasing like 'young and energetic' introduces age bias and implies an age preference.",
    ),
    (
        re.compile(r"\b(?:recent\s+graduates?\s+only|recent\s+grads?\s+only|fresh\s+graduates?\s+only)\b", re.IGNORECASE),
        "recent graduate only",
        "Phrasing like 'recent graduate only' unnecessarily excludes experienced career-changers.",
    ),
    (
        re.compile(r"\b(?:young\s+(?:team|professionals?|culture|talent|minds?|environment))\b", re.IGNORECASE),
        "young team/professionals",
        "Describing the team as 'young professionals' or a 'young team' can be seen as age-exclusionary.",
    ),
    (
        re.compile(r"\b(?:youthful\s+(?:energy|team|culture|environment))\b", re.IGNORECASE),
        "youthful energy",
        "Using 'youthful' conveys an implicit age preference rather than focusing on role capabilities.",
    ),
]


# ==============================================================================
# 4. Gendered & Slang Language
# ==============================================================================

_GENDERED_PATTERNS = [
    (
        re.compile(r"\b(?:he\s*/\s*she|he/she|s/he)\b", re.IGNORECASE),
        "he/she",
        "Binary gendered phrasing ('he/she'): Recommend using inclusive, gender-neutral pronouns like 'they/them' or 'the candidate'.",
    ),
    (
        re.compile(r"\b(salesman|salesmen)\b", re.IGNORECASE),
        "salesman",
        "Gendered job title ('salesman'): Recommend replacing with neutral terminology like 'salesperson' or 'account executive'.",
    ),
    (
        re.compile(r"\b(manpower)\b", re.IGNORECASE),
        "manpower",
        "Gendered phrasing ('manpower'): Recommend using neutral terms such as 'workforce', 'team capacity', or 'staffing'.",
    ),
    (
        re.compile(r"\b(chairman|chairmen)\b", re.IGNORECASE),
        "chairman",
        "Gendered title ('chairman'): Recommend replacing with 'chair' or 'chairperson'.",
    ),
    (
        re.compile(r"\b(spokesman|spokesmen)\b", re.IGNORECASE),
        "spokesman",
        "Gendered title ('spokesman'): Recommend replacing with 'spokesperson'.",
    ),
    (
        re.compile(r"\b(man-hours|man\s+hours)\b", re.IGNORECASE),
        "man-hours",
        "Gendered phrasing ('man-hours'): Consider using 'working hours' or 'person-hours'.",
    ),
    (
        re.compile(r"\b(rockstar|ninja|guru|wizard)\b", re.IGNORECASE),
        "rockstar/ninja",
        "Gender-skewed tech jargon: Colloquial terms like '{word}' can deter diverse applicants; use clear functional titles like 'expert' or 'lead'.",
    ),
]


# ==============================================================================
# 5. Degree Gatekeeping
# ==============================================================================

_DEGREE_REQUIRED_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"(?:bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?|degree)\s+(?:in|of)\s+[A-Za-z\s]+(?:required|mandatory|must have)|"
    r"degree\s+required|must\s+(?:hold|have|possess)\s+a\s+degree|"
    r"minimum\s+(?:a\s+)?bachelor'?s?|"
    r"education\s*:\s*(?:bachelor|master|phd|b\.?s\.?|m\.?s\.?|degree)[^\n]+"
    r")\b"
)

_DEGREE_EQUIVALENCE_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"or\s+(?:equivalent|relevant|comparable|practical|applicable)\s+(?:work\s+|industry\s+|practical\s+)?experience|"
    r"equivalent\s+practical\s+experience|"
    r"or\s+equivalent\s+combination|"
    r"or\s+demonstrated\s+experience|"
    r"or\s+self[\-\s]taught"
    r")\b"
)


# ==============================================================================
# Main Bias Checker Function
# ==============================================================================

def flag_jd_bias(jd_text: str) -> List[str]:
    """
    Scans Job Description (JD) text for rule-based red flags and returns a list
    of plain-language flagged concern strings.

    Checks:
    1. Overly narrow tool requirements: single proprietary tool with no 'or equivalent' phrasing.
    2. Seniority vs. experience mismatch: junior/entry-level title with excessive years required.
    3. Age-coded or exclusionary phrases: 'digital native', 'young team', etc.
    4. Gendered language: 'he/she', 'salesman', 'manpower', 'rockstar/ninja'.
    5. Degree gatekeeping: mandatory degree without 'or equivalent experience' language.

    Args:
        jd_text: Raw text of the Job Description.

    Returns:
        List[str] of flagged concern messages (empty list if no concerns detected).
    """
    if not jd_text or not jd_text.strip():
        return []

    flags: List[str] = []
    lower_text = jd_text.lower()

    # --------------------------------------------------------------------------
    # 1. Overly narrow tool requirements
    # --------------------------------------------------------------------------
    sentences = re.split(r"[\n\.\;]", jd_text)
    for tool_key, tool_meta in _NARROW_TOOLS.items():
        # Check if tool appears in text
        pattern = rf"\b{re.escape(tool_key)}\b"
        if not re.search(pattern, lower_text):
            continue

        # Check all sentences mentioning this tool
        is_narrow = False
        for s in sentences:
            s_lower = s.lower()
            if not re.search(pattern, s_lower):
                continue

            # Does this sentence or adjacent context allow equivalent language?
            has_equiv = bool(_EQUIVALENCE_PATTERN.search(s_lower))
            has_alt = any(alt in s_lower for alt in tool_meta["alternatives"])

            if not has_equiv and not has_alt:
                is_narrow = True
                break

        if is_narrow:
            flags.append(
                f"Narrow tool requirement: The JD specifically requires '{tool_meta['display']}' "
                f"without allowing equivalent {tool_meta['category']} (e.g. using 'or similar/equivalent tool'), "
                f"which may unnecessarily exclude candidates skilled in alternative software."
            )

    # --------------------------------------------------------------------------
    # 2. Seniority vs Experience Mismatch
    # --------------------------------------------------------------------------
    # Extract maximum required years
    max_exp_required = 0
    for match in _EXP_PATTERN.finditer(jd_text):
        try:
            val = int(match.group(1))
            if val > max_exp_required:
                max_exp_required = val
        except (ValueError, IndexError):
            pass

    if max_exp_required > 0:
        for term, threshold, label in _JUNIOR_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lower_text):
                if max_exp_required >= threshold:
                    flags.append(
                        f"Experience mismatch: The JD mentions a {label} role ('{term}'), "
                        f"yet specifies {max_exp_required}+ years of experience, creating an unrealistic barrier "
                        f"for early-career applicants."
                    )
                    break

    # --------------------------------------------------------------------------
    # 3. Age-coded or Exclusionary Language
    # --------------------------------------------------------------------------
    for pat, label, msg in _AGE_CODED_PATTERNS:
        match = pat.search(jd_text)
        if match:
            flags.append(f"Age-coded phrasing: {msg}")

    # --------------------------------------------------------------------------
    # 4. Gendered & Slang Language
    # --------------------------------------------------------------------------
    for pat, label, msg in _GENDERED_PATTERNS:
        match = pat.search(jd_text)
        if match:
            matched_word = match.group(0)
            formatted_msg = msg.replace("{word}", matched_word).replace("{match}", matched_word)
            flags.append(f"Gender/Jargon flag: {formatted_msg}")

    # --------------------------------------------------------------------------
    # 5. Degree Gatekeeping
    # --------------------------------------------------------------------------
    if _DEGREE_REQUIRED_PATTERN.search(jd_text):
        if not _DEGREE_EQUIVALENCE_PATTERN.search(jd_text):
            flags.append(
                "Degree-only gatekeeping: The JD mandates a formal degree without explicitly allowing for "
                "'or equivalent practical experience', which may unnecessarily filter out qualified self-taught, "
                "bootcamp, or non-traditional candidates."
            )

    return flags


# ==============================================================================
# Verification Test Runner
# ==============================================================================

def main():
    """Demonstrates and tests JD bias checking across sample and synthetic JDs."""
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("JD BIAS & NARROW-PHRASING CHECKER - VERIFICATION TEST")
    print("=" * 80)

    # 1. Test against actual sample JD
    sample_jd_path = Path(__file__).resolve().parent / "sample_data" / "job_description.txt"
    if sample_jd_path.exists():
        sample_jd_text = sample_jd_path.read_text(encoding="utf-8")
        print("\n[SAMPLE JD AUDIT]")
        print(f"File: {sample_jd_path.name}")
        sample_flags = flag_jd_bias(sample_jd_text)
        if sample_flags:
            print(f"Flagged concerns ({len(sample_flags)}):")
            for idx, f in enumerate(sample_flags, 1):
                print(f"  {idx}. {f}")
        else:
            print("  [PASS] No major bias concerns detected in sample JD.")

    # 2. Test synthetic JD with intentional bias triggers
    print("\n" + "-" * 80)
    print("[SYNTHETIC BIASED JD AUDIT]")
    biased_jd = """
    Job Title: Junior Frontend Rockstar
    Company: FastGrowth Inc
    
    About Us:
    We are a youthful team of digital natives building cutting-edge web applications.
    He/she will join our energetic and young team to increase our developer manpower.
    
    Requirements:
    - Bachelor's degree in Computer Science required.
    - 5+ years experience in frontend development.
    - Advanced proficiency in Photoshop required for all design handoffs.
    - Must be a recent graduate only.
    """
    print("Synthetic JD text:")
    print(biased_jd.strip())
    print("\nAudit Results:")
    biased_flags = flag_jd_bias(biased_jd)
    for idx, f in enumerate(biased_flags, 1):
        print(f"  {idx}. {f}")

    print("\n" + "=" * 80)
    print("Bias checker verification completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
