"""
Resume Quality & Completeness Checking Module.

Evaluates resume structural completeness by checking for the presence of 6 essential sections:
1. Education
2. Skills
3. Experience
4. Projects
5. Certifications
6. Contact information (email / phone)

Combines case-insensitive header regex matching and content heuristics.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# ==============================================================================
# 1. Header Detection Patterns & Aliases
# ==============================================================================

# Education: common headings and academic phrasing
_EDUCATION_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:education(?:al\s+background)?|academics?(?:\s+background|\s+qualifications)?|"
    r"degrees?|qualifications?|scholastic\s+achievements?|university|college)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

# Skills: technical, core, and professional skills
_SKILLS_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:(?:technical|core|key|professional)?\s*skills?(?:\s*set)?|technologies|"
    r"tech\s+stack|tools\s*(&|and)\s*technologies|programming\s+languages|"
    r"core\s+competencies|proficiencies|areas\s+of\s+expertise)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

# Experience: work history, employment history, professional experience
_EXPERIENCE_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:(?:work|professional|career|relevant|employment|industry)?\s*experience|"
    r"employment\s+history|work\s+history|professional\s+background|internships?)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

# Projects: personal, academic, and technical projects
_PROJECTS_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:(?:technical|academic|personal|key|selected|recent|open\s+source|side)?\s*projects?|"
    r"project\s+experience|project\s+portfolio|portfolio)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

# Certifications: courses, certificates, licenses, accreditations
_CERTIFICATIONS_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:(?:professional|courses\s*(&|and)\s*|licenses\s*(&|and)\s*)?certifications?|"
    r"licenses?|accreditations?|certified(?:\s+credentials?)?|certificates?)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

# Contact: header cues and content heuristics (email / phone)
_CONTACT_HEADER_RE = re.compile(
    r"(?:^|[\r\n\•\-\*\|\#])\s*"
    r"(?:contact(?:\s+info|\s+information)?|personal\s+(?:details|info|information)|"
    r"get\s+in\s+touch)\s*[:\-\—\–]?",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

_PHONE_PATTERN = re.compile(
    r"(?:\b(?:phone|tel|mobile|cell)\s*[:\-\s]*)?"
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
    re.IGNORECASE,
)


# ==============================================================================
# 2. Result Data Structure
# ==============================================================================

class ResumeCompletenessResult(dict):
    """
    Result dictionary representing resume completeness.
    Allows standard dict key access:
        res["Education"], res["completeness_score"]
    as well as attribute access and helper properties:
        res.completeness_score, res.sections, res.missing_sections
    """
    def __init__(
        self,
        sections: Dict[str, bool],
        completeness_score: float,
        found_count: int,
        total_sections: int = 6,
        details: Optional[Dict[str, Any]] = None,
    ):
        data = dict(sections)
        data["completeness_score"] = completeness_score
        super().__init__(data)
        self.sections = sections
        self.completeness_score = completeness_score
        self.found_count = found_count
        self.total_sections = total_sections
        self.details = details or {}

    @property
    def present_sections(self) -> List[str]:
        """Returns list of sections detected in resume."""
        return [k for k, v in self.sections.items() if v]

    @property
    def missing_sections(self) -> List[str]:
        """Returns list of sections missing from resume."""
        return [k for k, v in self.sections.items() if not v]

    def __repr__(self) -> str:
        return (
            f"ResumeCompletenessResult(\n"
            f"  completeness_score={self.completeness_score:.4f} ({self.found_count}/{self.total_sections}),\n"
            f"  present={self.present_sections},\n"
            f"  missing={self.missing_sections}\n"
            f")"
        )


# ==============================================================================
# 3. Completeness Checker
# ==============================================================================

def check_resume_completeness(resume_text: str) -> ResumeCompletenessResult:
    """
    Checks for the presence of 6 essential sections/content in a resume:
    1. Education
    2. Skills
    3. Experience
    4. Projects
    5. Certifications
    6. Contact information (email / phone)

    Uses header detection (case-insensitive regex for common phrasings and aliases)
    plus content heuristics for contact info (email and phone regex).

    Args:
        resume_text: Raw text of candidate resume.

    Returns:
        ResumeCompletenessResult: A dict with {section: found (bool)}
        and completeness_score = found_count / 6.
    """
    if not resume_text or not resume_text.strip():
        sections = {
            "Education": False,
            "Skills": False,
            "Experience": False,
            "Projects": False,
            "Certifications": False,
            "Contact information": False,
        }
        return ResumeCompletenessResult(
            sections=sections,
            completeness_score=0.0,
            found_count=0,
            total_sections=6,
            details={"has_email": False, "has_phone": False},
        )

    # 1. Header detection for primary sections
    has_education = bool(_EDUCATION_HEADER_RE.search(resume_text))
    has_skills = bool(_SKILLS_HEADER_RE.search(resume_text))
    has_experience = bool(_EXPERIENCE_HEADER_RE.search(resume_text))
    has_projects = bool(_PROJECTS_HEADER_RE.search(resume_text))
    has_certifications = bool(_CERTIFICATIONS_HEADER_RE.search(resume_text))

    # 2. Contact information: header detection OR email regex OR phone regex
    has_contact_header = bool(_CONTACT_HEADER_RE.search(resume_text))
    has_email = bool(_EMAIL_PATTERN.search(resume_text))
    has_phone = bool(_PHONE_PATTERN.search(resume_text))
    has_contact = has_contact_header or has_email or has_phone

    sections: Dict[str, bool] = {
        "Education": has_education,
        "Skills": has_skills,
        "Experience": has_experience,
        "Projects": has_projects,
        "Certifications": has_certifications,
        "Contact information": has_contact,
    }

    found_count = sum(1 for found in sections.values() if found)
    completeness_score = round(found_count / 6.0, 4)

    details = {
        "has_email": has_email,
        "has_phone": has_phone,
        "has_contact_header": has_contact_header,
    }

    return ResumeCompletenessResult(
        sections=sections,
        completeness_score=completeness_score,
        found_count=found_count,
        total_sections=6,
        details=details,
    )


# ==============================================================================
# Verification Runner
# ==============================================================================

def main():
    """Runs verification tests across sample resumes and edge cases."""
    from parser import extract_resumes

    print("=" * 75)
    print("RESUME QUALITY - COMPLETENESS CHECK VERIFICATION")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    resumes_dir = base_dir / "sample_data" / "resumes"
    resumes = extract_resumes(resumes_dir)

    print(f"\n[1] Evaluating Sample Resumes ({len(resumes)} found):")
    print("-" * 75)
    header_cols = ["Education", "Skills", "Exp", "Proj", "Cert", "Contact"]
    print(f"{'Resume Filename':<28} | {' | '.join(header_cols)} | {'Score':<6} | Status")
    print("-" * 75)

    for filename, text in resumes.items():
        res = check_resume_completeness(text)
        s = res.sections
        cols_str = (
            f"{'YES' if s['Education'] else 'NO':^9} | "
            f"{'YES' if s['Skills'] else 'NO':^6} | "
            f"{'YES' if s['Experience'] else 'NO':^3} | "
            f"{'YES' if s['Projects'] else 'NO':^4} | "
            f"{'YES' if s['Certifications'] else 'NO':^4} | "
            f"{'YES' if s['Contact information'] else 'NO':^7}"
        )
        print(f"{filename:<28} | {cols_str} | {res.completeness_score:.2f}   | {res.found_count}/6")

    # Evaluate a 100% complete resume with all 6 sections
    full_resume = """
    David Miller
    Email: david.miller@example.com | Phone: (555) 234-5678

    Work Experience:
    Senior Software Engineer at Nexora Tech (2020 - Present)
    - Architected backend data pipelines and ML inference APIs.

    Technical Skills:
    Python, FastAPI, Docker, Kubernetes, AWS, PyTorch, PostgreSQL

    Personal Projects:
    - Open-source distributed crawler and NLP indexing pipeline.
    - Real-time Streamlit dashboard for telemetry monitoring.

    Education:
    B.S. in Computer Science, University of Washington

    Licenses & Certifications:
    - AWS Certified Solutions Architect - Associate
    - Certified Kubernetes Administrator (CKA)
    """

    print("\n[2] Evaluating 100% Complete Comprehensive Resume:")
    print("-" * 75)
    full_res = check_resume_completeness(full_resume)
    for section, found in full_res.sections.items():
        status = "FOUND" if found else "MISSING"
        print(f"    - {section:<20}: [{status}]")
    print(f"    Completeness Score: {full_res.completeness_score:.2%} ({full_res.found_count}/6)")

    # Evaluate a minimal / sparse resume
    sparse_resume = """
    Jane Doe
    Phone: 555-9876
    Education: B.A. in English
    """
    print("\n[3] Evaluating Sparse / Incomplete Resume:")
    print("-" * 75)
    sparse_res = check_resume_completeness(sparse_resume)
    for section, found in sparse_res.sections.items():
        status = "FOUND" if found else "MISSING"
        print(f"    - {section:<20}: [{status}]")
    print(f"    Completeness Score: {sparse_res.completeness_score:.2%} ({sparse_res.found_count}/6)")

    print("\n" + "=" * 75)
    print("Completeness verification completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
