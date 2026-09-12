"""
Keyword-based matching module for Resume Ranker.

Features:
- Predefined tech-skill taxonomy (~160+ canonical skills) and comprehensive alias mapping.
- Exact alias-normalized regex matching with special character support (c++, c#, .net, etc.).
- RapidFuzz fuzzy matching for typo tolerance (threshold ~85).
- JD extraction separating required vs preferred skills based on section/cue proximity.
- Resume skill extraction with canonical normalization.
- Keyword scoring with 0.7 required / 0.3 preferred weighting.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple, Any
from rapidfuzz import fuzz, process


# ==============================================================================
# 1. Tech-Skill Taxonomy (~160+ Canonical Skills) & Aliases
# ==============================================================================

TECH_SKILLS: Set[str] = {
    # Programming Languages
    "python", "javascript", "typescript", "java", "c", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "perl", "shell",
    "bash", "powershell", "sql", "html", "css", "sass", "lua", "julia", "matlab",
    "groovy", "solidity", "elixir",

    # Web & Application Frameworks / Libraries
    "react", "react native", "next.js", "vue.js", "nuxt.js", "angular", "svelte",
    "tailwind css", "bootstrap", "redux", "mobx", "graphql", "rest api", "webpack",
    "vite", "node.js", "express.js", "django", "flask", "fastapi", "spring boot",
    "asp.net", "ruby on rails", "laravel", "nestjs", "flutter", "electron", "jquery",

    # Databases, Caching & Data Stores
    "postgresql", "mysql", "mongodb", "redis", "sqlite", "cassandra", "elasticsearch",
    "opensearch", "neo4j", "dynamodb", "oracle", "mariadb", "couchdb", "firebase",
    "supabase", "snowflake", "bigquery", "clickhouse", "pinecone", "chroma", "milvus",
    "weaviate", "qdrant", "faiss", "cockroachdb", "hive", "hbase",

    # Cloud, DevOps & Infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "github actions", "gitlab ci", "linux", "nginx", "apache", "prometheus", "grafana",
    "kafka", "rabbitmq", "helm", "vagrant", "circleci", "datadog", "serverless", "git",
    "ci/cd", "argo cd", "istio", "openstack", "cloudflare", "pulumi",

    # Machine Learning, AI & Data Science
    "pytorch", "tensorflow", "keras", "scikit-learn", "hugging face", "transformers",
    "sentence-transformers", "langchain", "llamaindex", "pandas", "numpy", "scipy",
    "opencv", "matplotlib", "seaborn", "spacy", "nltk", "xgboost", "lightgbm", "catboost",
    "apache spark", "hadoop", "dbt", "apache airflow", "databricks", "tableau",
    "power bi", "mlops", "mlflow", "wandb", "deep learning", "machine learning",
    "computer vision", "natural language processing", "large language models",
    "generative ai", "vector search", "reinforcement learning",

    # Software Engineering, Architecture & Testing
    "microservices", "unit testing", "pytest", "selenium", "cypress", "junit",
    "agile", "scrum", "jira", "api design", "distributed systems", "websocket",
    "grpc", "message queues", "system design"
}

# Alias dictionary mapping abbreviations, common spellings, and variations to canonical skill
SKILL_ALIASES: Dict[str, str] = {
    # Languages
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "py3": "python",
    "python3": "python",
    "golang": "go",
    "cplusplus": "c++",
    "cpp": "c++",
    "c#": "c#",
    "csharp": "c#",
    "c sharp": "c#",
    "rb": "ruby",
    "sh": "shell",
    "postgres": "postgresql",
    "postgresql db": "postgresql",
    "psql": "postgresql",
    "mysql db": "mysql",
    "mongo": "mongodb",
    "mongodb database": "mongodb",
    
    # Web & Frameworks
    "node": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "react-native": "react native",
    "reactnative": "react native",
    "vue": "vue.js",
    "vuejs": "vue.js",
    "angularjs": "angular",
    "next": "next.js",
    "nextjs": "next.js",
    "nuxt": "nuxt.js",
    "nuxtjs": "nuxt.js",
    "nest": "nestjs",
    "nest.js": "nestjs",
    "express": "express.js",
    "expressjs": "express.js",
    "spring": "spring boot",
    "springboot": "spring boot",
    "django rest framework": "django",
    "drf": "django",
    "tailwind": "tailwind css",
    "tailwindcss": "tailwind css",
    "fast-api": "fastapi",
    ".net": "asp.net",
    "dotnet": "asp.net",

    # Cloud & DevOps
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "amazon web services": "aws",
    "amazon aws": "aws",
    "aws cloud": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "azure cloud": "azure",
    "gh actions": "github actions",
    "github action": "github actions",
    "gitlab": "gitlab ci",
    "docker compose": "docker",
    "docker-compose": "docker",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",

    # ML, AI & Data Science
    "sklearn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "tf": "tensorflow",
    "huggingface": "hugging face",
    "hf": "hugging face",
    "sentence transformers": "sentence-transformers",
    "sentencetransformers": "sentence-transformers",
    "spark": "apache spark",
    "pyspark": "apache spark",
    "airflow": "apache airflow",
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "ml": "machine learning",
    "dl": "deep learning",
    "rl": "reinforcement learning",
    "powerbi": "power bi",
    "vector database": "vector search",
    "vector db": "vector search",
    "vector embeddings": "vector search",
    "embeddings": "vector search",
    "elastic search": "elasticsearch",

    # Other
    "restful": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "unit tests": "unit testing",
}

# Auto-register every canonical skill as mapping to itself
for skill in TECH_SKILLS:
    SKILL_ALIASES[skill.lower()] = skill

# Sorted aliases by length descending so longer phrases match before substrings
SORTED_ALIASES: List[Tuple[str, str]] = sorted(
    SKILL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
)

# Precompile regex boundary patterns for all aliases
_ALIAS_PATTERNS: List[Tuple[re.Pattern, str]] = []
for alias_key, canonical_skill in SORTED_ALIASES:
    escaped = re.escape(alias_key)
    # Match boundary that respects symbols like +, #, ., /
    pattern = re.compile(rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])", re.IGNORECASE)
    _ALIAS_PATTERNS.append((pattern, canonical_skill))

# Targets for fuzzy matching: only multi-character canonical skills (>= 4 chars)
# to avoid false positive matches on short acronyms like 'c', 'r', 'go', 'js', 'aws'.
_FUZZY_TARGETS: List[str] = [skill for skill in TECH_SKILLS if len(skill) >= 4]

# Stopwords to ignore during candidate token extraction
_STOP_WORDS: Set[str] = {
    "with", "from", "that", "this", "have", "been", "will", "your", "they", "their",
    "work", "years", "year", "team", "role", "help", "good", "need", "plus", "time",
    "date", "name", "city", "state", "user", "build", "lead", "high", "well", "more",
    "most", "each", "both", "must", "some", "such", "than", "then", "into", "over",
    "also", "part", "self", "best", "like", "other", "about", "using", "used", "strong",
    "experience", "experienced", "skills", "tools", "looking", "candidate", "responsibilities",
    "requirements", "qualifications", "education", "phone", "email", "proficient", "proficiency",
    "familiarity", "knowledge", "design", "develop", "developing", "implement", "implementing",
    "scale", "scalable", "applications", "frameworks", "pipelines", "models", "automated"
}


# ==============================================================================
# Helper Result Classes
# ==============================================================================

class JDKeywordsResult(tuple):
    """
    Result of extract_keywords_from_jd.
    Supports tuple unpacking:
        jd_required, jd_preferred = extract_keywords_from_jd(text)
    as well as attribute access and dict-like key access:
        res.required_skills, res.preferred_skills
        res["required_skills"], res["preferred_skills"]
    """
    def __new__(cls, required_skills: Set[str], preferred_skills: Set[str]):
        return super().__new__(cls, (required_skills, preferred_skills))

    @property
    def required_skills(self) -> Set[str]:
        return self[0]

    @property
    def preferred_skills(self) -> Set[str]:
        return self[1]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in ("required_skills", "required"):
                return self.required_skills
            elif item in ("preferred_skills", "preferred"):
                return self.preferred_skills
            raise KeyError(f"Invalid key '{item}'. Use 'required_skills' or 'preferred_skills'.")
        return super().__getitem__(item)

    def __repr__(self) -> str:
        return (
            f"JDKeywordsResult(\n"
            f"  required_skills={sorted(self.required_skills)},\n"
            f"  preferred_skills={sorted(self.preferred_skills)}\n"
            f")"
        )


class KeywordScoreResult(dict):
    """
    Result of keyword_score.
    Provides dictionary access, attribute access, and tuple unpacking:
        res.score, res.matched_required, res.missing_required, res.matched_preferred
        res["score"], res["matched_required"], etc.
        score, matched_req, missing_req, matched_pref = keyword_score(...)
    """
    def __init__(
        self,
        score: float,
        matched_required: List[str],
        missing_required: List[str],
        matched_preferred: List[str],
    ):
        super().__init__(
            score=score,
            matched_required=matched_required,
            missing_required=missing_required,
            matched_preferred=matched_preferred,
        )
        self.score = score
        self.matched_required = matched_required
        self.missing_required = missing_required
        self.matched_preferred = matched_preferred

    def __iter__(self):
        # Enables tuple unpacking: score, matched_req, missing_req, matched_pref = keyword_score(...)
        return iter((self.score, self.matched_required, self.missing_required, self.matched_preferred))

    def __repr__(self) -> str:
        return (
            f"KeywordScoreResult(\n"
            f"  score={self.score:.4f},\n"
            f"  matched_required={self.matched_required},\n"
            f"  missing_required={self.missing_required},\n"
            f"  matched_preferred={self.matched_preferred}\n"
            f")"
        )


# ==============================================================================
# 2. Skill Extraction Core Logic
# ==============================================================================

def extract_keywords_from_text(text: str, fuzzy_threshold: float = 85.0) -> Set[str]:
    """
    Extracts canonical skills from text using:
    1. Case-insensitive alias-normalized matching (exact regex).
    2. RapidFuzz fuzzy matching (score >= fuzzy_threshold) for candidate tokens/phrases.

    Args:
        text: Raw text to extract skills from.
        fuzzy_threshold: Minimum RapidFuzz match ratio (0-100), default 85.0.

    Returns:
        A set of canonical skill names found in the text.
    """
    if not text:
        return set()

    found_skills: Set[str] = set()

    # Step 1: Exact alias-normalized regex matching
    for pattern, canonical_skill in _ALIAS_PATTERNS:
        if pattern.search(text):
            found_skills.add(canonical_skill)

    # Step 2: RapidFuzz fuzzy matching for typos/variants
    # Extract candidate 1-gram, 2-gram, and 3-gram tokens from text
    cleaned = re.sub(r"[^\w\s\-\.]", " ", text)
    raw_tokens = cleaned.split()

    candidate_phrases: Set[str] = set()

    for idx, tok in enumerate(raw_tokens):
        cleaned_tok = tok.strip(".-_").lower()
        if len(cleaned_tok) >= 4 and cleaned_tok not in _STOP_WORDS:
            candidate_phrases.add(cleaned_tok)

        # 2-grams
        if idx + 1 < len(raw_tokens):
            w2 = f"{raw_tokens[idx]} {raw_tokens[idx+1]}".strip(".-_").lower()
            if len(w2) >= 4:
                candidate_phrases.add(w2)

        # 3-grams
        if idx + 2 < len(raw_tokens):
            w3 = f"{raw_tokens[idx]} {raw_tokens[idx+1]} {raw_tokens[idx+2]}".strip(".-_").lower()
            if len(w3) >= 5:
                candidate_phrases.add(w3)

    for candidate in candidate_phrases:
        match = process.extractOne(
            candidate,
            _FUZZY_TARGETS,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        )
        if match:
            matched_skill = match[0]
            found_skills.add(matched_skill)

    return found_skills


# ==============================================================================
# 3. JD Extraction (Required vs Preferred)
# ==============================================================================

_REQ_CUES_PATTERN = re.compile(
    r"\b(required|requirements?|must\s+have|must-have|must\s+possess|"
    r"proficien(?:t|cy)\s+in|proficien(?:t|cy)\s+with|essential|mandatory|"
    r"minimum\s+qualifications?|basic\s+qualifications?|minimum\s+requirements?|"
    r"qualifications?|what\s+you(?:\'ll)?\s+need|what\s+we\s+are\s+looking\s+for)\b",
    re.IGNORECASE,
)

_PREF_CUES_PATTERN = re.compile(
    r"\b(preferred|preferences?|preferred\s+qualifications?|nice\s+to\s+have|"
    r"nice-to-have|bonus|plus|good\s+to\s+have|desired|desirable|optional)\b",
    re.IGNORECASE,
)

_REQ_SECTION_HEADER = re.compile(
    r"^(requirements?|qualifications?|must\s+haves?|what\s+you(?:\'ll)?\s+need|"
    r"minimum\s+requirements?|essential\s+skills?|what\s+we(?:\'re)?\s+looking\s+for)",
    re.IGNORECASE,
)

_PREF_SECTION_HEADER = re.compile(
    r"^(preferred|nice\s+to\s+have|bonus|desired|desirable|optional|good\s+to\s+have)",
    re.IGNORECASE,
)

_OTHER_SECTION_HEADER = re.compile(
    r"^(responsibilities|overview|about|duties|what\s+you(?:\'ll)?\s+do|summary)",
    re.IGNORECASE,
)


def extract_keywords_from_jd(jd_text: str, fuzzy_threshold: float = 85.0) -> JDKeywordsResult:
    """
    Extracts required and preferred skills from a Job Description.

    Skills near words like 'required', 'must have', 'proficient in' (or inside
    requirements/qualifications sections) are classified as required_skills.
    All other skill mentions are classified as preferred_skills.

    Args:
        jd_text: Raw text of the Job Description.
        fuzzy_threshold: Threshold for fuzzy matching (default 85.0).

    Returns:
        JDKeywordsResult containing (required_skills, preferred_skills).
        Supports tuple unpacking and attribute access.
    """
    if not jd_text:
        return JDKeywordsResult(set(), set())

    all_skills = extract_keywords_from_text(jd_text, fuzzy_threshold=fuzzy_threshold)
    required_skills: Set[str] = set()

    # Section-aware line parsing
    current_section = "OTHER"
    lines = jd_text.splitlines()

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        header_probe = cleaned_line.rstrip(":-#* ")
        if _REQ_SECTION_HEADER.match(header_probe):
            current_section = "REQUIRED"
            continue
        elif _PREF_SECTION_HEADER.match(header_probe):
            current_section = "PREFERRED"
            continue
        elif _OTHER_SECTION_HEADER.match(header_probe):
            current_section = "OTHER"
            continue

        # Check line content
        has_req_cue = bool(_REQ_CUES_PATTERN.search(cleaned_line))
        has_pref_cue = bool(_PREF_CUES_PATTERN.search(cleaned_line))

        if (has_req_cue or current_section == "REQUIRED") and not has_pref_cue:
            line_skills = extract_keywords_from_text(cleaned_line, fuzzy_threshold=fuzzy_threshold)
            required_skills.update(line_skills)

    # Keyword proximity window check around required cues
    for match in _REQ_CUES_PATTERN.finditer(jd_text):
        start = max(0, match.start() - 30)
        end = min(len(jd_text), match.end() + 200)
        window = jd_text[start:end]
        window_skills = extract_keywords_from_text(window, fuzzy_threshold=fuzzy_threshold)
        required_skills.update(window_skills)

    # Only include skills actually present in all_skills
    required_skills = required_skills.intersection(all_skills)

    # Preferred skills are other skill mentions
    preferred_skills = all_skills - required_skills

    return JDKeywordsResult(required_skills, preferred_skills)


# ==============================================================================
# 4. Resume Skill Extraction
# ==============================================================================

def extract_keywords_from_resume(resume_text: str, fuzzy_threshold: float = 85.0) -> Set[str]:
    """
    Extracts canonical tech skills from a resume text using alias normalization
    and rapidfuzz fuzzy matching.

    Args:
        resume_text: Raw extracted resume text.
        fuzzy_threshold: Threshold for fuzzy matching (default 85.0).

    Returns:
        Set of canonical skill names found in the resume.
    """
    return extract_keywords_from_text(resume_text, fuzzy_threshold=fuzzy_threshold)


# ==============================================================================
# 5. Keyword Scoring (0.7 Required + 0.3 Preferred)
# ==============================================================================

def keyword_score(
    jd_required: Set[str] | List[str],
    jd_preferred: Set[str] | List[str],
    resume_skills: Set[str] | List[str],
) -> KeywordScoreResult:
    """
    Computes a keyword match score between 0.0 and 1.0.

    Weights:
        - Required skills match: 0.7
        - Preferred skills match: 0.3

    Args:
        jd_required: Required skills from Job Description.
        jd_preferred: Preferred skills from Job Description.
        resume_skills: Skills extracted from the candidate's resume.

    Returns:
        KeywordScoreResult object containing:
            - score (float between 0.0 and 1.0)
            - matched_required (List[str])
            - missing_required (List[str])
            - matched_preferred (List[str])
    """
    req_set = set(jd_required)
    pref_set = set(jd_preferred)
    res_set = set(resume_skills)

    matched_required = sorted(list(req_set & res_set))
    missing_required = sorted(list(req_set - res_set))
    matched_preferred = sorted(list(pref_set & res_set))

    req_total = len(req_set)
    pref_total = len(pref_set)

    if req_total > 0 and pref_total > 0:
        req_score = len(matched_required) / req_total
        pref_score = len(matched_preferred) / pref_total
        score = 0.7 * req_score + 0.3 * pref_score
    elif req_total > 0 and pref_total == 0:
        score = len(matched_required) / req_total
    elif req_total == 0 and pref_total > 0:
        score = len(matched_preferred) / pref_total
    else:
        score = 0.0

    score = max(0.0, min(1.0, float(score)))

    return KeywordScoreResult(
        score=round(score, 4),
        matched_required=matched_required,
        missing_required=missing_required,
        matched_preferred=matched_preferred,
    )


# ==============================================================================
# Verification Test Runner
# ==============================================================================

def main():
    """Runs a verification test comparing sample JD against sample resumes."""
    from parser import extract_jd, extract_resumes

    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "sample_data" / "job_description.txt"
    resumes_dir = base_dir / "sample_data" / "resumes"

    print("=" * 70)
    print("KEYWORD MATCH MODULE - VERIFICATION TEST")
    print("=" * 70)

    # 1. Parse JD
    jd_text = extract_jd(jd_path)
    jd_required, jd_preferred = extract_keywords_from_jd(jd_text)

    print("\n[JOB DESCRIPTION]")
    print(f"File: {jd_path.name}")
    print(f"Total Required Skills ({len(jd_required)}): {sorted(jd_required)}")
    print(f"Total Preferred Skills ({len(jd_preferred)}): {sorted(jd_preferred)}")

    # 2. Parse Resumes
    resumes = extract_resumes(resumes_dir)
    print(f"\nFound {len(resumes)} resume(s) in {resumes_dir.name}/")

    # 3. Score Each Resume
    print("\n" + "=" * 70)
    print("RESUME MATCHING RESULTS")
    print("=" * 70)

    for filename, resume_text in resumes.items():
        resume_skills = extract_keywords_from_resume(resume_text)
        result = keyword_score(jd_required, jd_preferred, resume_skills)

        print(f"\nCandidate Resume: {filename}")
        print("-" * 50)
        print(f"Extracted Resume Skills: {sorted(resume_skills)}")
        print(f"Keyword Match Score:     {result.score:.2%} ({result.score:.4f})")
        print(f"Matched Required ({len(result.matched_required)}/{len(jd_required)}): {result.matched_required}")
        print(f"Missing Required ({len(result.missing_required)}/{len(jd_required)}): {result.missing_required}")
        print(f"Matched Preferred ({len(result.matched_preferred)}/{len(jd_preferred)}): {result.matched_preferred}")

    print("\n" + "=" * 70)
    print("Verification test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
