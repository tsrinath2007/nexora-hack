"""
Keyword-based matching module for Resume Ranker.

Features:
- Predefined tech-skill taxonomy (~175+ canonical skills) and comprehensive alias mapping.
- Exact alias-normalized regex matching with word-boundary safety (e.g. 'c' never matches inside 'c++', 'c#', 'c--').
- Structured "Skill    Level" table parsing (Advanced=1.0, Intermediate=0.7, Beginner=0.4, None=-1.0).
- Fallback freeform matching with negation window detection ('no experience', 'not familiar', 'none', 'n/a').
- Support for multi-word skills like 'unreal engine', 'data structures & algorithms', 'object-oriented programming'.
- JD extraction separating required vs preferred skills based on section/cue proximity.
- Keyword scoring summing proficiency weights, normalizing to 0-1.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple, Any, Optional, Union
from rapidfuzz import fuzz, process


# ==============================================================================
# 1. Tech-Skill Taxonomy (~175+ Canonical Skills) & Aliases
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

    # Game Development & Real-time Systems
    "unity", "unreal engine", "3d game development", "game physics", "ai programming",
    "multiplayer/network programming", "shader programming", "game development",

    # Core Computer Science & Architecture
    "data structures & algorithms", "object-oriented programming",
    "microservices", "unit testing", "pytest", "selenium", "cypress", "junit",
    "agile", "scrum", "jira", "api design", "distributed systems", "websocket",
    "grpc", "message queues", "system design",

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
    "generative ai", "vector search", "reinforcement learning"
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

    # Game Development & Multi-word Skills
    "unity": "unity",
    "unity3d": "unity",
    "unity 3d": "unity",
    "unity engine": "unity",
    "unreal": "unreal engine",
    "unreal engine": "unreal engine",
    "unreal engine 4": "unreal engine",
    "unreal engine 5": "unreal engine",
    "ue4": "unreal engine",
    "ue5": "unreal engine",
    "data structures & algorithms": "data structures & algorithms",
    "data structures and algorithms": "data structures & algorithms",
    "data structures & algorithm": "data structures & algorithms",
    "data structures and algorithm": "data structures & algorithms",
    "data structures": "data structures & algorithms",
    "algorithm design": "data structures & algorithms",
    "dsa": "data structures & algorithms",
    "object-oriented programming": "object-oriented programming",
    "object oriented programming": "object-oriented programming",
    "object-oriented software design": "object-oriented programming",
    "object oriented software design": "object-oriented programming",
    "object-oriented design": "object-oriented programming",
    "oop": "object-oriented programming",
    "oops": "object-oriented programming",
    "3d game development": "3d game development",
    "interactive 3d development": "3d game development",
    "game development": "game development",
    "game-development": "game development",
    "game dev": "game development",
    "game-dev": "game development",
    "game engine development": "game development",
    "game engine": "game development",
    "game physics": "game physics",
    "interactive physics": "game physics",
    "interactive physics & collision systems": "game physics",
    "interactive physics and collision systems": "game physics",
    "ai programming": "ai programming",
    "game ai": "ai programming",
    "virtual character behaviour": "ai programming",
    "virtual character behavior": "ai programming",
    "character behaviour": "ai programming",
    "character behavior": "ai programming",
    "npc ai": "ai programming",
    "multiplayer/network programming": "multiplayer/network programming",
    "multiplayer networking": "multiplayer/network programming",
    "networked multiplayer systems": "multiplayer/network programming",
    "network programming": "multiplayer/network programming",
    "multiplayer programming": "multiplayer/network programming",
    "shader programming": "shader programming",
    "shaders": "shader programming",
    "shader": "shader programming",

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
    "version control / collaborative development": "git",
    "version control": "git",
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

def _make_boundary_pattern(term: str) -> re.Pattern:
    """
    Creates a strict boundary-safe regex pattern.
    Disallows preceding or following word characters or special tech characters +, #, -.
    Ensures 'c' never matches inside 'c++', 'c#', 'c--', etc.
    """
    escaped = re.escape(term)
    prefix = r"(?<![a-zA-Z0-9_+#\-])"
    suffix = r"(?![a-zA-Z0-9_+#\-])"
    return re.compile(rf"{prefix}{escaped}{suffix}", re.IGNORECASE)

# Precompile regex boundary patterns for all aliases
_ALIAS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_make_boundary_pattern(alias_key), canonical_skill)
    for alias_key, canonical_skill in SORTED_ALIASES
]

# Targets for fuzzy matching: only multi-character canonical skills (>= 4 chars)
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
    "scale", "scalable", "applications", "frameworks", "pipelines", "models", "automated",
    "programming", "software", "development", "developer", "computer", "systems", "system"
}

# Negation detection patterns for window check (~5 words around match)
_PRE_NEGATION_RE = re.compile(
    r"\b(no\s+(?:prior\s+)?experience(?:\s+(?:in|with|of))?|not\s+familiar(?:\s+with)?|no\s+knowledge(?:\s+of)?|not\s+proficient(?:\s+in)?|zero\s+experience(?:\s+in)?)\b",
    re.IGNORECASE,
)

_POST_NEGATION_RE = re.compile(
    r"^[\s:,\-–—\(\)]*\b(none|n/a|na|no\s+experience)\b",
    re.IGNORECASE,
)

# Table row detection pattern: ^(.+?)\s+(Advanced|Intermediate|Beginner|None)$
_TABLE_ROW_RE = re.compile(
    r"^\s*(.+?)\s+(Advanced|Intermediate|Beginner|None)\s*$",
    re.IGNORECASE
)

LEVEL_WEIGHTS: Dict[str, float] = {
    "advanced": 1.0,
    "intermediate": 0.7,
    "beginner": 0.4,
    "none": -1.0,
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
# 2. Skill Normalization & Extraction Helpers
# ==============================================================================

def normalize_skill_name(raw_skill: str) -> Optional[str]:
    """
    Normalizes a skill string against the taxonomy using exact alias lookup,
    word-boundary-safe regex matching, or fuzzy matching.
    Guarantees 'C' will NOT match inside 'C++', 'C#', or 'C--'.
    """
    cleaned = raw_skill.strip().lower()
    if not cleaned:
        return None

    # 1. Exact alias / taxonomy lookup
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]

    # 2. Boundary-safe regex matching (longer aliases evaluated first)
    for pattern, canonical_skill in _ALIAS_PATTERNS:
        if pattern.search(raw_skill):
            return canonical_skill

    # 3. Fuzzy matching for multi-character skills (len >= 4)
    if len(cleaned) >= 4:
        match = process.extractOne(cleaned, _FUZZY_TARGETS, scorer=fuzz.ratio, score_cutoff=85)
        if match:
            return match[0]

    return None


def _is_negated_in_window(text: str, start: int, end: int, window_words: int = 5) -> bool:
    """
    Checks whether negation words appear within ~5 words before or after the match,
    respecting sentence/clause boundaries (. ; \n) and directional context:
    - Preceding negation: phrases like 'no experience in', 'not familiar with' before the skill.
    - Following negation: terms like 'none', 'n/a', 'no experience' immediately following the skill.
    """
    # Pre-window: up to window_words before match within the same sentence/clause
    pre_text = text[:start]
    last_boundary = max(pre_text.rfind("."), pre_text.rfind("\n"), pre_text.rfind(";"))
    pre_clause = pre_text[last_boundary + 1:] if last_boundary != -1 else pre_text
    pre_tokens = pre_clause.split()[-window_words:]
    if pre_tokens and _PRE_NEGATION_RE.search(" ".join(pre_tokens)):
        return True

    # Post-window: up to window_words after match within the same sentence/clause
    post_text = text[end:]
    first_boundary = len(post_text)
    for sep in [".", "\n", ";"]:
        pos = post_text.find(sep)
        if pos != -1 and pos < first_boundary:
            first_boundary = pos
    post_clause = post_text[:first_boundary]
    post_tokens = post_clause.split()[:window_words]
    if post_tokens and _POST_NEGATION_RE.search(" ".join(post_tokens)):
        return True

    return False


def extract_keywords_from_text(text: str, fuzzy_threshold: float = 85.0) -> Set[str]:
    """
    Extracts canonical skills from text using boundary-safe regex matching
    and fuzzy matching with negation filtering.
    """
    if not text:
        return set()

    found_skills: Set[str] = set()

    # Step 1: Word-boundary-safe exact regex matching
    for pattern, canonical_skill in _ALIAS_PATTERNS:
        for m in pattern.finditer(text):
            if not _is_negated_in_window(text, m.start(), m.end()):
                found_skills.add(canonical_skill)

    # Step 2: RapidFuzz fuzzy matching for typos/variants within clauses
    # Split text by punctuation delimiters so n-grams do not cross phrase boundaries
    candidate_phrases: Set[Tuple[str, int, int]] = set()

    for clause_match in re.finditer(r"[^,;\n\.\:\(\)\[\]]+", text):
        clause = clause_match.group().strip()
        clause_start = clause_match.start()
        clause_end = clause_match.end()

        if not clause:
            continue

        tokens = clause.split()
        for n in (1, 2, 3):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n]).strip(".-_")
                cleaned_ngram = ngram.lower()
                if len(cleaned_ngram) < 4 or cleaned_ngram in _STOP_WORDS:
                    continue

                # Find exact position within clause
                ngram_match = re.search(r"\b" + re.escape(ngram) + r"\b", text[clause_start:clause_end], re.IGNORECASE)
                if ngram_match:
                    start_pos = clause_start + ngram_match.start()
                    end_pos = clause_start + ngram_match.end()
                    candidate_phrases.add((cleaned_ngram, start_pos, end_pos))

    for candidate, start, end in candidate_phrases:
        if _is_negated_in_window(text, start, end):
            continue
        match = process.extractOne(
            candidate,
            _FUZZY_TARGETS,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        )
        if match:
            found_skills.add(match[0])

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
    r"minimum\s+requirements?|essential\s+skills?|what\s+we(?:\'re)?\s+looking\s+for|required)$",
    re.IGNORECASE,
)

_PREF_SECTION_HEADER = re.compile(
    r"^(preferred|nice\s+to\s+have|bonus|desired|desirable|optional|good\s+to\s+have|preferred\s+qualifications?)$",
    re.IGNORECASE,
)

_OTHER_SECTION_HEADER = re.compile(
    r"^(responsibilities|overview|about|duties|what\s+you(?:\'ll)?\s+do|summary|education|experience|suggested\s+test\s+order)",
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
    preferred_skills: Set[str] = set()

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
        elif (has_pref_cue or current_section == "PREFERRED") and not has_req_cue:
            line_skills = extract_keywords_from_text(cleaned_line, fuzzy_threshold=fuzzy_threshold)
            preferred_skills.update(line_skills)

    # Fallback to proximity window only if no section headers or cues identified required skills
    if not required_skills:
        for match in _REQ_CUES_PATTERN.finditer(jd_text):
            start = max(0, match.start() - 30)
            end = min(len(jd_text), match.end() + 200)
            window = jd_text[start:end]
            pref_match = _PREF_CUES_PATTERN.search(window)
            if pref_match:
                window = window[:pref_match.start()]
            window_skills = extract_keywords_from_text(window, fuzzy_threshold=fuzzy_threshold)
            required_skills.update(window_skills)

    # Only include skills actually present in all_skills
    required_skills = required_skills.intersection(all_skills)

    # Determine preferred skills
    if not preferred_skills:
        preferred_skills = all_skills - required_skills
    else:
        preferred_skills = preferred_skills.intersection(all_skills) - required_skills

    return JDKeywordsResult(required_skills, preferred_skills)


# ==============================================================================
# 4. Resume Skill Extraction with Proficiency & Table Parsing
# ==============================================================================

def extract_keywords_from_resume(
    resume_text: str,
    fuzzy_threshold: float = 85.0,
) -> Dict[str, float]:
    r"""
    Extracts canonical tech skills and proficiency weights from candidate resume text:
    1. First pass: regex-matches lines matching ^(.+?)\s+(Advanced|Intermediate|Beginner|None)$
       Maps level to weights: Advanced=1.0, Intermediate=0.7, Beginner=0.4, None=-1.0.
       Normalizes skills with boundary safety ('C' never matches inside 'C++' or 'C#').
    2. Fallback: for freeform resumes or skills not covered in the table, extracts
       mentions while checking a ~5-word window around each match for negation cues
       ('no experience', 'not familiar', 'none', 'n/a') and discards matches if found.

    Args:
        resume_text: Raw extracted resume text.
        fuzzy_threshold: Threshold for fuzzy matching (default 85.0).

    Returns:
        Dict[str, float] mapping canonical skill -> level weight.
    """
    if not resume_text or not resume_text.strip():
        return {}

    skill_weights: Dict[str, float] = {}
    table_matched = False

    # Pass 1: Structured "Skill    Level" table matching
    for line in resume_text.splitlines():
        line_clean = line.strip()
        match = _TABLE_ROW_RE.match(line_clean)
        if match:
            raw_skill_str = match.group(1).strip()
            level_str = match.group(2).strip().lower()
            weight = LEVEL_WEIGHTS.get(level_str, 0.0)

            canonical = normalize_skill_name(raw_skill_str)
            if canonical:
                skill_weights[canonical] = weight
                table_matched = True

    # Pass 2: Fallback for freeform resumes (or skills mentioned in freeform text)
    # If a skill was not in the table, extract with negation window check
    for pattern, canonical in _ALIAS_PATTERNS:
        # If skill already recorded from table, respect the table's explicit proficiency
        if canonical in skill_weights:
            continue

        for m in pattern.finditer(resume_text):
            if not _is_negated_in_window(resume_text, m.start(), m.end()):
                # Un-negated freeform mention gets standard weight 1.0
                skill_weights[canonical] = 1.0

    return skill_weights


# ==============================================================================
# 5. Keyword Scoring with Proficiency Summing
# ==============================================================================

def keyword_score(
    jd_required: Set[str] | List[str],
    jd_preferred: Set[str] | List[str],
    resume_skill_weights: Union[Dict[str, float], Set[str], List[str]],
) -> KeywordScoreResult:
    """
    Computes a keyword match score between 0.0 and 1.0 taking skill proficiency into account:
    - Sums level weights for required skills present in jd_required:
      Advanced (+1.0), Intermediate (+0.7), Beginner (+0.4), None (-1.0), Unmentioned (0.0).
    - Normalizes sum to 0-1.
    - Classifies:
      - matched_required: required skills with weight > 0
      - missing_required: required skills with weight <= 0 (unmentioned or explicitly None)
      - matched_preferred: preferred skills with weight > 0

    Args:
        jd_required: Required skills from Job Description.
        jd_preferred: Preferred skills from Job Description.
        resume_skill_weights: Dict mapping skill -> weight, or set/list of skills.

    Returns:
        KeywordScoreResult with score, matched_required, missing_required, matched_preferred.
    """
    if not isinstance(resume_skill_weights, dict):
        weights_dict: Dict[str, float] = {s: 1.0 for s in resume_skill_weights}
    else:
        weights_dict = resume_skill_weights

    req_set = set(jd_required)
    pref_set = set(jd_preferred)

    matched_required: List[str] = []
    missing_required: List[str] = []
    req_weight_sum = 0.0

    for skill in sorted(req_set):
        weight = weights_dict.get(skill, 0.0)
        req_weight_sum += weight
        if weight > 0:
            matched_required.append(skill)
        else:
            missing_required.append(skill)

    matched_preferred: List[str] = []
    pref_weight_sum = 0.0

    for skill in sorted(pref_set):
        weight = weights_dict.get(skill, 0.0)
        if weight > 0:
            pref_weight_sum += weight
            matched_preferred.append(skill)

    # Normalize required score: max possible sum is len(req_set) * 1.0
    num_req = len(req_set)
    num_pref = len(pref_set)

    if num_req > 0:
        req_score = max(0.0, min(1.0, req_weight_sum / num_req))
    else:
        req_score = 0.0

    if num_pref > 0:
        pref_score = max(0.0, min(1.0, pref_weight_sum / num_pref))
    else:
        pref_score = 0.0

    # Weighted combination: 0.7 required + 0.3 preferred
    if num_req > 0 and num_pref > 0:
        final_score = 0.7 * req_score + 0.3 * pref_score
    elif num_req > 0:
        final_score = req_score
    elif num_pref > 0:
        final_score = pref_score
    else:
        final_score = 0.0

    final_score = round(max(0.0, min(1.0, final_score)), 4)

    return KeywordScoreResult(
        score=final_score,
        matched_required=matched_required,
        missing_required=missing_required,
        matched_preferred=matched_preferred,
    )


# ==============================================================================
# Verification Test Runner
# ==============================================================================

def main():
    """Runs verification test evaluating the 6 Game Developer test resumes."""
    from parser import extract_text_from_pdf

    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "test_data" / "Software_Game_Developer_Job_Description.pdf"
    resumes_dir = base_dir / "test_data" / "resumes"

    # Fallback to Downloads if not in test_data
    if not jd_path.exists():
        jd_path = Path(r"C:\Users\SES\Downloads\Software_Game_Developer_Job_Description.pdf")
    if not resumes_dir.exists():
        resumes_dir = Path(r"C:\Users\SES\Downloads")

    print("=" * 80)
    print("KEYWORD MATCH MODULE - PROFICIENCY & NEGATION VERIFICATION TEST")
    print("=" * 80)

    # 1. Parse JD
    jd_text = extract_text_from_pdf(jd_path)
    jd_required, jd_preferred = extract_keywords_from_jd(jd_text)

    print("\n[JOB DESCRIPTION]")
    print(f"File: {jd_path.name}")
    print(f"Total Required Skills ({len(jd_required)}): {sorted(jd_required)}")
    print(f"Total Preferred Skills ({len(jd_preferred)}): {sorted(jd_preferred)}")

    # 2. Test specific multi-word skills detection
    print("\n[MULTI-WORD SKILLS DETECTION TEST]")
    test_text = (
        "Experienced in Unreal Engine, Data Structures and Algorithms, "
        "Data Structures & Algorithms, Object-Oriented Programming, and OOP."
    )
    detected = extract_keywords_from_text(test_text)
    print("Test text:", test_text)
    print("Detected skills:", sorted(detected))
    assert "unreal engine" in detected, "Failed to detect 'unreal engine'"
    assert "data structures & algorithms" in detected, "Failed to detect 'data structures & algorithms'"
    assert "object-oriented programming" in detected, "Failed to detect 'object-oriented programming'"
    print("[PASS] Multi-word skills and aliases correctly detected!")

    # 3. Test C vs C++ boundary safety
    c_boundary_text = "I know C++ and C#."
    detected_c = extract_keywords_from_text(c_boundary_text)
    print("\n[C vs C++ BOUNDARY TEST]")
    print("Test text:", c_boundary_text)
    print("Detected skills:", sorted(detected_c))
    assert "c++" in detected_c and "c#" in detected_c, "C++ or C# missing"
    assert "c" not in detected_c, "'C' incorrectly matched inside C++ or C#!"
    print("[PASS] 'C' does not match inside 'C++' or 'C#'!")

    # 3b. Test freeform negation detection
    neg_test_text = "Skills: Python, Git. No experience in C++, not familiar with Unreal Engine, and Unity: none."
    detected_neg = extract_keywords_from_text(neg_test_text)
    print("\n[FREEFORM NEGATION TEST]")
    print("Test text:", neg_test_text)
    print("Detected skills:", sorted(detected_neg))
    assert "python" in detected_neg and "git" in detected_neg, "Positive skills missing"
    assert "c++" not in detected_neg, "'C++' should be discarded due to 'No experience in'"
    assert "unreal engine" not in detected_neg, "'Unreal Engine' should be discarded due to 'not familiar with'"
    assert "unity" not in detected_neg, "'Unity' should be discarded due to 'none'"
    print("[PASS] Freeform negation words ('no experience', 'not familiar', 'none') correctly filter out skills!")

    # 4. Evaluate all 6 test resumes
    print("\n" + "=" * 80)
    print("6 TEST RESUMES KEYWORD SCORING RESULTS")
    print("=" * 80)

    resume_files = sorted(resumes_dir.glob("CAND_*.pdf"))
    if not resume_files:
        print("No CAND_*.pdf resumes found in", resumes_dir)
        return

    for rf in resume_files:
        resume_text = extract_text_from_pdf(rf)
        weights = extract_keywords_from_resume(resume_text)
        res = keyword_score(jd_required, jd_preferred, weights)

        cand_id = rf.name.split("_")[0] + "_" + rf.name.split("_")[1]
        print(f"\nCandidate: {cand_id} ({rf.name})")
        print("-" * 60)
        print(f"Keyword Score:     {res.score:.2%} ({res.score:.4f})")
        print(f"Matched Required ({len(res.matched_required)}/{len(jd_required)}): {res.matched_required}")
        print(f"Missing Required ({len(res.missing_required)}/{len(jd_required)}): {res.missing_required}")
        print(f"Matched Preferred ({len(res.matched_preferred)}/{len(jd_preferred)}): {res.matched_preferred}")

        if "CAND_005" in rf.name:
            print("\n>>> CONFIRMING CAND_005 BEHAVIOR <<<")
            print(f"    CAND_005 matched_required: {res.matched_required}")
            print(f"    CAND_005 missing_required count: {len(res.missing_required)} / {len(jd_required)}")
            assert res.matched_required == [], f"Expected [] but got {res.matched_required}"
            assert len(res.missing_required) == 8, f"Expected 8 missing, got {len(res.missing_required)}"
            assert len(jd_required) == 8, f"Expected 8 required skills in JD, got {len(jd_required)}"
            print("    [PASS] CAND_005 shows matched_required: [] and missing_required: all 8 required skills!")

    print("\n" + "=" * 80)
    print("All 6 resumes evaluated and verified successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
