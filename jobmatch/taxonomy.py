"""Canonical skill taxonomy with synonym collapsing.

The taxonomy is the reason two descriptions of the same thing score as a match.
Without it, "Torch" and "PyTorch" look like two different skills, and a resume
that says one loses to a posting that says the other. Everything here is
deterministic and auditable; no model writes into it.
"""

from __future__ import annotations

import re
import unicodedata

# Canonical name -> surface forms that should collapse onto it.
SYNONYMS: dict[str, list[str]] = {
    "PyTorch": ["torch", "pytorch", "pytorch lightning", "lightning"],
    "TensorFlow": ["tensorflow", "tf", "tf.keras", "keras"],
    "scikit-learn": ["sklearn", "scikit learn", "scikit-learn", "sk learn"],
    "pandas": ["pandas", "pd"],
    "NumPy": ["numpy", "np"],
    "Python": ["python", "python3", "py3", "python 3"],
    "PostgreSQL": ["postgres", "postgresql", "psql", "pgsql"],
    "MySQL": ["mysql", "my sql"],
    "MongoDB": ["mongodb", "mongo"],
    "SQL": ["sql", "structured query language"],
    "Docker": ["docker", "dockerfile", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s", "k8"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Git": ["git", "version control", "github", "gitlab", "bitbucket"],
    "Linux": ["linux", "unix", "ubuntu", "rhel"],
    "React": ["react", "react.js", "reactjs", "react js"],
    "Next.js": ["next.js", "nextjs", "next js"],
    "Node.js": ["node.js", "nodejs", "node"],
    "TypeScript": ["typescript", "ts"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "Java": ["java", "java se"],
    "REST API": ["rest api", "rest", "restful", "restful api", "rest apis", "rest endpoints"],
    "gRPC": ["grpc", "protobuf", "protocol buffers"],
    "GraphQL": ["graphql", "apollo"],
    "MLOps": ["mlops", "model deployment", "model serving", "inference serving"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Prompt Engineering": ["prompt engineering", "prompt design", "prompting", "prompt optimisation", "prompt optimization"],
    "LangGraph": ["langgraph", "lang graph"],
    "FastAPI": ["fastapi", "fast api"],
    "Streamlit": ["streamlit", "stream lit"],
    "Hugging Face": ["hugging face", "huggingface", "hf hub", "transformers library"],
    "OpenCV": ["opencv", "opencv-python", "cv2", "computer vision"],
    "Computer Vision": ["computer vision", "image classification", "object detection", "segmentation", "vision models"],
    "NLP": ["nlp", "natural language processing", "text mining", "text analytics"],
    "Power BI": ["power bi", "powerbi", "power-bi"],
    "Tableau": ["tableau", "tableau server"],
    "Excel": ["excel", "microsoft excel", "ms excel", "pivot tables", "advanced excel", "vlookup", "power query"],
    "ETL": ["etl", "elt", "data pipeline", "data pipelines", "data engineering", "data ingestion"],
    "Data Modeling": ["data modeling", "data modelling", "dimensional modeling", "star schema"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing", "regression analysis", "statistical modelling"],
    "A/B Testing": ["a/b testing", "ab testing", "split testing", "experimentation", "experiment design"],
    "Time Series": ["time series", "time series analysis", "forecasting", "arima", "prophet"],
    "COBOL": ["cobol"],
    "Data Governance": ["data governance", "data lineage", "data catalog", "data quality"],
    "Excel VBA": ["vba", "visual basic for applications", "macro", "macros"],
    "Primavera P6": ["primavera p6", "primavera", "p6", "primavera6"],
    "AutoCAD": ["autocad", "auto cad", "drafting"],
    "Civil 3D": ["civil 3d", "autodesk civil 3d", "civil3d"],
    "Revit": ["revit", "autodesk revit", "bim"],
    "Quantity Take-off": [
        "quantity take-off", "quantity takeoff", "qto", "take off",
        "quantity estimation", "bill of quantities", "boq", "boq preparation",
    ],
    "Cost Estimation": [
        "cost estimation", "cost estimate", "rate analysis", "estimating",
        "cost planning", "budgeting", "estimate",
    ],
    "Tendering": ["tendering", "tender", "tenders", "bid preparation", "bid/no bid", "bidding", "pre bid"],
    "Contract Management": ["contract management", "contracts", "contract administration", "agreements"],
    "Site Supervision": ["site supervision", "site engineer", "site management", "field supervision", "site execution", "construction site operations", "site operations"],
    "Quality Assurance": ["quality assurance", "qa", "quality control", "qc", "quality management"],
    "HSE": ["hse", "health and safety", "safety", "safety management", "ohs", "risk assessment"],
    "Project Management": ["pmp", "prince2", "programme management", "program management"],
    "Scheduling": ["scheduling", "project scheduling", "tendering and estimating", "tms"],
    "Progress Reporting": ["progress reporting", "progress tracking", "reporting", "weekly reports", "mis reporting"],
    "Documentation": ["documentation", "project documentation", "technical documentation", "document control"],
    "Data Analysis": ["data analysis", "data analytics", "analytics", "analysing", "analyzing", "data analysis and reporting"],
    "Excel Pivot": ["pivot table", "pivot tables", "pivot analysis"],
    "Communication": [
        "communication", "communication skills", "verbal communication", "written communication",
        "stakeholder management", "stakeholder engagement", "stakeholder management skills",
        "stakeholders", "stakeholder", "interpersonal skills", "presenting",
        "liaise", "liaison", "public speaking", "presentation skills",
    ],
    "Leadership": ["leadership", "team leadership", "leading", "mentoring", "line management"],
    "Problem Solving": ["problem solving", "problem-solving", "analytical thinking", "critical thinking"],
    "Attention to Detail": ["attention to detail", "detail oriented", "detail-oriented", "meticulous"],
    "Time Management": ["time management", "prioritisation", "prioritization", "multitasking"],
    "Teamwork": ["teamwork", "team work", "collaboration", "collaborative", "cross functional", "cross-functional"],
    "SAP": ["sap", "sap fi", "sap mm", "sap erp", "sap s/4hana"],
    "Oracle": ["oracle", "oracle erp", "oracle netsuite"],
    "PeopleSoft": ["peoplesoft"],
    "C": ["c", "c language", "ansi c"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c sharp", ".net", "dotnet", ".net core"],
    "Go": ["golang", "go language"],
    "Rust": ["rust"],
    "PHP": ["php", "laravel", "symfony"],
    "Ruby": ["ruby", "rails", "ruby on rails"],
    "Scala": ["scala"],
    "Spark": ["spark", "apache spark", "pyspark", "databricks"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce"],
    "Kafka": ["kafka", "apache kafka", "event streaming"],
    "Airflow": ["airflow", "apache airflow", "dag orchestration"],
    "dbt": ["dbt", "data build tool"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],
    "Redshift": ["redshift"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    "Ansible": ["ansible"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions", "actions runner"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous delivery", "continuous integration and delivery", "build automation", "deployment pipeline", "release pipeline", "ci cd pipelines", "devops pipeline"],
    "OpenTelemetry": ["opentelemetry", "otel", "prometheus", "grafana", "observability"],
    "Selenium": ["selenium", "cypress", "playwright", "test automation", "end to end testing", "e2e testing"],
    "Unit Testing": ["unit testing", "unit tests", "pytest", "junit", "jest"],
    "Data Structures": ["data structures", "algorithms", "algorithm design", "dsa", "complexity analysis"],
    "System Design": ["system design", "distributed systems", "scalability", "software architecture", "architecture design"],
    "OOPS": ["oops", "oop", "object oriented programming", "object-oriented programming"],
    "Web Scraping": ["web scraping", "scraping", "crawling", "beautifulsoup", "scrapy"],
    "Microservices": ["microservices", "microservice", "service oriented architecture", "soa"],
    "ETL Tools": ["fivetran", "stitch", "informatica", "talend", "ssis", "kettle"],
    "Financial Modelling": ["financial modelling", "financial modeling", "valuation model", "three statement model"],
    "Valuation": ["valuation", "dcf", "discounted cash flow", "comps analysis"],
    "Budgeting": ["budgeting", "budget control", "budgeting and forecasting", "cost control"],
    "Procurement": ["procurement", "vendor management", "supplier management", "sourcing"],
    "Subcontractor Management": ["subcontractor management", "subcontract management", "contractor management"],
    "Change Order": ["change order", "variation order", "change management", "variations"],
    "Microsoft Office": ["microsoft office", "ms office", "office suite"],
    "SAP FICO": ["sap fico", "fico"],
}

_STOPWORDS = {
    "and", "or", "with", "the", "a", "an", "of", "in", "for", "to", "on", "at",
    "experience", "knowledge", "skills", "ability", "strong", "good", "solid",
    "familiarity", "familiar", "exposure", "working", "excellent", "proven",
    "using", "use", "used", "hands", "on", "must", "have", "has", "plus", "etc",
    "you", "your", "we", "our", "role", "job", "work", "team", "teams", "people",
}

_PUNCT = re.compile(r"[^\w+#.]+")
# Symbols that are part of a name must stay attached to the word before them,
# so "C #" and "C#" collapse together while "Node . js" does not become "node".
_SYMBOL_GLUING = re.compile(r"\s+([#+])")


def _norm_key(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_form = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = ascii_form.lower().strip()
    glued = _SYMBOL_GLUING.sub(r"\1", lowered)
    stripped = _PUNCT.sub(" ", glued)
    collapsed = " ".join(stripped.split())
    return collapsed.replace(" #", "#").replace(" +", "+")


def _singular_variants(key: str) -> list[str]:
    """Plausible singular spellings of a key, for plural surface forms.

    Kept deliberately naive: only a trailing "s" on a single word is removed,
    and never from words ending in "ss", so "SAP" and "process" survive.
    """
    out: list[str] = []
    if key.endswith("ies") and len(key) > 5:
        out.append(key[:-3] + "y")
    if key.endswith("es") and len(key) > 4:
        out.append(key[:-2])
    if key.endswith("s") and len(key) > 3 and key[-2] != "s":
        out.append(key[:-1])
    return out


# Longest-first so multiword canonical forms win over their substrings.
_CANONICAL_KEYS = sorted(SYNONYMS, key=len, reverse=True)

# Alias -> canonical. Canonical names are registered first so a skill that has
# its own entry is never swallowed by a broader entry that lists it as an
# alias, which is what happened to FastAPI under the "Python web" grouping.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _name in _CANONICAL_KEYS:
    _ALIAS_TO_CANONICAL[_norm_key(_name)] = _name
for _canonical, _aliases in SYNONYMS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL.setdefault(_norm_key(_alias), _canonical)

# Plurals are folded at lookup time rather than expanded into the table, so
# "change orders" reaches the "change order" alias without the map needing a
# generated entry per plural.

# Inverted canonical -> its own known surface forms, for evidence searching.
_CANONICAL_FORMS: dict[str, list[str]] = {
    _name: sorted({_name, *_aliases}, key=len, reverse=True)
    for _name, _aliases in SYNONYMS.items()
}


def normalize_skill(name: str) -> str | None:
    """Collapse a surface form onto its canonical name, or return None.

    Returns None when the skill is outside the taxonomy, which is the signal
    the matcher uses to fall back to embedding similarity.
    """
    key = _norm_key(name)
    if not key:
        return None
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    for variant in _singular_variants(key):
        if variant in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[variant]
    return None


def normalize_or_self(name: str) -> str:
    """Canonicalize when possible, else return a title-cased trimmed form."""
    canonical = normalize_skill(name)
    if canonical:
        return canonical
    cleaned = " ".join(name.split())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def surface_forms(canonical: str) -> list[str]:
    """Every spelling that should count as evidence for a canonical skill."""
    forms = _CANONICAL_FORMS.get(canonical)
    if forms:
        return forms
    key = _norm_key(canonical)
    for name, aliases in _CANONICAL_FORMS.items():
        if _norm_key(name) == key:
            return [name, *aliases]
    return [canonical]


def content_tokens(text: str) -> set[str]:
    """Meaningful lowercase tokens, used for keyword-level overlap signals."""
    key = _norm_key(text)
    return {tok for tok in key.split() if tok and tok not in _STOPWORDS and len(tok) > 1}


def is_in_taxonomy(name: str) -> bool:
    return normalize_skill(name) is not None


def taxonomy_size() -> int:
    return len(SYNONYMS)
