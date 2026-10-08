"""
resume_analyzer.py
==================
Local, lightweight NLP/ML resume analysis module for Resumate.

Place this file in the project root (E:\\Resumate\\resume_analyzer.py),
next to app.py.

No external paid API is used. Techniques:
  - Regular expressions for entity extraction (email, phone, dates, degrees)
  - Keyword/dictionary matching with safe word boundaries for skills
  - Section segmentation of the resume text
  - TF-IDF + cosine similarity (scikit-learn) for career category matching
    (falls back to pure keyword-overlap scoring if scikit-learn is absent)
  - Rule-based scoring for overall score and ATS score

Main entry point:
    analyze_resume(resume_text: str) -> dict
"""

import re

# scikit-learn is optional: career matching falls back to keyword overlap.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. SKILL DATABASE (dictionary matching with safe boundaries)
# ---------------------------------------------------------------------------
# skill display name -> list of lowercase regex-safe aliases
SKILL_DB = {
    "Programming Languages": {
        "Python": ["python"],
        "Java": ["java"],
        "C": ["c"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "c sharp"],
        "JavaScript": ["javascript", "js"],
        "TypeScript": ["typescript"],
        "PHP": ["php"],
        "Go": ["golang", "go lang"],
        "Ruby": ["ruby"],
        "Kotlin": ["kotlin"],
        "Swift": ["swift"],
        "R": ["r programming", "r language"],
        "MATLAB": ["matlab"],
        "Scala": ["scala"],
        "Rust": ["rust"],
    },
    "Web Development": {
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3"],
        "React": ["react", "reactjs", "react.js"],
        "Angular": ["angular", "angularjs"],
        "Vue.js": ["vue", "vuejs", "vue.js"],
        "Node.js": ["node", "nodejs", "node.js"],
        "Express.js": ["express", "expressjs", "express.js"],
        "Flask": ["flask"],
        "Django": ["django"],
        "Bootstrap": ["bootstrap"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "jQuery": ["jquery"],
        "REST API": ["rest api", "restful", "rest apis"],
        "Spring Boot": ["spring boot", "springboot", "spring"],
        "Next.js": ["next.js", "nextjs"],
    },
    "Databases": {
        "MySQL": ["mysql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MongoDB": ["mongodb", "mongo"],
        "SQLite": ["sqlite"],
        "Oracle": ["oracle"],
        "SQL": ["sql"],
        "Redis": ["redis"],
        "Firebase": ["firebase"],
    },
    "Data Science & ML": {
        "Machine Learning": ["machine learning", "ml"],
        "Deep Learning": ["deep learning", "dl"],
        "Data Science": ["data science"],
        "Data Analysis": ["data analysis", "data analytics"],
        "NLP": ["nlp", "natural language processing"],
        "Computer Vision": ["computer vision", "opencv"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "Keras": ["keras"],
        "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "Matplotlib": ["matplotlib"],
        "Seaborn": ["seaborn"],
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau"],
        "Excel": ["excel", "ms excel", "microsoft excel"],
        "Statistics": ["statistics", "statistical analysis"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure", "microsoft azure"],
        "Google Cloud": ["gcp", "google cloud"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Jenkins": ["jenkins"],
        "CI/CD": ["ci/cd", "ci cd", "continuous integration"],
        "Linux": ["linux", "ubuntu"],
        "Terraform": ["terraform"],
    },
    "Tools & Platforms": {
        "Git": ["git"],
        "GitHub": ["github"],
        "GitLab": ["gitlab"],
        "VS Code": ["vs code", "vscode", "visual studio code"],
        "Jupyter": ["jupyter", "jupyter notebook"],
        "Postman": ["postman"],
        "Figma": ["figma"],
        "Android Studio": ["android studio"],
        "Jira": ["jira"],
    },
    "Cybersecurity & Networking": {
        "Cybersecurity": ["cybersecurity", "cyber security"],
        "Ethical Hacking": ["ethical hacking", "penetration testing", "pentesting"],
        "Network Security": ["network security"],
        "Networking": ["networking", "computer networks", "tcp/ip"],
        "Kali Linux": ["kali linux", "kali"],
        "Wireshark": ["wireshark"],
        "Nmap": ["nmap"],
        "Burp Suite": ["burp suite", "burpsuite"],
        "Cryptography": ["cryptography"],
    },
    "Soft Skills": {
        "Communication": ["communication", "communication skills"],
        "Teamwork": ["teamwork", "team work", "team player", "collaboration"],
        "Leadership": ["leadership"],
        "Problem Solving": ["problem solving", "problem-solving"],
        "Time Management": ["time management"],
        "Adaptability": ["adaptability"],
        "Critical Thinking": ["critical thinking"],
    },
}

# ---------------------------------------------------------------------------
# 2. CAREER CATEGORY PROFILES (used for career suggestion + ATS keywords)
# ---------------------------------------------------------------------------
CAREER_PROFILES = {
    "Software Development": [
        "python", "java", "c++", "c#", "data structures", "algorithms", "oop",
        "git", "github", "sql", "rest api", "software", "debugging", "testing",
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "angular", "node.js", "flask",
        "django", "bootstrap", "frontend", "backend", "full stack", "rest api",
        "php", "express",
    ],
    "Data Science": [
        "python", "machine learning", "statistics", "pandas", "numpy",
        "data science", "deep learning", "nlp", "tensorflow", "scikit-learn",
        "data visualization", "matplotlib", "jupyter",
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "tableau", "data analysis", "statistics",
        "python", "pandas", "reporting", "dashboards", "visualization",
    ],
    "Machine Learning Engineer": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
        "python", "nlp", "computer vision", "scikit-learn", "model deployment",
    ],
    "Cybersecurity": [
        "cybersecurity", "ethical hacking", "network security", "kali linux",
        "wireshark", "nmap", "penetration testing", "cryptography", "firewall",
        "burp suite", "vulnerability",
    ],
    "Cloud Computing / DevOps": [
        "aws", "azure", "google cloud", "docker", "kubernetes", "jenkins",
        "ci/cd", "linux", "terraform", "devops", "cloud",
    ],
    "Mobile App Development": [
        "android", "kotlin", "swift", "flutter", "react native", "ios",
        "android studio", "mobile", "dart",
    ],
    "Database Administration": [
        "sql", "mysql", "postgresql", "oracle", "mongodb", "database design",
        "normalization", "stored procedures", "backup",
    ],
    "UI/UX Design": [
        "figma", "ui", "ux", "wireframe", "prototyping", "adobe xd",
        "user research", "design",
    ],
}

# ---------------------------------------------------------------------------
# 3. SECTION HEADER PATTERNS
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "contact":        r"(contact|personal\s+details|personal\s+information)",
    "summary":        r"(summary|objective|profile|about\s+me|career\s+objective)",
    "skills":         r"(skills|technical\s+skills|core\s+competencies|technologies|technical\s+proficienc)",
    "education":      r"(education|academic|qualification|educational\s+background)",
    "experience":     r"(experience|employment|work\s+history|internship|professional\s+background)",
    "projects":       r"(projects|academic\s+projects|personal\s+projects|mini\s+projects|major\s+project)",
    "certifications": r"(certification|certificate|courses|training|licenses)",
    "achievements":   r"(achievement|awards|honors|accomplishment|extra[- ]?curricular|activities)",
}

ACTION_VERBS = [
    "developed", "designed", "implemented", "built", "created", "led",
    "managed", "improved", "optimized", "analyzed", "automated", "deployed",
    "collaborated", "achieved", "reduced", "increased", "launched", "tested",
    "integrated", "maintained", "trained", "organized", "researched",
]

DEGREE_PATTERNS = [
    (r"\b(b\.?\s?tech|bachelor\s+of\s+technology)\b", "B.Tech"),
    (r"\b(m\.?\s?tech|master\s+of\s+technology)\b", "M.Tech"),
    (r"\b(b\.?\s?e\.?|bachelor\s+of\s+engineering)\b", "B.E."),
    (r"\b(m\.?\s?e\.?|master\s+of\s+engineering)\b", "M.E."),
    (r"\b(b\.?\s?sc|bachelor\s+of\s+science)\b", "B.Sc"),
    (r"\b(m\.?\s?sc|master\s+of\s+science)\b", "M.Sc"),
    (r"\b(bca|bachelor\s+of\s+computer\s+applications?)\b", "BCA"),
    (r"\b(mca|master\s+of\s+computer\s+applications?)\b", "MCA"),
    (r"\b(bba|bachelor\s+of\s+business\s+administration)\b", "BBA"),
    (r"\b(mba|master\s+of\s+business\s+administration)\b", "MBA"),
    (r"\b(b\.?\s?com|bachelor\s+of\s+commerce)\b", "B.Com"),
    (r"\b(m\.?\s?com|master\s+of\s+commerce)\b", "M.Com"),
    (r"\b(ph\.?\s?d|doctorate)\b", "Ph.D"),
    (r"\b(diploma|polytechnic)\b", "Diploma"),
    (r"\b(12th|xii|intermediate|senior\s+secondary|higher\s+secondary|hsc)\b", "Class XII"),
    (r"\b(10th|x\b|matriculation|secondary\s+school|ssc)\b", "Class X"),
]

FIELD_PATTERNS = [
    "computer science", "information technology", "electronics", "electrical",
    "mechanical", "civil", "artificial intelligence", "data science",
    "machine learning", "cyber security", "cybersecurity", "software engineering",
    "it", "cse", "ece", "eee", "aiml", "ai & ml", "commerce", "science", "arts",
    "biotechnology", "chemical",
]

JOB_TITLE_PATTERN = (
    r"(intern(ship)?|developer|engineer|analyst|designer|consultant|trainee|"
    r"manager|administrator|programmer|freelancer?|associate|assistant|"
    r"researcher|tutor|teacher|volunteer)"
)

MONTH = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)"
DURATION_PATTERN = re.compile(
    rf"({MONTH}\.?\s*'?\d{{2,4}}|\b(19|20)\d{{2}}\b)\s*[-–—to]+\s*"
    rf"({MONTH}\.?\s*'?\d{{2,4}}|\b(19|20)\d{{2}}\b|present|current|now|ongoing)",
    re.IGNORECASE,
)

CERT_PROVIDERS = [
    "coursera", "udemy", "nptel", "edx", "udacity", "google", "microsoft",
    "aws", "ibm", "cisco", "oracle", "linkedin learning", "hackerrank",
    "freecodecamp", "great learning", "simplilearn", "internshala", "infosys",
    "tcs", "meta", "deeplearning.ai", "kaggle",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _skill_regex(alias: str) -> re.Pattern:
    """Build a word-boundary-safe regex for a skill alias.

    Normal \\b fails for tokens like 'c++', 'c#', '.net', so custom
    lookarounds are used instead.
    """
    escaped = re.escape(alias)
    return re.compile(
        r"(?<![a-z0-9+#./-])" + escaped + r"(?![a-z0-9+#])",
        re.IGNORECASE,
    )


def _find_sections(lines):
    """Map section name -> (start_line_index, end_line_index) based on headers.

    A line is treated as a section header if it is short and matches a
    known section pattern.
    """
    headers = []  # (line_index, section_name)
    for i, line in enumerate(lines):
        stripped = line.strip().strip(":•-_=|#").strip()
        if not stripped or len(stripped) > 40:
            continue
        low = stripped.lower()
        for name, pat in SECTION_PATTERNS.items():
            if re.fullmatch(pat + r"s?\s*:?", low) or re.match(r"^" + pat, low):
                # require the header line to be mostly just the header word(s)
                if len(low.split()) <= 4:
                    headers.append((i, name))
                    break

    sections = {}
    for idx, (line_i, name) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        if name not in sections:  # keep the first occurrence
            sections[name] = (line_i, end)
    return sections


def _section_text(lines, sections, name):
    if name not in sections:
        return ""
    start, end = sections[name]
    return "\n".join(lines[start + 1:end]).strip()


def _clean_line(line: str) -> str:
    return re.sub(r"^[\s•●▪·*\-–—o>]+", "", line).strip()


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------
def extract_contact_info(text: str) -> dict:
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.search(r"(\+?\d{1,3}[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", text)
    linkedin = re.search(r"linkedin\.com/[A-Za-z0-9_/.\-]+|linkedin", text, re.I)
    github = re.search(r"github\.com/[A-Za-z0-9_/.\-]+|github", text, re.I)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0).strip() if phone else None,
        "linkedin": bool(linkedin),
        "github": bool(github),
    }


def detect_skills(text: str):
    """Return (flat_skill_list, skills_by_category)."""
    low = text.lower()
    found_flat, found_by_cat = [], {}
    for category, skills in SKILL_DB.items():
        for skill_name, aliases in skills.items():
            for alias in aliases:
                if _skill_regex(alias).search(low):
                    found_by_cat.setdefault(category, []).append(skill_name)
                    found_flat.append(skill_name)
                    break
    return found_flat, found_by_cat


def extract_education(text: str, lines, sections):
    """Return a list of education entry dicts. Never invents data."""
    edu_text = _section_text(lines, sections, "education") or text
    edu_lines = [l for l in edu_text.split("\n") if l.strip()]

    entries = []
    seen_degrees = set()
    for i, raw in enumerate(edu_lines):
        line = _clean_line(raw)
        low = line.lower()
        for pat, degree in DEGREE_PATTERNS:
            if re.search(pat, low) and degree not in seen_degrees:
                seen_degrees.add(degree)
                # look in this line and the next two lines for details
                context = " | ".join(
                    _clean_line(x) for x in edu_lines[i:i + 3]
                )
                clow = context.lower()

                field = next((f.title() for f in FIELD_PATTERNS
                              if re.search(r"\b" + re.escape(f) + r"\b", clow)), None)

                inst = re.search(
                    r"([A-Z][A-Za-z&.,'() ]{3,70}"
                    r"(university|institute|college|school|academy|iit|nit|iiit|vidyalaya|polytechnic))",
                    context, re.IGNORECASE)

                # prefer years found on the same line as the degree;
                # only fall back to the wider context if none are there
                years = re.findall(r"\b((?:19|20)\d{2})\b", line)
                if not years:
                    years = re.findall(r"\b((?:19|20)\d{2})\b", context)

                score = re.search(
                    r"(cgpa|gpa|percentage|aggregate|sgpa)\s*[:\-]?\s*([\d.]+\s*%?)|([\d.]{1,5})\s*(cgpa|gpa|%)",
                    clow)

                entries.append({
                    "degree": degree,
                    "field": field if field else "Not detected",
                    "institution": inst.group(1).strip() if inst else "Not detected",
                    "year": " - ".join(years[:2]) if years else "Not detected",
                    "score": (score.group(0).upper().strip() if score else "Not detected"),
                })
                break
    return entries


def extract_experience(text: str, lines, sections):
    """Return list of experience entry dicts. Never invents data."""
    exp_text = _section_text(lines, sections, "experience")
    entries = []

    source_lines = exp_text.split("\n") if exp_text else lines
    for i, raw in enumerate(source_lines):
        line = _clean_line(raw)
        if not line or len(line) > 120:
            continue
        if re.search(JOB_TITLE_PATTERN, line, re.IGNORECASE):
            # skip obvious non-entries (bullet descriptions are usually long)
            if not exp_text and not (
                DURATION_PATTERN.search(line)
                or (i + 1 < len(source_lines)
                    and DURATION_PATTERN.search(source_lines[i + 1]))
                or re.search(r"\bat\b|@|,", line)
            ):
                continue

            context = " ".join(source_lines[i:i + 2])
            dur = DURATION_PATTERN.search(context)

            # company guess: words after 'at'/'@'/',' in the same line
            comp = re.search(
                r"(?:\bat\b|@)\s*([A-Z][A-Za-z0-9&.,' \-]{2,60})", line)

            title_match = re.search(
                r"([A-Za-z/&\- ]*" + JOB_TITLE_PATTERN + r")", line, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else line[:60]
            # cut off anything after 'at' / '@' (that's the company part)
            title = re.split(r"\s+(?:at|@)\s*$|\s+(?:at|@)\s+", title, flags=re.IGNORECASE)[0].strip().title()

            entry = {
                "title": title,
                "company": comp.group(1).strip().rstrip(",.") if comp else "Not detected",
                "duration": dur.group(0) if dur else "Not detected",
            }
            if entry not in entries:
                entries.append(entry)
        if len(entries) >= 8:
            break
    return entries


def extract_projects(text: str, lines, sections):
    """Return list of project dicts from the projects section only."""
    proj_text = _section_text(lines, sections, "projects")
    if not proj_text:
        return []

    proj_lines = [l for l in proj_text.split("\n") if l.strip()]
    projects, current = [], None

    for raw in proj_lines:
        is_bullet = bool(re.match(r"^[\s•●▪·*\-–—>o]", raw)) and len(raw.strip()) > 2
        line = _clean_line(raw)
        if not line:
            continue
        # A title line: short-ish, not a bullet, often contains '|' , ':' or '–'
        looks_like_title = (not is_bullet and len(line) <= 90
                            and not line.endswith(".")
                            and len(line.split()) <= 12)
        if looks_like_title:
            if current:
                projects.append(current)
            name = re.split(r"\s*[|:–—]\s*", line)[0].strip()
            current = {"name": name, "description": "", "technologies": []}
            tech, _ = detect_skills(line)
            current["technologies"] = tech
        elif current:
            current["description"] = (current["description"] + " " + line).strip()
            tech, _ = detect_skills(line)
            for t in tech:
                if t not in current["technologies"]:
                    current["technologies"].append(t)
        else:
            # description text before any title line
            current = {"name": line[:60], "description": "", "technologies": []}

    if current:
        projects.append(current)

    # truncate long descriptions for display
    for p in projects:
        if len(p["description"]) > 300:
            p["description"] = p["description"][:297] + "..."
        if not p["description"]:
            p["description"] = "No description provided."
    return projects[:8]


def extract_certifications(text: str, lines, sections):
    """Return list of certification strings. Never invents data."""
    cert_text = _section_text(lines, sections, "certifications")
    certs = []

    if cert_text:
        for raw in cert_text.split("\n"):
            line = _clean_line(raw)
            if 3 < len(line) <= 150:
                certs.append(line)
    else:
        # fall back: scan whole text for certification-like lines
        for raw in lines:
            line = _clean_line(raw)
            low = line.lower()
            if 3 < len(line) <= 150 and (
                "certif" in low
                or any(p in low for p in CERT_PROVIDERS)
                and ("course" in low or "training" in low or "certificate" in low)
            ):
                certs.append(line)

    # de-duplicate, keep order
    seen, unique = set(), []
    for c in certs:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique[:10]


# ---------------------------------------------------------------------------
# Career category suggestion (TF-IDF + cosine similarity, with fallback)
# ---------------------------------------------------------------------------
def suggest_careers(text: str, skills: list):
    low = text.lower()
    skill_set = {s.lower() for s in skills}

    overlap_scores = {}
    for career, keywords in CAREER_PROFILES.items():
        hits = sum(
            1 for kw in keywords
            if kw in skill_set or _skill_regex(kw).search(low)
        )
        overlap_scores[career] = hits / len(keywords)

    tfidf_scores = {c: 0.0 for c in CAREER_PROFILES}
    if _SKLEARN_AVAILABLE:
        try:
            docs = [" ".join(kws * 3) for kws in CAREER_PROFILES.values()]
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            matrix = vec.fit_transform(docs + [low])
            sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            for career, s in zip(CAREER_PROFILES.keys(), sims):
                tfidf_scores[career] = float(s)
        except Exception:
            pass

    combined = {
        c: 0.65 * overlap_scores[c] + 0.35 * tfidf_scores[c]
        for c in CAREER_PROFILES
    }
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

    suggestions = []
    for career, score in ranked[:3]:
        if score <= 0.05:
            continue
        matched = [
            kw for kw in CAREER_PROFILES[career]
            if kw in skill_set or _skill_regex(kw).search(low)
        ]
        suggestions.append({
            "career": career,
            "match_percent": min(99, round(score * 100)),
            "matched_keywords": [k.title() for k in matched[:8]],
        })

    if not suggestions:
        suggestions.append({
            "career": "Not enough information detected",
            "match_percent": 0,
            "matched_keywords": [],
        })
    return suggestions


# ---------------------------------------------------------------------------
# Scoring, strengths, weaknesses, suggestions
# ---------------------------------------------------------------------------
def _structure_metrics(text: str, lines):
    words = re.findall(r"\b\w+\b", text)
    bullets = sum(1 for l in lines if re.match(r"^[\s]*[•●▪·*\-–—>]", l))
    low = text.lower()
    verbs = [v for v in ACTION_VERBS if re.search(r"\b" + v + r"\b", low)]
    numbers = re.findall(r"\b\d+(\.\d+)?\s*(%|percent|\+|k\b|users|students|projects|hours|members)?", low)
    quantified = len(re.findall(r"\b\d+\s*(%|percent|\+)|\bincreased\b|\breduced\b|\bimproved\b.*\d", low))
    return {
        "word_count": len(words),
        "bullet_count": bullets,
        "action_verbs": verbs,
        "quantified_achievements": quantified,
        "number_mentions": len(numbers),
    }


def _compute_scores(contact, skills, education, experience, projects,
                    certifications, sections, metrics, career_suggestions):
    # ---------------- Overall score (out of 100) ----------------
    score = 0.0

    # Contact (10)
    score += 4 if contact["email"] else 0
    score += 3 if contact["phone"] else 0
    score += 1.5 if contact["linkedin"] else 0
    score += 1.5 if contact["github"] else 0

    # Skills (25) – full credit at 12+ distinct skills
    score += min(len(skills), 12) / 12 * 25

    # Education (15) – full credit at 2+ entries
    score += min(len(education), 2) / 2 * 15

    # Experience (15) – full credit at 2+ entries
    score += min(len(experience), 2) / 2 * 15

    # Projects (15) – full credit at 3+ projects
    score += min(len(projects), 3) / 3 * 15

    # Certifications (10) – full credit at 3+
    score += min(len(certifications), 3) / 3 * 10

    # Structure & content quality (10)
    struct = 0.0
    if 250 <= metrics["word_count"] <= 1100:
        struct += 3
    elif metrics["word_count"] >= 120:
        struct += 1.5
    if metrics["bullet_count"] >= 5:
        struct += 2
    elif metrics["bullet_count"] >= 1:
        struct += 1
    struct += min(len(metrics["action_verbs"]), 6) / 6 * 3
    if metrics["quantified_achievements"] >= 2:
        struct += 2
    elif metrics["quantified_achievements"] == 1:
        struct += 1
    score += struct

    overall = max(0, min(100, round(score)))

    # ---------------- ATS score (out of 100) ----------------
    # Section completeness (35)
    expected = ["skills", "education", "experience", "projects", "certifications", "summary"]
    present = [s for s in expected if s in sections]
    section_completeness = len(present) / len(expected)

    # Keyword coverage vs best-matching career profile (30)
    best_match = career_suggestions[0]["match_percent"] / 100 if career_suggestions else 0
    keyword_coverage = min(1.0, best_match * 1.25)

    # Contact parseability (15)
    contact_score = (
        (0.4 if contact["email"] else 0) + (0.3 if contact["phone"] else 0)
        + (0.15 if contact["linkedin"] else 0) + (0.15 if contact["github"] else 0)
    )

    # Readability / formatting (20)
    read = 0.0
    if 250 <= metrics["word_count"] <= 1100:
        read += 0.4
    elif metrics["word_count"] >= 120:
        read += 0.2
    if metrics["bullet_count"] >= 5:
        read += 0.3
    elif metrics["bullet_count"] >= 1:
        read += 0.15
    read += min(len(metrics["action_verbs"]), 5) / 5 * 0.3

    ats = round(
        section_completeness * 35
        + keyword_coverage * 30
        + contact_score * 15
        + read * 20
    )
    ats = max(0, min(100, ats))

    ats_details = {
        "section_completeness": round(section_completeness * 100),
        "keyword_coverage": round(keyword_coverage * 100),
        "contact_info": round(contact_score * 100),
        "readability": round(read * 100),
        "sections_found": [s.title() for s in present],
        "sections_missing": [s.title() for s in expected if s not in sections],
    }
    return overall, ats, ats_details


def _strengths(contact, skills, skills_by_cat, education, experience,
               projects, certifications, metrics):
    s = []
    if contact["email"] and contact["phone"]:
        s.append("Complete contact information (email and phone) is present.")
    if contact["linkedin"]:
        s.append("LinkedIn profile is included, which adds professional credibility.")
    if contact["github"]:
        s.append("GitHub profile is included, letting recruiters verify your code.")
    if len(skills) >= 10:
        s.append(f"Strong and diverse skill set — {len(skills)} technical/professional skills detected.")
    elif len(skills) >= 5:
        s.append(f"Good skill coverage — {len(skills)} relevant skills detected.")
    if len(skills_by_cat) >= 4:
        s.append(f"Skills span {len(skills_by_cat)} different areas (e.g., {', '.join(list(skills_by_cat.keys())[:3])}).")
    if education:
        s.append(f"Educational background is clearly mentioned ({len(education)} qualification(s) detected).")
    if experience:
        s.append(f"Practical exposure detected — {len(experience)} work/internship entry(ies) found.")
    if len(projects) >= 2:
        s.append(f"Hands-on project work is well represented ({len(projects)} projects detected).")
    elif len(projects) == 1:
        s.append("At least one project is included, showing practical application of skills.")
    if certifications:
        s.append(f"Certifications ({len(certifications)}) show initiative for continuous learning.")
    if len(metrics["action_verbs"]) >= 4:
        s.append("Good use of action verbs (e.g., " + ", ".join(metrics["action_verbs"][:4]) + ").")
    if metrics["quantified_achievements"] >= 2:
        s.append("Achievements are quantified with numbers, which recruiters value highly.")
    if not s:
        s.append("The resume was processed, but not enough content was detected to identify strengths. Consider adding more detail.")
    return s


def _weaknesses(contact, skills, education, experience, projects,
                certifications, sections, metrics):
    w = []
    if not contact["email"]:
        w.append("No email address detected — contact information is incomplete.")
    if not contact["phone"]:
        w.append("No phone number detected.")
    if not contact["linkedin"]:
        w.append("No LinkedIn profile mentioned.")
    if not contact["github"]:
        w.append("No GitHub (or portfolio) link mentioned.")
    if len(skills) == 0:
        w.append("No recognizable technical skills detected.")
    elif len(skills) < 5:
        w.append(f"Only {len(skills)} skills detected — the skills section looks thin.")
    if not education:
        w.append("No education details detected.")
    if not experience:
        w.append("No work experience or internships detected.")
    if not projects:
        w.append("No projects section detected — projects are crucial for freshers.")
    elif any(len(p["description"]) < 40 for p in projects):
        w.append("Some project descriptions are very short or missing details.")
    if not certifications:
        w.append("No certifications detected.")
    if "summary" not in sections:
        w.append("No summary/objective section detected at the top of the resume.")
    if metrics["quantified_achievements"] == 0:
        w.append("No measurable/quantified achievements found (numbers, percentages, impact).")
    if len(metrics["action_verbs"]) < 3:
        w.append("Few strong action verbs used — descriptions may sound passive.")
    if metrics["word_count"] < 200:
        w.append(f"Resume content is quite short ({metrics['word_count']} words) — it may look underdeveloped.")
    elif metrics["word_count"] > 1200:
        w.append(f"Resume is lengthy ({metrics['word_count']} words) — consider condensing to 1–2 pages.")
    if not w:
        w.append("No major weaknesses detected. Well done!")
    return w


def _suggestions(contact, skills, education, experience, projects,
                 certifications, sections, metrics):
    sug = []
    if not contact["email"] or not contact["phone"]:
        sug.append("Add complete contact details (professional email + phone) at the top of the resume.")
    if not contact["linkedin"] or not contact["github"]:
        sug.append("Add your LinkedIn and GitHub/portfolio links so recruiters can learn more about you.")
    if len(skills) < 8:
        sug.append("Expand the skills section with specific tools, languages, and frameworks you have used.")
    if not experience:
        sug.append("Add internships, freelance work, or volunteer roles — even small experiences count.")
    if len(projects) < 2:
        sug.append("Include 2–3 strong projects with the problem solved, technologies used, and results.")
    if projects and any(len(p["description"]) < 40 for p in projects):
        sug.append("Describe each project in 2–3 bullet points covering what you built and its impact.")
    if not certifications:
        sug.append("Add relevant certifications (Coursera, NPTEL, Udemy, AWS, Google, etc.) to strengthen credibility.")
    if metrics["quantified_achievements"] < 2:
        sug.append("Quantify achievements with numbers (e.g., 'improved accuracy by 15%', 'handled 500+ records').")
    if len(metrics["action_verbs"]) < 4:
        sug.append("Start bullet points with strong action verbs like 'developed', 'implemented', 'optimized'.")
    if "summary" not in sections:
        sug.append("Add a brief 2–3 line professional summary/objective at the top.")
    if metrics["bullet_count"] < 5:
        sug.append("Use bullet points instead of paragraphs for easier recruiter and ATS scanning.")
    if metrics["word_count"] > 1200:
        sug.append("Trim the resume to 1–2 pages; keep only the most relevant and recent information.")
    sug.append("Tailor keywords in the resume to each job description before applying to pass ATS filters.")
    return sug


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def analyze_resume(resume_text: str) -> dict:
    """Analyze extracted resume text and return a structured results dict.

    Nothing is invented: every field is derived from the actual text, and
    missing information is reported as missing/empty.
    """
    if not resume_text or not resume_text.strip():
        return {
            "error": "No text could be extracted from the PDF. "
                     "It may be a scanned/image-based resume.",
            "overall_score": 0, "ats_score": 0,
            "skills": [], "skills_by_category": {},
            "education": [], "experience": [], "projects": [],
            "certifications": [], "strengths": [],
            "weaknesses": ["No readable text found in the uploaded PDF."],
            "suggestions": ["Upload a text-based (not scanned) PDF resume."],
            "career_suggestions": [], "ats_details": {}, "contact": {},
            "word_count": 0,
        }

    text = resume_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")

    word_count = len(re.findall(r"\b\w+\b", text))
    insufficient = word_count < 50

    sections = _find_sections(lines)
    contact = extract_contact_info(text)
    skills, skills_by_cat = detect_skills(text)
    education = extract_education(text, lines, sections)
    experience = extract_experience(text, lines, sections)
    projects = extract_projects(text, lines, sections)
    certifications = extract_certifications(text, lines, sections)
    metrics = _structure_metrics(text, lines)
    career_suggestions = suggest_careers(text, skills)

    overall, ats, ats_details = _compute_scores(
        contact, skills, education, experience, projects,
        certifications, sections, metrics, career_suggestions)

    strengths = _strengths(contact, skills, skills_by_cat, education,
                           experience, projects, certifications, metrics)
    weaknesses = _weaknesses(contact, skills, education, experience,
                             projects, certifications, sections, metrics)
    suggestions = _suggestions(contact, skills, education, experience,
                               projects, certifications, sections, metrics)

    result = {
        "overall_score": overall,
        "ats_score": ats,
        "ats_details": ats_details,
        "contact": contact,
        "skills": skills,
        "skills_by_category": skills_by_cat,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "career_suggestions": career_suggestions,
        "word_count": word_count,
    }

    if insufficient:
        result["warning"] = (
            f"Very little text was extracted ({word_count} words). "
            "The analysis may be incomplete — please upload a proper text-based resume PDF."
        )
    return result


if __name__ == "__main__":
    # quick self-test
    import json
    sample = open("sample_resume.txt").read()
    print(json.dumps(analyze_resume(sample), indent=2))
