import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

st.set_page_config(page_title="ATS Resume Builder", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
APP_DATA_DIR = Path(".streamlit_resume_builder")
KNOWLEDGE_CACHE_PATH = APP_DATA_DIR / "knowledge.txt"
RESUME_CACHE_PATH = APP_DATA_DIR / "resume.txt"
OPENAI_API_KEY = ""

MODEL_OPTIONS = [
    {"id": "gpt-4o-mini",  "name": "GPT-4o Mini",   "group": "GPT-4",      "info": "Fastest and cheapest for quick drafts."},
    {"id": "gpt-4o",       "name": "GPT-4o",         "group": "GPT-4",      "info": "Balanced quality and speed for most resume edits."},
    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano",   "group": "GPT-4.1",   "info": "Lightweight option for short transformations."},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini",   "group": "GPT-4.1",   "info": "Good default for quality, speed, and cost."},
    {"id": "gpt-4.1",      "name": "GPT-4.1",        "group": "GPT-4.1",   "info": "Higher quality output for tighter tailoring."},
    {"id": "o4-mini",      "name": "o4-mini",         "group": "Reasoning",  "info": "Reasoning-heavy model for JD match analysis."},
    {"id": "o3",           "name": "o3",              "group": "Reasoning",  "info": "Strong reasoning for complex rewrites and constraints."},
    {"id": "gpt-5-nano",   "name": "GPT-5 Nano",      "group": "GPT-5",      "info": "Latest lightweight option for rapid iterations."},
    {"id": "gpt-5-mini",   "name": "GPT-5 Mini",      "group": "GPT-5",      "info": "Good default in GPT-5 family for daily use."},
    {"id": "gpt-5",        "name": "GPT-5",           "group": "GPT-5",      "info": "Best quality for final output and polish."},
]

# ---------------------------------------------------------------------------
# PDF DESIGN TOKENS
# ---------------------------------------------------------------------------
PDF_ACCENT      = colors.HexColor("#2e75b6")
PDF_DARK        = colors.HexColor("#000000")
PDF_MID         = colors.HexColor("#000000")
PDF_LIGHT       = colors.HexColor("#333333")
PDF_BG_SKILL    = colors.HexColor("#eef3f8")

FONT_REGULAR    = "Helvetica"
FONT_BOLD       = "Helvetica-Bold"
FONT_ITALIC     = "Helvetica-Oblique"
FONT_BOLDITALIC = "Helvetica-BoldOblique"

MARGIN          = 44
TOP_MARGIN      = 48
BOTTOM_MARGIN   = 36

SZ_NAME         = 14.0
SZ_CONTACT      = 7.5
SZ_SECTION      = 8.0
SZ_COMPANY      = 8.4
SZ_JOBTITLE     = 8.0
SZ_DATES        = 8.0
SZ_BULLET       = 8.0
SZ_SKILL_CAT    = 8.2
SZ_SKILL_ITEMS  = 8.0
SZ_BODY         = 8.2

LD_NAME         = 16.8
LD_CONTACT      = 10.2
LD_SECTION      = 12.0
LD_COMPANY      = 12.0
LD_JOBTITLE     = 10.8
LD_DATES        = 10.0
LD_BULLET       = 9.8
LD_SKILL        = 10.4
LD_BODY         = 10.4

BULLET_CHAR     = "\u2022"
INDENT_BULLET   = 14
SECTION_GAP     = 5

# ---------------------------------------------------------------------------
# FILE UTILITIES
# ---------------------------------------------------------------------------

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([(page.extract_text() or "") for page in reader.pages])

def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_uploaded_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):  return read_txt(uploaded_file)
    if name.endswith(".pdf"):  return read_pdf(uploaded_file)
    if name.endswith(".docx"): return read_docx(uploaded_file)
    raise ValueError("Unsupported format. Upload TXT, PDF, or DOCX.")

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ensure_data_dir():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_cached_text(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8", errors="ignore").strip() if path.exists() else None

def save_cached_text(path: Path, value: str):
    ensure_data_dir()
    path.write_text(value.strip(), encoding="utf-8")

def modified_label(path: Path) -> str:
    if not path.exists():
        return "Not saved yet"
    ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"Saved (last updated: {ts})"

def get_api_key() -> str:
    configured = OPENAI_API_KEY.strip()
    if configured and configured != "REPLACE_WITH_YOUR_OPENAI_API_KEY":
        return configured
    return os.getenv("OPENAI_API_KEY", "").strip()


def _slug_component(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or fallback


def _infer_role_company(job_description: str, summary: str) -> Tuple[str, str]:
    jd = job_description or ""
    role = ""
    company = ""

    role_patterns = [
        r"(?im)^\s*(?:job\s*title|title|role|position)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*we are hiring\s+(.+?)\s*$",
    ]
    for pattern in role_patterns:
        m = re.search(pattern, jd)
        if m:
            role = m.group(1).strip()
            break

    if not role and summary:
        m = re.match(r"^\s*([A-Za-z][A-Za-z/&\-\s]{2,60})\s+with\s+5\+\s+years", summary, flags=re.IGNORECASE)
        if m:
            role = m.group(1).strip()

    company_patterns = [
        r"(?im)^\s*(?:company|organization|employer)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*about\s+([A-Z][A-Za-z0-9& .,\-]{2,60})\s*$",
    ]
    for pattern in company_patterns:
        m = re.search(pattern, jd)
        if m:
            company = m.group(1).strip()
            break

    if not company:
        m = re.search(r"(?i)\bat\s+([A-Z][A-Za-z0-9& .,\-]{2,60})", jd)
        if m:
            company = m.group(1).strip()

    return role or "Role", company or "Company"

# ---------------------------------------------------------------------------
# LLM — JSON GENERATION
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert resume writer and career strategist. "
    "Output ONLY valid JSON matching the schema exactly. No prose, no markdown, no code fences. "
    "Use facts only from the candidate knowledge/memory and current resume. "
    "Do not invent employers, dates, degrees, projects, certifications, tools, or metrics. "
    "If a metric is unavailable, use directional impact language instead."
)

def build_user_prompt(knowledge_text, current_resume_text, job_description, additional_notes):
    return f"""
Create a one-page ATS-friendly resume tailored to this job description.

=== HARD RULES ===

CONTACT:
- name field: full name only, no title
- contact fields: city_state, email, phone, linkedin (no https://), github (no https://)

SUMMARY (exactly 2 concise sentences, written specifically for THIS job description):
- Read the JD carefully. Identify: (a) the exact job title, (b) the top 2-3 technical domains it emphasizes, (c) the primary deliverable or responsibility.
- Sentence 1: [Exact JD job title] with 5+ years building [domain A] and [domain B] systems — mirror the JD's own language for the domains.
- Sentence 2: One achievement that directly maps to the JD's PRIMARY responsibility — must include a real metric from the candidate background and mention concrete stack/tools.
- BANNED phrases: "adept at", "proven track record", "results-driven", "high-impact", "across disciplines", "passionate about", "extensive experience", "dynamic", "innovative", "seeking"
- EVERY word in the summary must be justified by JD content. Do not write generic summaries.

CORE SKILLS:
- Read the JD and extract the top 15 hard-skill keywords VERBATIM as they appear in the JD (exact same capitalization and phrasing).
- Every one of those 15 keywords MUST appear somewhere in the resume — in skills, bullets, or projects.
- Group into exactly 5-6 labeled categories relevant to the JD.
- No skill may appear in more than one category.
- Each category must have 4-8 items.
- Skills must be concrete technologies only: languages, frameworks, libraries, cloud services, databases, MLOps tools, APIs, protocols.
- Do NOT include vague/non-technical phrases such as "real time", "high traffic environments", "robust training", "continuous improvements", "rigorous metrics", "performance benchmarks", "structured human evaluation".

WORK EXPERIENCE:
- Section label must be "WORK EXPERIENCE" (not "Professional Experience").
- Reverse chronological order.
- Most recent role: exactly 5 bullets. All other roles: exactly 4 bullets each.
- BULLET LENGTH REQUIREMENT: keep bullets concise (18-28 words).
- Every bullet MUST cover ALL FOUR of these elements in ONE complete sentence:
    1. WHAT: the specific system, model, pipeline, or feature built
    2. HOW: exact technologies and methods used (name at least 2 tools)
    3. WHY: the business problem it solved or challenge it addressed
    4. RESULT: a quantified metric (number, %, time saved) OR a clear directional outcome with scope (e.g. "across 500+ vessels")
- GOOD EXAMPLE: "Rebuilt vessel diagnostics pipeline using **AWS Step Functions** and **Python** to replace manual batch processing, reducing turnaround by **98%** (from **8 hours to 10 minutes**) and enabling real-time analytics for **500+** maritime clients."
- BAD EXAMPLE — REJECT THIS: "Architected sensor pipelines using AWS to reduce latency by 60%." (only 12 words, missing WHY and scope)
- BAD EXAMPLE — REJECT THIS: "Built RAG system with LangChain and GPT-4o that reduced troubleshooting time by 45%." (no WHY, no scope, no context of what problem it solved)
- Wrap every technology name AND every number/metric in double asterisks: **Python**, **98%**, **500+**.
- No bullet may repeat a technology or metric already used in a prior bullet within the same role.
- Date format: Mon YYYY. End date can be "Present".

PROJECTS:
- Include only personal/independent projects. Never include work projects from Professional Experience.
- Exclude study courses, tutorials, bootcamps, or assigned coursework.
- Select 1-2 projects with highest overlap to JD domain and tools.
- For ML Engineer / Applied Scientist roles: always put PatchCamelyon CNN project first if evidence exists.
- Project name field: project name only (no company, no "Personal").
- Each project: exactly 2 bullets.
- Each project bullet must include: problem/context, implementation approach with concrete tools, and measurable result or scope.
- Same bold-wrapping rule applies (**tech**, **metric**).

EDUCATION: degree, school, location, start, end.

CERTIFICATIONS: name, issuer, and URL if available. Include the URL.

DATES: all dates Mon YYYY format. Never calculate total years — always write "5+ years".

=== OUTPUT JSON SCHEMA (return ONLY this, no other text) ===
{{
  "name": "Full Name",
  "contact": {{
    "city_state": "City, ST",
    "email": "email@example.com",
    "phone": "+1 (xxx) xxx-xxxx",
    "linkedin": "linkedin.com/in/...",
    "github": "github.com/..."
  }},
  "summary": "Two sentence string.",
  "skills": [
    {{"category": "Category Name", "items": ["skill1", "skill2", "skill3"]}}
  ],
  "experience": [
    {{
      "company": "Company Name",
      "title": "Job Title",
      "location": "City, ST",
      "start": "Mon YYYY",
      "end": "Mon YYYY",
      "bullets": ["bullet text with **Tech** and **metric**"]
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "bullets": ["bullet with **Tech** and **metric**"]
    }}
  ],
  "education": [
    {{
      "degree": "Degree Name",
      "school": "University Name",
      "location": "City, ST",
      "start": "Mon YYYY",
      "end": "Mon YYYY"
    }}
  ],
  "certifications": [
    {{"name": "Cert Name", "issuer": "Issuer", "url": "https://..."}}
  ]
}}

=== INPUT DATA ===
Candidate Knowledge / Memory:
{knowledge_text}

Current Resume:
{current_resume_text}

Job Description:
{job_description}

Additional Notes:
{additional_notes or "None"}
"""

def extract_json_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(text[start:end + 1])


def sanitize_resume_json(data: Dict[str, Any]) -> Dict[str, Any]:
    data.pop("match_notes", None)

    def clean_inline(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).replace("\n", " ").replace("\r", " ")).strip()

    summary = clean_inline(str(data.get("summary", "")))
    summary = re.sub(r'\b\d+\+?\s+years?\b', '5+ years', summary, flags=re.IGNORECASE)
    summary_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
    summary = " ".join(summary_sentences[:2]).strip()
    words = summary.split()
    if len(words) > 45:
        summary = " ".join(words[:45]).rstrip(",.;:") + "."
    data["summary"] = summary

    tech_hints = {
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm",
        "langchain", "llamaindex", "transformers", "bert", "gpt", "spacy", "nltk", "pandas", "numpy",
        "sql", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "databricks",
        "aws", "s3", "lambda", "ec2", "step functions", "sagemaker", "gcp", "azure",
        "docker", "kubernetes", "airflow", "mlflow", "wandb", "onnx", "triton", "cuda",
        "fastapi", "flask", "django", "react", "typescript", "javascript", "node", "github actions",
        "ci/cd", "rest", "grpc",
    }

    def is_concrete_tech(label: str) -> bool:
        low = label.lower()
        if any(tok in low for tok in tech_hints):
            return True
        if re.search(r"[+#/]|[0-9]", label):
            return True
        return False

    seen_skills: set = set()
    clean_skills = []
    non_tech_skill_phrases = {
        "real time",
        "latency sensitive",
        "high traffic environments",
        "robust training",
        "continuous improvements",
        "rigorous metrics",
        "performance benchmarks",
        "structured human evaluation",
        "reliability at scale",
        "post deployment monitoring",
        "deploy models",
        "production environments",
    }
    for cat in data.get("skills", []):
        cname = clean_inline(str(cat.get("category", "")))
        raw_items = [clean_inline(str(item)) for item in cat.get("items", []) if clean_inline(str(item))]
        items = []
        for label in raw_items:
            key = label.lower().strip("*")
            if key in non_tech_skill_phrases:
                continue
            if not is_concrete_tech(label):
                continue
            if label and key not in seen_skills:
                seen_skills.add(key)
                items.append(label)
        if not items:
            for label in raw_items[:2]:
                key = label.lower().strip("*")
                if label and key not in seen_skills:
                    seen_skills.add(key)
                    items.append(label)
        if cname and items:
            clean_skills.append({"category": cname, "items": items})
    data["skills"] = clean_skills

    for i, role in enumerate(data.get("experience", [])):
        role["company"] = clean_inline(role.get("company", ""))
        role["title"] = clean_inline(role.get("title", ""))
        role["location"] = clean_inline(role.get("location", ""))
        role["start"] = clean_inline(role.get("start", ""))
        role["end"] = clean_inline(role.get("end", ""))
        bullets = [clean_inline(str(b)) for b in role.get("bullets", []) if clean_inline(str(b))]
        target = 5 if i == 0 else 4
        role["bullets"] = bullets[:target]

    banned_terms = {"course", "tutorial", "bootcamp", "udemy", "coursera project",
                    "leetcode", "class assignment", "homework", "coursework"}
    projects = []
    for proj in data.get("projects", []):
        name = clean_inline(str(proj.get("name", "")))
        bullets = [clean_inline(str(b)) for b in proj.get("bullets", []) if clean_inline(str(b))]
        bullets = [b for b in bullets if len(b.split()) >= 12]
        blob = f"{name} {' '.join(bullets)}".lower()
        if any(t in blob for t in banned_terms):
            continue
        if name and bullets:
            projects.append({"name": name, "bullets": bullets[:2]})
    patch_idx = next((i for i, p in enumerate(projects)
                      if "patchcamelyon" in p["name"].lower()), None)
    if patch_idx is not None and patch_idx != 0:
        projects.insert(0, projects.pop(patch_idx))
    data["projects"] = projects[:3]

    certs = []
    for cert in data.get("certifications", []):
        name   = clean_inline(str(cert.get("name",   "")))
        issuer = clean_inline(str(cert.get("issuer", "")))
        url    = clean_inline(str(cert.get("url",    "")))
        if name:
            certs.append({"name": name, "issuer": issuer, "url": url})
    data["certifications"] = certs

    return data


def generate_resume_json(
    api_key: str,
    model: str,
    knowledge_text: str,
    current_resume_text: str,
    job_description: str,
    additional_notes: Optional[str] = None,
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)

    augmented_knowledge = (
        f"{knowledge_text}\n\n"
        "=== FIXED FACTS (always use these exactly) ===\n"
        "- ZeroNorth employment: Feb 2023 - Dec 2025\n"
        "- Total years of experience: always write '5+ years', never calculate from dates\n"
        "- Two distinct ZeroNorth pipeline achievements (never merge):\n"
        "  1) Sensor diagnostic AI agent: eliminated hardcoded logic across 200+ sensors, "
        "reduced diagnostic turnaround from hours to minutes (60% latency reduction)\n"
        "  2) HPM analytics pipeline: reduced runtime from 8 hours to 10 minutes (98% reduction)\n"
        "- Endera Systems: background verification company. Role: spaCy NLP + React.js dev.\n"
        "- Oracle Cerner: healthcare ML only. No NLP work at Cerner.\n"
    )

    user_prompt = build_user_prompt(
        augmented_knowledge, current_resume_text, job_description, additional_notes
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "").strip()
    parsed = extract_json_object(raw)
    return sanitize_resume_json(parsed)

# ---------------------------------------------------------------------------
# MATCH ANALYSIS
# ---------------------------------------------------------------------------

MATCH_SYSTEM_PROMPT = (
    "You are a senior technical recruiter and ATS specialist. "
    "Analyze how well a candidate's background matches a job description. "
    "Output ONLY valid JSON. No prose, no markdown, no code fences."
)

def generate_match_analysis(
    api_key: str,
    model: str,
    knowledge_text: str,
    current_resume_text: str,
    job_description: str,
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)

    prompt = f"""
Analyze this candidate's fit for the job description below.

Score the match from 0-100 based on:
- Keyword overlap (required skills vs candidate background): 40%
- Relevant experience depth and recency: 30%
- Domain/industry alignment: 15%
- Education and certifications: 15%

Return JSON with exactly this schema:
{{
  "overall_score": <integer 0-100>,
  "confidence_label": "<one of: Strong Match | Good Match | Partial Match | Weak Match>",
  "matched_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["keyword1", "keyword2"],
  "strengths": [
    {{"point": "short title", "evidence": "specific evidence from candidate background"}}
  ],
  "gaps": [
    {{"gap": "what is missing", "severity": "<Critical | Moderate | Minor>", "suggestion": "how to address it"}}
  ],
  "ats_verdict": "One sentence prediction of ATS pass/fail and why.",
  "keyword_coverage_pct": <integer 0-100>,
  "experience_alignment_pct": <integer 0-100>,
  "domain_alignment_pct": <integer 0-100>
}}

Rules:
- matched_keywords: only include terms that genuinely appear in the candidate background
- missing_keywords: only hard skills, tools, or certifications — not soft skills
- strengths: 3-5 items, each grounded in a specific achievement or metric
- gaps: 2-5 items ordered by severity (Critical first)
- Be honest — do not inflate scores

=== CANDIDATE BACKGROUND ===
{knowledge_text}

=== CURRENT RESUME ===
{current_resume_text}

=== JOB DESCRIPTION ===
{job_description}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        return extract_json_object(raw)
    except Exception as e:
        return {"error": str(e)}


def render_match_panel(analysis: Dict[str, Any]) -> None:
    if "error" in analysis:
        st.error(f"Match analysis failed: {analysis['error']}")
        return

    score       = analysis.get("overall_score", 0)
    label       = analysis.get("confidence_label", "Unknown")
    matched_kw  = analysis.get("matched_keywords", [])
    missing_kw  = analysis.get("missing_keywords", [])
    strengths   = analysis.get("strengths", [])
    gaps        = analysis.get("gaps", [])
    ats_verdict = analysis.get("ats_verdict", "")
    kw_pct      = analysis.get("keyword_coverage_pct", 0)
    exp_pct     = analysis.get("experience_alignment_pct", 0)
    domain_pct  = analysis.get("domain_alignment_pct", 0)

    if score >= 80:
        score_color = "#1a7a4a"
        bar_color   = "#2ecc71"
    elif score >= 60:
        score_color = "#b07d00"
        bar_color   = "#f39c12"
    else:
        score_color = "#c0392b"
        bar_color   = "#e74c3c"

    severity_colors = {"Critical": "#e74c3c", "Moderate": "#f39c12", "Minor": "#3498db"}
    severity_icons  = {"Critical": "🔴",       "Moderate": "🟡",      "Minor":  "🔵"}

    st.markdown("### 🎯 JD Match Analysis")

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        st.markdown(
            f"""<div style="background:{score_color}18;border:2px solid {score_color};border-radius:12px;
                padding:16px 10px;text-align:center;">
                <div style="font-size:2.8rem;font-weight:800;color:{score_color};line-height:1">{score}</div>
                <div style="font-size:0.75rem;color:{score_color};font-weight:600;margin-top:4px">/100</div>
                <div style="font-size:0.8rem;color:#444;margin-top:6px;font-weight:500">{label}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**Score Breakdown**")
        st.markdown(
            f"""<div style="margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>Keyword Coverage</span><span><b>{kw_pct}%</b></span></div>
                <div style="background:#e0e0e0;border-radius:4px;height:8px">
                    <div style="background:{bar_color};width:{kw_pct}%;height:8px;border-radius:4px"></div></div></div>
            <div style="margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>Experience Alignment</span><span><b>{exp_pct}%</b></span></div>
                <div style="background:#e0e0e0;border-radius:4px;height:8px">
                    <div style="background:{bar_color};width:{exp_pct}%;height:8px;border-radius:4px"></div></div></div>
            <div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>Domain Alignment</span><span><b>{domain_pct}%</b></span></div>
                <div style="background:#e0e0e0;border-radius:4px;height:8px">
                    <div style="background:{bar_color};width:{domain_pct}%;height:8px;border-radius:4px"></div></div></div>""",
            unsafe_allow_html=True)

    with c3:
        st.markdown("**ATS Prediction**")
        st.info(ats_verdict, icon="🤖")

    st.markdown("---")

    kc1, kc2 = st.columns(2)
    with kc1:
        st.markdown(f"**✅ Matched Keywords** &nbsp; `{len(matched_kw)} found`", unsafe_allow_html=True)
        if matched_kw:
            pills = " ".join(
                f'<span style="background:#d4edda;color:#155724;padding:2px 8px;border-radius:12px;font-size:0.78rem;margin:2px;display:inline-block">{kw}</span>'
                for kw in matched_kw)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None detected.")

    with kc2:
        st.markdown(f"**❌ Missing Keywords** &nbsp; `{len(missing_kw)} gaps`", unsafe_allow_html=True)
        if missing_kw:
            pills = " ".join(
                f'<span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:12px;font-size:0.78rem;margin:2px;display:inline-block">{kw}</span>'
                for kw in missing_kw)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No critical gaps found.")

    st.markdown("---")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**💪 What Aligns**")
        for item in strengths:
            point    = item.get("point", "")
            evidence = item.get("evidence", "")
            st.markdown(
                f"""<div style="background:#f0f9f0;border-left:3px solid #2ecc71;
                    padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:8px">
                    <div style="font-size:0.85rem;font-weight:600;color:#1a5c2a">{point}</div>
                    <div style="font-size:0.78rem;color:#444;margin-top:2px">{evidence}</div>
                </div>""", unsafe_allow_html=True)

    with sc2:
        st.markdown("**⚠️ Gaps & How to Address**")
        for item in gaps:
            gap        = item.get("gap", "")
            severity   = item.get("severity", "Minor")
            suggestion = item.get("suggestion", "")
            color      = severity_colors.get(severity, "#3498db")
            icon       = severity_icons.get(severity, "🔵")
            st.markdown(
                f"""<div style="background:#fff8f0;border-left:3px solid {color};
                    padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:8px">
                    <div style="font-size:0.85rem;font-weight:600;color:{color}">{icon} {gap}</div>
                    <div style="font-size:0.78rem;color:#666;margin-top:2px;font-style:italic">💡 {suggestion}</div>
                </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PDF LAYOUT HELPERS
# ---------------------------------------------------------------------------

def _bold_segments(text: str) -> List[Tuple[str, bool]]:
    parts = re.split(r'(\*\*.*?\*\*)', text)
    segments = []
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            segments.append((part[2:-2], True))
        elif part:
            segments.append((part, False))
    return segments


def _text_width(text: str, font: str, size: float) -> float:
    return pdfmetrics.stringWidth(text, font, size)


def _measure_rich_text_lines(
    text: str,
    indent_x: float,
    max_width: float,
    size: float,
) -> int:
    """Return number of wrapped lines for a rich-text string."""
    segments = _bold_segments(text)
    tokens = []
    for seg_text, is_bold in segments:
        font = FONT_BOLD if is_bold else FONT_REGULAR
        for word in seg_text.split(" "):
            if word:
                tokens.append((word + " ", font))
    if not tokens:
        return 1
    avail = max_width - (INDENT_BULLET if indent_x > MARGIN + 2 else 0)
    lines = 1
    current_w = 0.0
    for word, font in tokens:
        w = _text_width(word, font, size)
        if current_w + w > avail and current_w > 0:
            lines += 1
            current_w = w
        else:
            current_w += w
    return lines


def _measure_rich_text_height(
    text: str,
    x: float,
    max_width: float,
    size: float,
    leading: float,
) -> float:
    lines = _measure_rich_text_lines(text, x, max_width, size)
    return lines * leading


def _draw_rich_text(
    pdf: canvas.Canvas,
    text: str,
    x: float, y: float,
    regular_font: str, bold_font: str,
    size: float,
    max_width: float,
    color=None,
    bold_color=None,
) -> float:
    if color is None:      color = PDF_DARK
    if bold_color is None: bold_color = PDF_ACCENT

    segments = _bold_segments(text)
    tokens = []
    for seg_text, is_bold in segments:
        font = bold_font if is_bold else regular_font
        for word in seg_text.split(" "):
            if word:
                tokens.append((word + " ", font, is_bold))
    if not tokens:
        return y

    lines_out: List[List[Tuple[str, str, bool]]] = []
    current_line: List[Tuple[str, str, bool]] = []
    current_w = 0.0

    for word, font, is_bold in tokens:
        w = _text_width(word, font, size)
        indent = INDENT_BULLET if x > MARGIN + 2 else 0
        avail = max_width - indent
        if current_w + w > avail and current_line:
            lines_out.append(current_line)
            current_line = [(word, font, is_bold)]
            current_w = w
        else:
            current_line.append((word, font, is_bold))
            current_w += w
    if current_line:
        lines_out.append(current_line)

    leading = LD_BULLET if x > MARGIN + 2 else LD_BODY

    for li, line_tokens in enumerate(lines_out):
        cx = x
        draw_y = y - li * leading
        for word, font, is_bold in line_tokens:
            pdf.setFont(font, size)
            pdf.setFillColor(bold_color if is_bold else color)
            pdf.drawString(cx, draw_y, word)
            cx += _text_width(word, font, size)

    return y - (len(lines_out) - 1) * leading


def _draw_section_header(pdf: canvas.Canvas, text: str, y: float, page_width: float) -> float:
    y -= SECTION_GAP
    pdf.setFont(FONT_BOLD, SZ_SECTION)
    pdf.setFillColor(PDF_DARK)
    pdf.drawString(MARGIN, y, text.upper())
    y -= LD_SECTION * 0.35
    pdf.setStrokeColor(PDF_DARK)
    pdf.setLineWidth(0.4)
    pdf.line(MARGIN, y, page_width - MARGIN, y)
    y -= LD_SECTION * 0.55
    return y


def _draw_company_row(
    pdf: canvas.Canvas,
    company: str, title: str, location: str,
    start: str, end: str,
    y: float,
    page_width: float,
) -> float:
    date_str = f"{start} – {end}" if start and end else (start or end)
    loc_str  = f"  ·  {location}" if location else ""

    pdf.setFont(FONT_BOLD, SZ_COMPANY)
    pdf.setFillColor(PDF_DARK)
    pdf.drawString(MARGIN, y, company)

    pdf.setFont(FONT_ITALIC, SZ_DATES)
    pdf.setFillColor(PDF_LIGHT)
    date_w = _text_width(date_str, FONT_ITALIC, SZ_DATES)
    pdf.drawString(page_width - MARGIN - date_w, y, date_str)

    y -= LD_COMPANY
    pdf.setFont(FONT_ITALIC, SZ_JOBTITLE)
    pdf.setFillColor(PDF_MID)
    pdf.drawString(MARGIN, y, f"{title}{loc_str}")

    y -= LD_JOBTITLE * 0.8
    return y


def _draw_bullet(
    pdf: canvas.Canvas,
    text: str,
    y: float,
    max_width: float,
) -> float:
    pdf.setFont(FONT_REGULAR, SZ_BULLET)
    pdf.setFillColor(PDF_DARK)
    pdf.drawString(MARGIN, y, BULLET_CHAR)

    text_x = MARGIN + INDENT_BULLET
    _draw_rich_text(
        pdf, text,
        x=text_x, y=y,
        regular_font=FONT_REGULAR, bold_font=FONT_BOLD,
        size=SZ_BULLET,
        max_width=max_width - INDENT_BULLET,
        color=PDF_DARK,
        bold_color=PDF_DARK,
    )
    height = _measure_rich_text_height(text, text_x, max_width - INDENT_BULLET, SZ_BULLET, LD_BULLET)
    return y - height


# ---------------------------------------------------------------------------
# RENDER-BASED CONTENT TRIMMER
# Trim content progressively and check actual rendered page count each step.
# This is far more reliable than height estimation.
# ---------------------------------------------------------------------------

def _page_count(data: Dict[str, Any]) -> int:
    """Render the resume and return page count."""
    from pypdf import PdfReader as _PdfReader
    pdf_bytes = resume_json_to_pdf_bytes(data)
    reader = _PdfReader(BytesIO(pdf_bytes))
    return len(reader.pages)


def _trim_data_to_fit(data: Dict[str, Any], *_args) -> Dict[str, Any]:
    """
    Progressively trim resume content until it fits on one page.
    Uses actual PDF rendering to check fit — no height estimation.

    Trim priority (lowest-impact first):
      1. Oldest role (last): 4→3 bullets
      2. Oldest role: 3→2 bullets
      3. Drop projects beyond 2 (keep top 2)
      4. Middle role(s): 4→3 bullets
      5. Drop projects beyond 1 (keep top 1)
      6. Newest role: 5→4 bullets  ← last resort
    """
    import copy
    d = copy.deepcopy(data)

    if _page_count(d) == 1:
        return d

    exp = d.get("experience", [])

    # Step 1: Oldest role 4→3
    if len(exp) >= 3 and len(exp[-1]["bullets"]) > 3:
        exp[-1]["bullets"] = exp[-1]["bullets"][:3]
    if _page_count(d) == 1: return d

    # Step 2: Oldest role 3→2
    if len(exp) >= 3 and len(exp[-1]["bullets"]) > 2:
        exp[-1]["bullets"] = exp[-1]["bullets"][:2]
    if _page_count(d) == 1: return d

    # Step 3: Drop to 2 projects
    if len(d.get("projects", [])) > 2:
        d["projects"] = d["projects"][:2]
    if _page_count(d) == 1: return d

    # Step 4: Middle roles 4→3
    for i in range(1, len(exp) - 1):
        if len(exp[i]["bullets"]) > 3:
            exp[i]["bullets"] = exp[i]["bullets"][:3]
    if _page_count(d) == 1: return d

    # Step 5: Drop to 1 project
    if len(d.get("projects", [])) > 1:
        d["projects"] = d["projects"][:1]
    if _page_count(d) == 1: return d

    # Step 6: Newest role 5→4
    if exp and len(exp[0]["bullets"]) > 4:
        exp[0]["bullets"] = exp[0]["bullets"][:4]
    if _page_count(d) == 1: return d

    # Step 7: Middle roles 3→2
    for i in range(1, len(exp) - 1):
        if len(exp[i]["bullets"]) > 2:
            exp[i]["bullets"] = exp[i]["bullets"][:2]
    if _page_count(d) == 1: return d

    return d


# ---------------------------------------------------------------------------
# PDF RENDERING
# ---------------------------------------------------------------------------

def resume_json_to_pdf_bytes(data: Dict[str, Any]) -> bytes:
    """Render resume JSON to PDF bytes at fixed font size (no scaling)."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    W, H = LETTER
    max_w = W - 2 * MARGIN

    y = H - TOP_MARGIN

    # ── NAME
    name = str(data.get("name", "")).strip()
    pdf.setFont(FONT_BOLD, SZ_NAME)
    pdf.setFillColor(PDF_DARK)
    pdf.drawString(MARGIN, y, name)
    y -= LD_NAME

    # ── CONTACT
    contact = data.get("contact", {})
    parts = [p.strip() for p in [
        contact.get("city_state", ""),
        contact.get("email", ""),
        contact.get("phone", ""),
        contact.get("linkedin", ""),
        contact.get("github", ""),
    ] if str(p).strip()]
    contact_line = "  |  ".join(parts)

    contact_size = SZ_CONTACT
    while _text_width(contact_line, FONT_REGULAR, contact_size) > max_w and contact_size > 6.5:
        contact_size -= 0.25

    pdf.setFont(FONT_REGULAR, contact_size)
    pdf.setFillColor(PDF_MID)
    pdf.drawString(MARGIN, y, contact_line)
    y -= contact_size + 5

    # ── SUMMARY
    summary = str(data.get("summary", "")).strip()
    if summary:
        y = _draw_section_header(pdf, "PROFESSIONAL SUMMARY", y, W)
        summary_size = SZ_BODY - 0.7
        summary_leading = LD_BODY - 0.8
        summary_paragraph = re.sub(r"\s+", " ", summary).strip()
        h = _measure_rich_text_height(summary_paragraph, MARGIN, max_w, summary_size, summary_leading)
        _draw_rich_text(
            pdf, summary_paragraph,
            x=MARGIN, y=y,
            regular_font=FONT_REGULAR, bold_font=FONT_BOLD,
            size=summary_size,
            max_width=max_w,
            color=PDF_DARK,
            bold_color=PDF_DARK,
        )
        y -= h + 1.5

    # ── CORE SKILLS
    skills = data.get("skills", [])
    if skills:
        y = _draw_section_header(pdf, "CORE SKILLS", y, W)
        for cat in skills:
            cname = str(cat.get("category", "")).strip()
            items = [str(i).strip() for i in cat.get("items", []) if str(i).strip()]
            if not cname or not items:
                continue
            label = f"{cname}: "
            label_w = _text_width(label, FONT_BOLD, SZ_SKILL_CAT)
            items_str = ", ".join(items)

            pdf.setFont(FONT_BOLD, SZ_SKILL_CAT)
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, label)

            pdf.setFont(FONT_REGULAR, SZ_SKILL_ITEMS)
            pdf.setFillColor(PDF_MID)
            avail_w = max_w - label_w
            if _text_width(items_str, FONT_REGULAR, SZ_SKILL_ITEMS) <= avail_w:
                pdf.drawString(MARGIN + label_w, y, items_str)
                y -= LD_SKILL
            else:
                words = items_str.split(", ")
                line = ""
                first = True
                for word in words:
                    candidate = f"{line}, {word}" if line else word
                    avail = avail_w if first else max_w - label_w
                    if _text_width(candidate, FONT_REGULAR, SZ_SKILL_ITEMS) <= avail:
                        line = candidate
                    else:
                        draw_x = MARGIN + label_w if first else MARGIN + label_w
                        pdf.drawString(draw_x, y, line)
                        y -= LD_SKILL
                        line = word
                        first = False
                if line:
                    pdf.drawString(MARGIN + label_w, y, line)
                    y -= LD_SKILL

    # ── WORK EXPERIENCE
    experience = data.get("experience", [])
    if experience:
        y = _draw_section_header(pdf, "WORK EXPERIENCE", y, W)
        for role in experience:
            company  = str(role.get("company",  "")).strip()
            title    = str(role.get("title",    "")).strip()
            location = str(role.get("location", "")).strip()
            start    = str(role.get("start",    "")).strip()
            end      = str(role.get("end",      "")).strip()
            bullets  = role.get("bullets", [])

            y = _draw_company_row(pdf, company, title, location, start, end, y, W)

            for b in bullets:
                y = _draw_bullet(pdf, str(b).strip(), y, max_w)
                y -= 1.5

            y -= 3

    # ── PROJECTS
    projects = data.get("projects", [])
    if projects:
        y = _draw_section_header(pdf, "PROJECTS", y, W)
        for proj in projects:
            name    = str(proj.get("name", "")).strip()
            bullets = proj.get("bullets", [])

            pdf.setFont(FONT_BOLD, SZ_COMPANY)
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, name)
            y -= LD_COMPANY * 0.85

            for b in bullets:
                y = _draw_bullet(pdf, str(b).strip(), y, max_w)
                y -= 1.5
            y -= 3

    # ── EDUCATION
    education = data.get("education", [])
    if education:
        y = _draw_section_header(pdf, "EDUCATION", y, W)
        for edu in education:
            degree   = str(edu.get("degree",   "")).strip()
            school   = str(edu.get("school",   "")).strip()
            location = str(edu.get("location", "")).strip()
            start    = str(edu.get("start",    "")).strip()
            end      = str(edu.get("end",      "")).strip()
            date_str = f"{start} – {end}" if start and end else (start or end)

            pdf.setFont(FONT_BOLD, SZ_BODY + 0.5)
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, degree)

            pdf.setFont(FONT_ITALIC, SZ_DATES)
            pdf.setFillColor(PDF_LIGHT)
            dw = _text_width(date_str, FONT_ITALIC, SZ_DATES)
            pdf.drawString(W - MARGIN - dw, y, date_str)

            y -= LD_BODY
            school_loc = f"{school}  ·  {location}" if location else school
            pdf.setFont(FONT_REGULAR, SZ_BODY)
            pdf.setFillColor(PDF_MID)
            pdf.drawString(MARGIN, y, school_loc)
            y -= LD_BODY + 2

    # ── CERTIFICATIONS
    certs = data.get("certifications", [])
    if certs:
        y = _draw_section_header(pdf, "CERTIFICATIONS", y, W)
        for cert in certs:
            cname  = str(cert.get("name",   "")).strip()
            issuer = str(cert.get("issuer", "")).strip()
            url    = str(cert.get("url",    "")).strip()
            line = f"{cname}  |  {issuer}" if issuer else cname
            if url:
                line = f"{line}  |  {url}"

            cert_size = SZ_BODY
            while _text_width(line, FONT_REGULAR, cert_size) > max_w and cert_size > 6.3:
                cert_size -= 0.2

            pdf.setFont(FONT_REGULAR, cert_size)
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, line)
            if url:
                link = url if url.startswith("http") else f"https://{url}"
                prefix = f"{cname}  |  {issuer}  | " if issuer else f"{cname}  | "
                x1 = MARGIN + _text_width(prefix, FONT_REGULAR, cert_size)
                x2 = x1 + _text_width(url, FONT_REGULAR, cert_size)
                pdf.linkURL(link, (x1, y - 2, x2, y + cert_size), relative=0)
            y -= LD_BODY + 2

    pdf.save()
    result = buffer.getvalue()
    buffer.close()
    return result


def auto_fit_pdf(data: Dict[str, Any]) -> bytes:
    """
    Fit resume to one page.
    1. Try full content at font scale 1.0 (most content, no trim needed in most cases)
    2. If overflow, apply render-based smart trimming (preserves all complete bullets)
    3. Only if trimming still can't fix it, apply mild font scaling as last resort
    """
    from pypdf import PdfReader as _PdfReader

    # Step 1: Try full content at full size
    pdf_bytes = resume_json_to_pdf_bytes(data)
    if len(_PdfReader(BytesIO(pdf_bytes)).pages) == 1:
        return pdf_bytes

    # Step 2: Smart content trimming (render-based, accurate)
    trimmed = _trim_data_to_fit(data)
    pdf_bytes = resume_json_to_pdf_bytes(trimmed)
    if len(_PdfReader(BytesIO(pdf_bytes)).pages) == 1:
        return pdf_bytes

    # Step 3: Font scaling as true last resort
    for scale in [0.97, 0.94, 0.91, 0.88, 0.85, 0.82]:
        pdf_bytes = _resume_json_to_pdf_bytes_scaled(trimmed, font_scale=scale)
        if len(_PdfReader(BytesIO(pdf_bytes)).pages) == 1:
            return pdf_bytes

    return _resume_json_to_pdf_bytes_scaled(trimmed, font_scale=0.79)


def _resume_json_to_pdf_bytes_scaled(data: Dict[str, Any], font_scale: float = 1.0) -> bytes:
    """
    Fallback scaled renderer — only used when smart trimming isn't enough.
    Identical to resume_json_to_pdf_bytes but with scaled fonts/leading.
    """
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    W, H = LETTER
    max_w = W - 2 * MARGIN

    def s(base): return base * font_scale
    def l(base): return base * font_scale

    y = H - TOP_MARGIN

    name = str(data.get("name", "")).strip()
    pdf.setFont(FONT_BOLD, s(SZ_NAME))
    pdf.setFillColor(PDF_DARK)
    pdf.drawString(MARGIN, y, name)
    y -= l(LD_NAME)

    contact = data.get("contact", {})
    parts = [p.strip() for p in [
        contact.get("city_state", ""), contact.get("email", ""),
        contact.get("phone", ""), contact.get("linkedin", ""), contact.get("github", ""),
    ] if str(p).strip()]
    contact_line = "  |  ".join(parts)
    contact_size = s(SZ_CONTACT)
    while _text_width(contact_line, FONT_REGULAR, contact_size) > max_w and contact_size > 6.5:
        contact_size -= 0.25
    pdf.setFont(FONT_REGULAR, contact_size)
    pdf.setFillColor(PDF_MID)
    pdf.drawString(MARGIN, y, contact_line)
    y -= contact_size + 5

    def draw_section_header_scaled(text, y):
        y -= SECTION_GAP
        pdf.setFont(FONT_BOLD, s(SZ_SECTION))
        pdf.setFillColor(PDF_DARK)
        pdf.drawString(MARGIN, y, text.upper())
        y -= l(LD_SECTION) * 0.35
        pdf.setStrokeColor(PDF_DARK)
        pdf.setLineWidth(0.4)
        pdf.line(MARGIN, y, W - MARGIN, y)
        y -= l(LD_SECTION) * 0.55
        return y

    summary = str(data.get("summary", "")).strip()
    if summary:
        y = draw_section_header_scaled("PROFESSIONAL SUMMARY", y)
        sz = max(6.8, s(SZ_BODY - 0.7))
        ld = max(8.5, l(LD_BODY - 0.8))
        h = _measure_rich_text_height(summary, MARGIN, max_w, sz, ld)
        _draw_rich_text(pdf, summary, MARGIN, y, FONT_REGULAR, FONT_BOLD, sz, max_w, PDF_DARK, PDF_DARK)
        y -= h + 1.5

    skills = data.get("skills", [])
    if skills:
        y = draw_section_header_scaled("CORE SKILLS", y)
        for cat in skills:
            cname = str(cat.get("category", "")).strip()
            items = [str(i).strip() for i in cat.get("items", []) if str(i).strip()]
            if not cname or not items: continue
            label = f"{cname}: "
            label_w = _text_width(label, FONT_BOLD, s(SZ_SKILL_CAT))
            items_str = ", ".join(items)
            pdf.setFont(FONT_BOLD, s(SZ_SKILL_CAT))
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, label)
            pdf.setFont(FONT_REGULAR, s(SZ_SKILL_ITEMS))
            pdf.setFillColor(PDF_MID)
            if _text_width(items_str, FONT_REGULAR, s(SZ_SKILL_ITEMS)) <= max_w - label_w:
                pdf.drawString(MARGIN + label_w, y, items_str)
                y -= l(LD_SKILL)
            else:
                words = items_str.split(", ")
                line = ""
                first = True
                for word in words:
                    candidate = f"{line}, {word}" if line else word
                    avail = max_w - label_w
                    if _text_width(candidate, FONT_REGULAR, s(SZ_SKILL_ITEMS)) <= avail:
                        line = candidate
                    else:
                        pdf.drawString(MARGIN + label_w, y, line)
                        y -= l(LD_SKILL)
                        line = word
                        first = False
                if line:
                    pdf.drawString(MARGIN + label_w, y, line)
                    y -= l(LD_SKILL)

    experience = data.get("experience", [])
    if experience:
        y = draw_section_header_scaled("WORK EXPERIENCE", y)
        for role in experience:
            company  = str(role.get("company",  "")).strip()
            title    = str(role.get("title",    "")).strip()
            location = str(role.get("location", "")).strip()
            start    = str(role.get("start",    "")).strip()
            end      = str(role.get("end",      "")).strip()
            date_str = f"{start} – {end}" if start and end else (start or end)
            loc_str  = f"  ·  {location}" if location else ""

            pdf.setFont(FONT_BOLD, s(SZ_COMPANY))
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, company)
            dw = _text_width(date_str, FONT_ITALIC, s(SZ_DATES))
            pdf.setFont(FONT_ITALIC, s(SZ_DATES))
            pdf.setFillColor(PDF_LIGHT)
            pdf.drawString(W - MARGIN - dw, y, date_str)
            y -= l(LD_COMPANY)
            pdf.setFont(FONT_ITALIC, s(SZ_JOBTITLE))
            pdf.setFillColor(PDF_MID)
            pdf.drawString(MARGIN, y, f"{title}{loc_str}")
            y -= l(LD_JOBTITLE) * 0.8

            for b in role.get("bullets", []):
                pdf.setFont(FONT_REGULAR, s(SZ_BULLET))
                pdf.setFillColor(PDF_DARK)
                pdf.drawString(MARGIN, y, BULLET_CHAR)
                text_x = MARGIN + INDENT_BULLET
                _draw_rich_text(pdf, str(b).strip(), text_x, y, FONT_REGULAR, FONT_BOLD,
                                s(SZ_BULLET), max_w - INDENT_BULLET, PDF_DARK, PDF_DARK)
                h = _measure_rich_text_height(str(b).strip(), text_x, max_w - INDENT_BULLET, s(SZ_BULLET), l(LD_BULLET))
                y -= h + 1.5
            y -= 3

    projects = data.get("projects", [])
    if projects:
        y = draw_section_header_scaled("PROJECTS", y)
        for proj in projects:
            name    = str(proj.get("name", "")).strip()
            pdf.setFont(FONT_BOLD, s(SZ_COMPANY))
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, name)
            y -= l(LD_COMPANY) * 0.85
            for b in proj.get("bullets", []):
                pdf.setFont(FONT_REGULAR, s(SZ_BULLET))
                pdf.setFillColor(PDF_DARK)
                pdf.drawString(MARGIN, y, BULLET_CHAR)
                text_x = MARGIN + INDENT_BULLET
                _draw_rich_text(pdf, str(b).strip(), text_x, y, FONT_REGULAR, FONT_BOLD,
                                s(SZ_BULLET), max_w - INDENT_BULLET, PDF_DARK, PDF_DARK)
                h = _measure_rich_text_height(str(b).strip(), text_x, max_w - INDENT_BULLET, s(SZ_BULLET), l(LD_BULLET))
                y -= h + 1.5
            y -= 3

    education = data.get("education", [])
    if education:
        y = draw_section_header_scaled("EDUCATION", y)
        for edu in education:
            degree   = str(edu.get("degree",   "")).strip()
            school   = str(edu.get("school",   "")).strip()
            location = str(edu.get("location", "")).strip()
            start    = str(edu.get("start",    "")).strip()
            end      = str(edu.get("end",      "")).strip()
            date_str = f"{start} – {end}" if start and end else (start or end)
            pdf.setFont(FONT_BOLD, s(SZ_BODY + 0.5))
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, degree)
            pdf.setFont(FONT_ITALIC, s(SZ_DATES))
            pdf.setFillColor(PDF_LIGHT)
            dw = _text_width(date_str, FONT_ITALIC, s(SZ_DATES))
            pdf.drawString(W - MARGIN - dw, y, date_str)
            y -= l(LD_BODY)
            school_loc = f"{school}  ·  {location}" if location else school
            pdf.setFont(FONT_REGULAR, s(SZ_BODY))
            pdf.setFillColor(PDF_MID)
            pdf.drawString(MARGIN, y, school_loc)
            y -= l(LD_BODY) + 2

    certs = data.get("certifications", [])
    if certs:
        y = draw_section_header_scaled("CERTIFICATIONS", y)
        for cert in certs:
            cname  = str(cert.get("name",   "")).strip()
            issuer = str(cert.get("issuer", "")).strip()
            url    = str(cert.get("url",    "")).strip()
            line = f"{cname}  |  {issuer}" if issuer else cname
            if url:
                line = f"{line}  |  {url}"

            cert_size = s(SZ_BODY)
            while _text_width(line, FONT_REGULAR, cert_size) > max_w and cert_size > 6.0:
                cert_size -= 0.2

            pdf.setFont(FONT_REGULAR, cert_size)
            pdf.setFillColor(PDF_DARK)
            pdf.drawString(MARGIN, y, line)
            if url:
                link = url if url.startswith("http") else f"https://{url}"
                prefix = f"{cname}  |  {issuer}  | " if issuer else f"{cname}  | "
                x1 = MARGIN + _text_width(prefix, FONT_REGULAR, cert_size)
                x2 = x1 + _text_width(url, FONT_REGULAR, cert_size)
                pdf.linkURL(link, (x1, y - 2, x2, y + cert_size), relative=0)
            y -= l(LD_BODY) + 2

    pdf.save()
    result = buffer.getvalue()
    buffer.close()
    return result


# ---------------------------------------------------------------------------
# PLAIN TEXT EXPORT
# ---------------------------------------------------------------------------

def resume_json_to_text(data: Dict[str, Any]) -> str:
    def strip_bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    lines = []
    contact = data.get("contact", {})
    lines.append(str(data.get("name", "")).strip())
    lines.append("  |  ".join(filter(None, [
        contact.get("city_state", ""), contact.get("email", ""),
        contact.get("phone", ""), contact.get("linkedin", ""), contact.get("github", ""),
    ])))

    lines.append("\nPROFESSIONAL SUMMARY")
    lines.append(strip_bold(str(data.get("summary", "")).strip()))

    lines.append("\nCORE SKILLS")
    for cat in data.get("skills", []):
        cname = cat.get("category", "")
        items = ", ".join(cat.get("items", []))
        lines.append(f"{cname}: {items}")

    lines.append("\nWORK EXPERIENCE")
    for role in data.get("experience", []):
        header = f'{role.get("title","")} | {role.get("company","")} | {role.get("location","")} | {role.get("start","")} - {role.get("end","")}'
        lines.append(header)
        for b in role.get("bullets", []):
            lines.append(f"- {strip_bold(str(b).strip())}")

    lines.append("\nPROJECTS")
    for proj in data.get("projects", []):
        lines.append(f'{proj.get("name","")}')
        for b in proj.get("bullets", []):
            lines.append(f"- {strip_bold(str(b).strip())}")

    lines.append("\nEDUCATION")
    for edu in data.get("education", []):
        lines.append(f'{edu.get("degree","")} | {edu.get("school","")} | {edu.get("location","")} | {edu.get("start","")} - {edu.get("end","")}')

    certs = data.get("certifications", [])
    if certs:
        lines.append("\nCERTIFICATIONS")
        for c in certs:
            line = f'{c.get("name","")} | {c.get("issuer","")}'
            if c.get("url"):
                line = f"{line} | {c.get('url','')}"
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STREAMLIT PREVIEW
# ---------------------------------------------------------------------------

def render_preview(data: Dict[str, Any]):
    st.markdown(f"## {data.get('name', '')}")
    c = data.get("contact", {})
    st.caption("  |  ".join(filter(None, [
        c.get("city_state",""), c.get("email",""), c.get("phone",""),
        c.get("linkedin",""), c.get("github","")
    ])))

    st.markdown("**PROFESSIONAL SUMMARY**")
    st.markdown(str(data.get("summary", "")))

    st.markdown("**CORE SKILLS**")
    for cat in data.get("skills", []):
        st.markdown(f"**{cat.get('category','')}:** {', '.join(cat.get('items',[]))}")

    st.markdown("**WORK EXPERIENCE**")
    for role in data.get("experience", []):
        st.markdown(
            f"**{role.get('company','')}** &nbsp;&nbsp; *{role.get('title','')} · {role.get('location','')}* &nbsp;&nbsp; "
            f"<span style='color:gray;font-size:0.85em'>{role.get('start','')} – {role.get('end','')}</span>",
            unsafe_allow_html=True
        )
        for b in role.get("bullets", []):
            st.markdown(f"- {b}")

    if data.get("projects"):
        st.markdown("**PROJECTS**")
        for proj in data.get("projects", []):
            st.markdown(f"**{proj.get('name','')}**")
            for b in proj.get("bullets", []):
                st.markdown(f"- {b}")

    st.markdown("**EDUCATION**")
    for edu in data.get("education", []):
        st.markdown(f"**{edu.get('degree','')}** — {edu.get('school','')} · {edu.get('location','')} &nbsp; *{edu.get('start','')} – {edu.get('end','')}*", unsafe_allow_html=True)

    certs = data.get("certifications", [])
    if certs:
        st.markdown("**CERTIFICATIONS**")
        for cert in certs:
            name   = cert.get("name", "")
            issuer = cert.get("issuer", "")
            url    = cert.get("url", "")
            line   = f"**{name}** | {issuer}"
            if url:
                link = url if url.startswith("http") else f"https://{url}"
                line += f" — [{url.replace('https://','').replace('http://','')[:60]}]({link})"
            st.markdown(f"- {line}")


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    st.title("📄 ATS Resume Builder")
    st.caption("Upload knowledge + resume, paste a JD, generate a tailored one-page resume.")

    with st.sidebar:
        st.header("⚙️ Model Settings")
        selected_model = st.selectbox(
            "Choose Model",
            options=MODEL_OPTIONS,
            index=3,
            format_func=lambda x: f'{x["name"]} ({x["id"]})',
        )
        st.caption(f"**Family:** {selected_model['group']}")
        st.caption(selected_model["info"])
        run_match_analysis = st.checkbox("Run JD Match Analysis (slower)", value=False)

    cached_knowledge = load_cached_text(KNOWLEDGE_CACHE_PATH)
    cached_resume    = load_cached_text(RESUME_CACHE_PATH)

    with st.expander("💾 Saved Memory", expanded=False):
        st.write(f"Knowledge file: {modified_label(KNOWLEDGE_CACHE_PATH)}")
        st.write(f"Current resume: {modified_label(RESUME_CACHE_PATH)}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear saved knowledge", use_container_width=True):
                if KNOWLEDGE_CACHE_PATH.exists(): KNOWLEDGE_CACHE_PATH.unlink()
                st.rerun()
        with c2:
            if st.button("Clear saved resume", use_container_width=True):
                if RESUME_CACHE_PATH.exists(): RESUME_CACHE_PATH.unlink()
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        knowledge_file      = st.file_uploader("Upload knowledge/memory file (TXT, PDF, DOCX)", type=["txt","pdf","docx"])
        current_resume_file = st.file_uploader("Upload current resume (TXT, PDF, DOCX)", type=["txt","pdf","docx"])
    with col2:
        job_description  = st.text_area("Paste job description", height=250)
        additional_notes = st.text_area("Optional notes (role focus, keywords to emphasize, gaps to address)", height=120)

    if st.button("💾 Save uploaded files to memory", use_container_width=True):
        did_save = False
        if knowledge_file:
            save_cached_text(KNOWLEDGE_CACHE_PATH, clean_text(read_uploaded_file(knowledge_file)))
            did_save = True
        if current_resume_file:
            save_cached_text(RESUME_CACHE_PATH, clean_text(read_uploaded_file(current_resume_file)))
            did_save = True
        if did_save:
            st.success("Files saved to memory.")
            st.rerun()
        else:
            st.warning("Upload at least one file first.")

    if st.button("🚀 Generate ATS Resume", type="primary", use_container_width=True):
        api_key = get_api_key()
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY in code or environment.")
            return
        if not job_description.strip():
            st.error("Please provide a job description.")
            return

        knowledge_text = None
        resume_text    = None

        if knowledge_file:
            knowledge_text = clean_text(read_uploaded_file(knowledge_file))
            save_cached_text(KNOWLEDGE_CACHE_PATH, knowledge_text)
        elif cached_knowledge:
            knowledge_text = cached_knowledge

        if current_resume_file:
            resume_text = clean_text(read_uploaded_file(current_resume_file))
            save_cached_text(RESUME_CACHE_PATH, resume_text)
        elif cached_resume:
            resume_text = cached_resume

        if not knowledge_text or not resume_text:
            st.error("Missing source files. Upload or save memory files first.")
            return

        try:
            import concurrent.futures

            resume_json    = None
            match_analysis = {}

            if run_match_analysis:
                with st.spinner("Generating resume + running JD match analysis..."):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_resume = executor.submit(
                            generate_resume_json,
                            api_key, selected_model["id"],
                            knowledge_text, resume_text,
                            job_description.strip(),
                            additional_notes.strip() or None,
                        )
                        future_match = executor.submit(
                            generate_match_analysis,
                            api_key, selected_model["id"],
                            knowledge_text, resume_text,
                            job_description.strip(),
                        )
                        resume_json    = future_resume.result()
                        match_analysis = future_match.result()
                st.success("✅ Resume generated and JD match analysis complete.")
            else:
                with st.spinner("Generating resume..."):
                    resume_json = generate_resume_json(
                        api_key, selected_model["id"],
                        knowledge_text, resume_text,
                        job_description.strip(),
                        additional_notes.strip() or None,
                    )
                st.success("✅ Resume generated.")

            st.session_state["resume_json"] = resume_json
            st.session_state["match_analysis"] = match_analysis
            st.session_state["last_job_description"] = job_description.strip()
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

    if "resume_json" in st.session_state:
        data     = st.session_state["resume_json"]
        analysis = st.session_state.get("match_analysis", {})

        st.divider()

        tab_match, tab_preview, tab_json = st.tabs([
            "🎯 JD Match Analysis",
            "👁️ Resume Preview",
            "📝 Edit JSON",
        ])

        with tab_match:
            if analysis:
                render_match_panel(analysis)
            else:
                st.info("Match analysis not available. Regenerate to see results.", icon="ℹ️")

        with tab_preview:
            render_preview(data)
            st.divider()
            role_name, company_name = _infer_role_company(
                st.session_state.get("last_job_description", ""),
                str(data.get("summary", "")),
            )
            pdf_filename = f"Gautham_{_slug_component(role_name, 'Role')}_{_slug_component(company_name, 'Company')}.pdf"
            txt_filename = f"Gautham_{_slug_component(role_name, 'Role')}_{_slug_component(company_name, 'Company')}.txt"
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "📄 Download as TXT",
                    data=resume_json_to_text(data),
                    file_name=txt_filename,
                    mime="text/plain",
                    use_container_width=True,
                )
            with d2:
                try:
                    pdf_bytes = auto_fit_pdf(data)
                    st.download_button(
                        "📥 Download as PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        with tab_json:
            st.caption("Edit the JSON directly to tweak any field, then apply.")
            edited_json_str = st.text_area(
                "Resume JSON",
                value=json.dumps(data, indent=2),
                height=500,
                label_visibility="collapsed",
            )
            if st.button("🔄 Apply JSON edits", use_container_width=True):
                try:
                    updated = json.loads(edited_json_str)
                    st.session_state["resume_json"] = sanitize_resume_json(updated)
                    st.success("JSON applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")


if __name__ == "__main__":
    main()
