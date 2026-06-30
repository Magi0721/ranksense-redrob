"""
RankSense — Redrob Hackathon Candidate Ranking Engine
=======================================================
Scoring logic. Pure Python + numpy/scikit-learn (CPU only, no network).

Design philosophy (see deck for full rationale):
  The JD explicitly says it does NOT want pure keyword matching, and that
  it actively penalizes:
    - keyword-stuffed profiles with the wrong job function
    - research-only / no-production candidates
    - shallow "LangChain + OpenAI in <12mo" candidates with no pre-LLM
      production ML experience
    - senior engineers who stopped writing code 18+ months ago
    - pure consulting-only (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini)
      career history with zero product-company exposure
    - title-chasers (job hop every <1.5 yrs chasing titles)
    - CV/speech/robotics-only without NLP/IR exposure
    - candidates who are not actually reachable/available right now
      (behavioral signals)

  So the engine is a hybrid of four layers:
    1. STRUCTURED FIT   — role/title relevance, seniority, experience years,
                           location/relocation, education tier
    2. TRUST-ADJUSTED SKILL MATCH — JD-critical skill coverage, but
                           discounted when a candidate's *claimed* skill
                           proficiency disagrees with their own Redrob
                           platform skill_assessment_scores (this is how we
                           catch keyword-stuffers: they list "expert" but
                           score 30/100 on the actual assessment, or list
                           a skill with duration_months = 0)
    3. SEMANTIC FIT     — TF-IDF + cosine similarity between the JD text and
                           each candidate's free-text (summary + career
                           history descriptions). This captures "they never
                           said the word RAG but clearly built a
                           recommendation system" style fits, lexically.
                           Pure local linear algebra, no GPU, no network,
                           no pretrained weight download — safe for the
                           5-min / 16GB / CPU-only / offline constraint.
    4. HARD DISQUALIFIERS — explicit JD red lines applied as multiplicative
                           penalties, not just feature weights, so a single
                           bad signal can't be "out-voted" by good ones.
    5. AVAILABILITY MULTIPLIER — behavioral signals (recruiter response
                           rate, last_active recency, open_to_work,
                           interview completion, notice period, verified
                           contact info) scale the final score down for
                           people who look great on paper but are not
                           actually reachable/hireable right now.
    6. HONEYPOT / INTEGRITY FILTER — internal-consistency checks
                           (impossible skill durations, "expert" with 0
                           months used, experience/education-year
                           contradictions) that hard-zero a candidate's
                           score so they cannot surface in the top 100.
"""

import json
import re
import math
from datetime import date

# ----------------------------------------------------------------------
# JD-derived configuration
# This is our structured interpretation of job_description.docx, written
# once by a human (us) reading the JD closely — NOT auto-extracted by an
# LLM at scoring time. This is what lets ranking stay fast & reproducible.
# ----------------------------------------------------------------------

TODAY = date(2026, 6, 30)

# Skills the JD calls "absolutely need" — the core hard requirement.
CORE_SKILLS = {
    # embeddings-based retrieval
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embeddings", "dense retrieval",
    # vector db / hybrid search infra
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
    # eval frameworks
    "ndcg", "mrr", "map", "a/b testing", "offline evaluation",
    "learning to rank", "ranking evaluation",
}

# Skills JD says are "nice to have" — smaller weight, never disqualifying.
BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "xgboost", "learning-to-rank", "neural ranking",
    "hr-tech", "recruiting tech", "marketplace",
    "distributed systems", "large-scale inference",
}

# Role/function keywords that indicate the candidate actually works in
# applied ML / search / ranking / retrieval / recsys (title + description),
# as opposed to merely listing AI buzzwords as "skills" while doing an
# unrelated job (the JD's explicit "Marketing Manager with RAG skill" trap).
ROLE_SIGNAL_TERMS = [
    "machine learning", "ml engineer", "applied scientist", "ai engineer",
    "search", "ranking", "retrieval", "recommendation", "recsys",
    "nlp", "information retrieval", "data scientist", "research scientist",
    "deep learning",
]

# Titles that should NOT be treated as AI/ML practitioners even if their
# skills list is keyword-stuffed (this is the keyword-stuffer trap filter).
NON_PRACTITIONER_TITLE_TERMS = [
    "marketing", "sales", "hr ", "human resources", "recruiter",
    "account manager", "business development", "content writer",
    "operations", "customer success", "finance", "legal",
    "product marketing", "social media",
]

PURE_RESEARCH_TERMS = ["research scientist", "research fellow", "postdoc",
                        "phd researcher", "academic researcher"]
RESEARCH_INDUSTRY_TERMS = ["academia", "research institute", "university"]

CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro",
                     "accenture", "cognizant", "capgemini"]

CV_SPEECH_ROBOTICS_ONLY_TERMS = ["computer vision", "robotics", "speech recognition"]
NLP_IR_TERMS = ["nlp", "natural language processing", "information retrieval",
                "search", "ranking", "retrieval", "recommendation"]

TIER1_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
                    "bengaluru", "gurgaon", "gurugram", "ncr"]

LANGCHAIN_SHALLOW_TERMS = ["langchain"]


def months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def candidate_full_text(c):
    """All free text used for semantic (TF-IDF) matching."""
    parts = [c["profile"].get("headline", ""), c["profile"].get("summary", "")]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    parts.append(" ".join(s["name"] for s in c.get("skills", [])))
    return " ".join(parts).lower()


def is_consulting_only(career_history):
    if not career_history:
        return False
    for job in career_history:
        company = (job.get("company") or "").lower()
        industry = (job.get("industry") or "").lower()
        is_consult = any(f in company for f in CONSULTING_FIRMS) or "it services" in industry
        if not is_consult:
            return False
    return True


def has_recent_product_experience(career_history):
    """True if candidate currently OR most-recently is at a non-consulting employer."""
    if not career_history:
        return False
    current = [j for j in career_history if j.get("is_current")]
    ref = current[0] if current else career_history[0]
    company = (ref.get("company") or "").lower()
    industry = (ref.get("industry") or "").lower()
    return not (any(f in company for f in CONSULTING_FIRMS) or "it services" in industry)


def is_pure_research(profile, career_history):
    title = (profile.get("current_title") or "").lower()
    industry = (profile.get("current_industry") or "").lower()
    if any(t in title for t in PURE_RESEARCH_TERMS):
        # check if ANY job in history shows production/deployment language
        prod_terms = ["deployed", "production", "shipped", "users", "scale", "launched"]
        text = " ".join((j.get("description") or "").lower() for j in career_history)
        if not any(t in text for t in prod_terms):
            return True
    if any(t in industry for t in RESEARCH_INDUSTRY_TERMS):
        return True
    return False


def title_chaser_score(career_history):
    """Penalize career histories with many very short (<18mo) stints, each a step up."""
    if len(career_history) < 3:
        return 0.0
    short_stints = sum(1 for j in career_history if (j.get("duration_months") or 999) < 18)
    ratio = short_stints / len(career_history)
    return ratio  # 0 = no title-chasing pattern, 1 = every job <18mo


def stale_coder_penalty(profile, career_history):
    """JD: senior engineers who haven't written production code in 18+ months
    because they moved to pure architecture/tech-lead roles, penalize."""
    title = (profile.get("current_title") or "").lower()
    leadership_only = any(t in title for t in
                           ["architect", "tech lead", "engineering manager", "head of",
                            "director", "vp ", "vice president"]) and "engineer" not in title
    if not leadership_only:
        return 0.0
    current = [j for j in career_history if j.get("is_current")]
    if current and (current[0].get("duration_months") or 0) >= 18:
        return 0.5  # moderate penalty, JD says "probably not" not "never"
    return 0.0


def shallow_langchain_only(profile, career_history, skills):
    """JD: AI experience that's only <12mo of LangChain+OpenAI calling,
    without older production ML experience -> disqualify-ish."""
    skill_names = [s["name"].lower() for s in skills]
    has_langchain = any(t in skill_names for t in LANGCHAIN_SHALLOW_TERMS)
    if not has_langchain:
        return False
    years_exp = profile.get("years_of_experience") or 0
    # look for any pre-LLM-era (more than 2 years of total relevant ML/IR tenure)
    ml_months = 0
    for j in career_history:
        title_desc = ((j.get("title") or "") + " " + (j.get("description") or "")).lower()
        if any(t in title_desc for t in ROLE_SIGNAL_TERMS):
            ml_months += j.get("duration_months") or 0
    has_deep_history = ml_months >= 24 or years_exp >= 5
    # if their langchain skill is recent (<12mo) AND they have no deep history -> shallow
    lc_skill = next((s for s in skills if s["name"].lower() in LANGCHAIN_SHALLOW_TERMS), None)
    lc_recent = lc_skill and (lc_skill.get("duration_months") or 0) < 12
    return bool(lc_recent and not has_deep_history)


def cv_speech_robotics_only(career_history, skills):
    text = " ".join((j.get("description") or "") + " " + (j.get("title") or "")
                     for j in career_history).lower()
    skill_names = " ".join(s["name"].lower() for s in skills)
    full = text + " " + skill_names
    has_cvsr = any(t in full for t in CV_SPEECH_ROBOTICS_ONLY_TERMS)
    has_nlp_ir = any(t in full for t in NLP_IR_TERMS)
    return has_cvsr and not has_nlp_ir


def skill_trust_adjusted_match(skills, skill_assessment_scores, target_set):
    """For each JD-relevant skill the candidate claims, trust it fully only if
    a Redrob assessment score backs it up (or no assessment exists, neutral).
    If they claim 'expert'/'advanced' but the assessment score is low, or the
    skill has duration_months == 0, heavily discount it — this is exactly how
    we catch keyword-stuffers."""
    prof_weight = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}
    matched_weight = 0.0
    matched_count = 0
    for s in skills:
        name = s["name"].lower()
        if name not in target_set:
            continue
        base = prof_weight.get(s.get("proficiency", "intermediate"), 0.6)
        dur = s.get("duration_months", 0) or 0
        trust = 1.0
        if dur == 0 and s.get("proficiency") in ("advanced", "expert"):
            trust = 0.15  # major red flag: claims mastery, zero time spent
        elif dur < 6 and s.get("proficiency") == "expert":
            trust = 0.4
        assess = skill_assessment_scores.get(s["name"])
        if assess is not None:
            assess_ratio = assess / 100.0
            claim_ratio = prof_weight.get(s.get("proficiency", "intermediate"), 0.6)
            if claim_ratio - assess_ratio > 0.35:
                trust = min(trust, 0.3)  # claimed much higher than assessed
        matched_weight += base * trust
        matched_count += 1
    return matched_weight, matched_count


def honeypot_flags(c):
    """Internal-consistency checks. Returns True if candidate looks like
    a synthetic 'impossible profile' honeypot."""
    profile = c["profile"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    edu = c.get("education", [])

    # expert proficiency + 0 months used, on 2+ skills -> near-certain honeypot
    zero_month_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if zero_month_experts >= 2:
        return True

    # years_of_experience wildly inconsistent with sum of career_history duration
    total_months = sum(j.get("duration_months") or 0 for j in career)
    yoe_months = (profile.get("years_of_experience") or 0) * 12
    if total_months > 0 and yoe_months > 0:
        if total_months > yoe_months * 2.2 or yoe_months > total_months * 2.5 + 24:
            return True

    # education end_year before start_year, or end_year in the future beyond plausibility
    for e in edu:
        if e.get("end_year") and e.get("start_year") and e["end_year"] < e["start_year"]:
            return True

    # overlapping full-time jobs that both claim long durations and both "is_current"
    current_jobs = [j for j in career if j.get("is_current")]
    if len(current_jobs) > 1:
        return True

    # career history start dates in the future
    for j in career:
        sd = parse_date(j.get("start_date"))
        if sd and sd > TODAY:
            return True

    return False


def availability_multiplier(sig):
    """Behavioral-signal based multiplier in [0.25, 1.15]."""
    score = 1.0

    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (TODAY - last_active).days
        if days_inactive > 180:
            score *= 0.5
        elif days_inactive > 90:
            score *= 0.75
        elif days_inactive > 30:
            score *= 0.92

    if not sig.get("open_to_work_flag", False):
        score *= 0.7

    rr = sig.get("recruiter_response_rate", 0.5)
    score *= (0.6 + 0.5 * rr)  # 0.6 .. 1.1

    icr = sig.get("interview_completion_rate", 0.7)
    score *= (0.8 + 0.3 * icr)  # 0.8 .. 1.1

    if sig.get("verified_email") and sig.get("verified_phone"):
        score *= 1.03

    notice = sig.get("notice_period_days", 30)
    if notice <= 30:
        score *= 1.05
    elif notice > 60:
        score *= 0.9

    return max(0.25, min(score, 1.2))


def location_fit(profile):
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "")
    if country != "India":
        # JD: outside India = case by case, no visa sponsorship -> penalize
        return 0.35
    if any(t in loc for t in ["pune", "noida"]):
        return 1.0
    if any(t in loc for t in TIER1_LOCATIONS):
        return 0.85
    return 0.6  # India, but not a named tier-1 city
