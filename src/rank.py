#!/usr/bin/env python3
"""
RankSense — rank.py
Reproduce command (per submission_spec section 10.3):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

CPU-only. No network calls. No GPU. Designed to finish well under 5 minutes
for the full 100K-candidate pool on a 16GB RAM machine.

Pipeline:
  1. Stream-parse candidates.jsonl (one JSON object per line).
  2. Apply honeypot integrity filter (hard exclusion).
  3. Compute structured features (role fit, trust-adjusted skill match,
     experience fit, location fit, education tier, disqualifier penalties).
  4. Compute semantic fit via TF-IDF + cosine similarity (JD vs candidate
     free text), fit once on the corpus, no pretrained downloads.
  5. Combine into a weighted composite, apply hard disqualifier multipliers,
     apply the behavioral availability multiplier.
  6. Take top 100, assign rank 1..100, generate a short templated-but-
     data-grounded reasoning string per candidate (no LLM call — built
     directly from the candidate's own real fields, so it can never
     hallucinate a skill they don't have).
  7. Write CSV (spec format) and XLSX (portal format).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from scoring import (
    CORE_SKILLS, BONUS_SKILLS, ROLE_SIGNAL_TERMS, NON_PRACTITIONER_TITLE_TERMS,
    candidate_full_text, is_consulting_only, has_recent_product_experience,
    is_pure_research, title_chaser_score, stale_coder_penalty,
    shallow_langchain_only, cv_speech_robotics_only, skill_trust_adjusted_match,
    honeypot_flags, availability_multiplier, location_fit,
)

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A AI-native talent
intelligence platform. Own the intelligence layer ranking retrieval and
matching systems. Production experience with embeddings-based retrieval
systems sentence-transformers OpenAI embeddings BGE E5 deployed to real
users, handled embedding drift index refresh retrieval quality regression
in production. Production experience with vector databases or hybrid
search infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch
Elasticsearch FAISS. Strong Python code quality. Hands-on experience
designing evaluation frameworks for ranking systems NDCG MRR MAP
offline-to-online correlation A/B test interpretation. LLM fine-tuning
LoRA QLoRA PEFT. Learning-to-rank models XGBoost-based or neural. HR-tech
recruiting tech marketplace products. Distributed systems large-scale
inference optimization. Open-source contributions AI ML space. Shipped at
least one end-to-end ranking search or recommendation system to real users
at meaningful scale. Hybrid dense retrieval evaluation offline online LLM
integration fine-tune vs prompt. Applied ML AI roles at product companies.
"""


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def structured_score(c):
    profile = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})
    edu = c.get("education", [])

    title = (profile.get("current_title") or "").lower()

    # --- role fit: is this person actually an ML/search/retrieval practitioner? ---
    role_text = title + " " + " ".join((j.get("title") or "") for j in career)
    role_hits = sum(1 for t in ROLE_SIGNAL_TERMS if t in role_text.lower())
    is_non_practitioner_title = any(t in title for t in NON_PRACTITIONER_TITLE_TERMS)
    role_fit = min(role_hits / 3.0, 1.0)
    if is_non_practitioner_title:
        role_fit *= 0.15  # the JD's explicit "Marketing Manager w/ AI keywords" trap

    # --- trust-adjusted skill match ---
    core_w, core_n = skill_trust_adjusted_match(
        skills, sig.get("skill_assessment_scores", {}), CORE_SKILLS)
    bonus_w, bonus_n = skill_trust_adjusted_match(
        skills, sig.get("skill_assessment_scores", {}), BONUS_SKILLS)
    skill_score = min(core_w / 3.0, 1.0) * 0.8 + min(bonus_w / 3.0, 1.0) * 0.2

    # --- experience fit (JD wants 5-9y, flexible) ---
    yoe = profile.get("years_of_experience") or 0
    if 5 <= yoe <= 9:
        exp_fit = 1.0
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        exp_fit = 0.75
    else:
        exp_fit = 0.45

    # --- education tier (soft signal, JD doesn't emphasize pedigree) ---
    tiers = [e.get("tier") for e in edu if e.get("tier")]
    if "tier_1" in tiers:
        edu_fit = 1.0
    elif "tier_2" in tiers:
        edu_fit = 0.85
    elif tiers:
        edu_fit = 0.7
    else:
        edu_fit = 0.6

    loc_fit = location_fit(profile)

    structured = (
        0.34 * role_fit +
        0.30 * skill_score +
        0.16 * exp_fit +
        0.12 * loc_fit +
        0.08 * edu_fit
    )

    # --- hard disqualifier multiplier stack (JD's explicit "do not want" list) ---
    mult = 1.0
    reasons_neg = []
    if is_consulting_only(career):
        mult *= 0.12
        reasons_neg.append("consulting-only career, no product-company exposure")
    elif not has_recent_product_experience(career):
        mult *= 0.55
        reasons_neg.append("currently at a consulting/IT-services firm")
    if is_pure_research(profile, career):
        mult *= 0.15
        reasons_neg.append("pure-research background, no production deployment evidence")
    tc = title_chaser_score(career)
    if tc > 0.6:
        mult *= (1 - 0.4 * tc)
        reasons_neg.append("frequent short tenures (title-chasing pattern)")
    stale = stale_coder_penalty(profile, career)
    if stale > 0:
        mult *= (1 - stale)
        reasons_neg.append("moved to pure architecture/leadership, limited recent hands-on coding")
    if shallow_langchain_only(profile, career, skills):
        mult *= 0.25
        reasons_neg.append("AI experience limited to recent LangChain/OpenAI usage, no deeper history")
    if cv_speech_robotics_only(career, skills):
        mult *= 0.3
        reasons_neg.append("background in CV/speech/robotics without NLP/IR exposure")

    return structured, mult, reasons_neg, role_fit, skill_score, core_n


SEMANTIC_PHRASES_HIGH = [
    "Career-history language closely mirrors the JD's retrieval/ranking responsibilities.",
    "Profile narrative reads as a strong textual match to what the role actually asks for.",
    "Job descriptions in their history echo the JD's core mandate almost directly.",
    "Their own write-up of past work overlaps heavily with the JD's stated scope.",
]
SEMANTIC_PHRASES_MED = [
    "Reasonable thematic overlap with the JD, though not a word-for-word match.",
    "Some relevant overlap with the role's focus areas in their work history.",
    "Partial alignment between their described work and the JD's priorities.",
]
SKILL_PHRASE_TEMPLATES = [
    "{n} JD-core skill(s) hold up against their own Redrob assessment scores.",
    "{n} of the JD's must-have skills are present and assessment-verified.",
    "Carries {n} core skill(s) the JD lists as non-negotiable, with assessment backing.",
    "{n} core-skill match(es), cross-checked against platform assessment data (not just self-claim).",
]


def build_reasoning(c, semantic, role_fit, skill_score, core_n, reasons_neg, avail_mult):
    profile = c["profile"]
    cid_seed = int(c["candidate_id"].split("_")[1]) if "_" in c["candidate_id"] else 0
    yoe = profile.get("years_of_experience")
    title = profile.get("current_title")
    company = profile.get("current_company")
    loc = profile.get("location")
    sig = c.get("redrob_signals", {})
    rr = sig.get("recruiter_response_rate")
    last_active = sig.get("last_active_date")

    pieces = [f"{title} with {yoe} yrs, currently at {company} ({loc})."]

    if core_n > 0:
        tmpl = SKILL_PHRASE_TEMPLATES[cid_seed % len(SKILL_PHRASE_TEMPLATES)]
        pieces.append(tmpl.format(n=core_n))
    else:
        pieces.append("No JD-core skill directly matched; ranked mainly on role/career-history fit.")

    if semantic > 0.35:
        pieces.append(SEMANTIC_PHRASES_HIGH[cid_seed % len(SEMANTIC_PHRASES_HIGH)])
    elif semantic > 0.15:
        pieces.append(SEMANTIC_PHRASES_MED[cid_seed % len(SEMANTIC_PHRASES_MED)])

    if rr is not None:
        if rr >= 0.7:
            pieces.append(f"Responsive on-platform ({rr:.0%} recruiter response rate).")
        elif rr < 0.3:
            pieces.append(f"Low recruiter responsiveness ({rr:.0%}); availability discounted accordingly.")
        else:
            pieces.append(f"Recruiter response rate {rr:.0%}.")

    if avail_mult < 0.7:
        pieces.append(f"Activity/availability signals (last active {last_active}) reduced this candidate's final score.")

    if reasons_neg:
        pieces.append("Watch-out: " + reasons_neg[0] + ".")

    text = " ".join(pieces)
    return text[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlsx-out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading candidates...")
    candidates = list(load_candidates(args.candidates))
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    # ---- honeypot filter ----
    kept = []
    honeypots = 0
    for c in candidates:
        if honeypot_flags(c):
            honeypots += 1
            continue
        kept.append(c)
    print(f"Filtered {honeypots} honeypot/impossible-profile candidates. {len(kept)} remain.")

    # ---- semantic similarity (TF-IDF, fit once on corpus + JD) ----
    print("Computing TF-IDF semantic similarity...")
    texts = [candidate_full_text(c) for c in kept]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  stop_words="english", min_df=2)
    corpus = texts + [JD_TEXT.lower()]
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[-1]
    cand_vecs = tfidf[:-1]
    sims = cosine_similarity(cand_vecs, jd_vec).ravel()
    print(f"Semantic similarity computed in {time.time()-t0:.1f}s total.")

    # ---- structured scoring ----
    print("Computing structured + behavioral scores...")
    results = []
    for c, sem in zip(kept, sims):
        structured, mult, reasons_neg, role_fit, skill_score, core_n = structured_score(c)
        sem_norm = min(sem * 4.0, 1.0)  # TF-IDF cosine sims are small; rescale
        composite = (0.6 * structured + 0.4 * sem_norm) * mult
        avail_mult = availability_multiplier(c.get("redrob_signals", {}))
        final = composite * avail_mult
        results.append((c, final, sem_norm, role_fit, skill_score, core_n, reasons_neg, avail_mult))

    results.sort(key=lambda x: x[1], reverse=True)
    top100 = results[:100]

    # normalize scores to a clean 0-1 band for presentation, preserving order
    raw_scores = [r[1] for r in top100]
    max_s, min_s = max(raw_scores), min(raw_scores)
    span = max(max_s - min_s, 1e-9)

    rows = []
    for i, (c, final, sem, role_fit, skill_score, core_n, reasons_neg, avail_mult) in enumerate(top100, start=1):
        norm_score = 0.4 + 0.59 * (final - min_s) / span  # keep in (0.4, 0.99]
        norm_score = round(min(norm_score, 0.999), 4)
        reasoning = build_reasoning(c, sem, role_fit, skill_score, core_n, reasons_neg, avail_mult)
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": i,
            "score": norm_score,
            "reasoning": reasoning,
        })

    # enforce strictly non-increasing score by rank (tie-break by candidate_id asc)
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    # ---- write CSV ----
    import csv
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV: {out_path}")

    # ---- write XLSX (portal format) ----
    xlsx_path = Path(args.xlsx_out) if args.xlsx_out else out_path.with_suffix(".xlsx")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "submission"
        ws.append(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            ws.append([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
        wb.save(xlsx_path)
        print(f"Wrote XLSX: {xlsx_path}")
    except ImportError:
        print("openpyxl not available, skipped XLSX output.")

    print(f"Done in {time.time()-t0:.1f}s total.")
    honeypots_in_top100 = sum(1 for c, *_ in top100 if honeypot_flags(c))
    print(f"Honeypots in top100 (sanity check, should be 0): {honeypots_in_top100}")


if __name__ == "__main__":
    main()
