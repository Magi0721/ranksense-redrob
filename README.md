# RankSense — Redrob Intelligent Candidate Discovery & Ranking Challenge

A hybrid candidate-ranking system built for the Redrob Data & AI Hackathon. Given a job description and a pool of 100,000 candidate profiles, it produces a ranked top-100 shortlist with a per-candidate reasoning string — entirely CPU-only, offline, and reproducible in under a minute.

## Why this approach

The JD for this challenge (Senior AI Engineer, Founding Team) is explicit that keyword matching is the wrong answer. It calls out several traps directly: candidates who list AI buzzwords as skills but work in unrelated functions, candidates with research-only backgrounds and no production deployment, candidates whose "AI experience" is a few months of LangChain wrapper code, senior engineers who've stopped writing code, pure-consulting career histories, and behaviorally unreachable candidates (inactive, unresponsive, not actually job-seeking).

So instead of one similarity score, RankSense layers four independent signal types and lets disqualifiers act as hard multipliers rather than features that can be "outvoted" by a strong skill list:

1. **Structured role/seniority/location/education fit** — read directly from `profile` and `career_history`.
2. **Trust-adjusted skill match** — a candidate's claimed skill proficiency is only trusted if it's backed by their own Redrob `skill_assessment_scores`, or if `duration_months` on that skill is plausible. A skill listed as "expert" with `duration_months: 0` is treated as a red flag, not a strength — this is exactly how we catch keyword-stuffers without needing an LLM to "read between the lines."
3. **Semantic fit** — TF-IDF + cosine similarity between the JD text and each candidate's free-text fields (summary + career history descriptions). This captures candidates who built the right thing without ever using the JD's exact vocabulary, using nothing more than local linear algebra (no pretrained model download, no GPU, no network call).
4. **Hard disqualifier multipliers** — consulting-only career history, pure-research-without-deployment, shallow recent-only LangChain experience, CV/speech/robotics-only background, title-chasing tenure patterns, and "moved to pure leadership, no recent hands-on code" all multiply the score down, mirroring the JD's explicit "things we do NOT want" section.
5. **Behavioral availability multiplier** — recruiter response rate, days since last active, `open_to_work_flag`, interview completion rate, notice period, and verified contact info scale the final score, so a perfect-on-paper but unreachable candidate is correctly down-weighted.
6. **Honeypot / integrity filter** — internal-consistency checks (e.g. "expert" proficiency claimed with zero months of use, overlapping full-time current jobs, education years that contradict each other, future-dated start dates) hard-exclude synthetic impossible profiles before scoring even begins.

Reasoning strings are built directly from each candidate's own real fields (no LLM call, so no hallucination risk) and rotate through several phrasings so the output doesn't read as templated.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── src/
│   ├── scoring.py      # feature engineering + disqualifier/behavioral logic
│   └── rank.py         # main pipeline: load -> filter -> score -> rank -> write CSV/XLSX
├── output/
│   ├── submission.csv  # final top-100 ranking (spec format)
│   └── submission.xlsx # same ranking, portal-required XLSX format
└── validate_submission.py   # organizer-provided validator (unmodified)
```

## How to reproduce

```bash
pip install -r requirements.txt
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv --xlsx-out ./submission.xlsx
python validate_submission.py ./submission.csv
```

`candidates.jsonl` is the released candidate pool (not included in this repo due to size — drop it in the repo root or pass its path with `--candidates`).

## Compute profile

- Pure CPU, no GPU usage anywhere in the pipeline.
- No network calls during ranking — TF-IDF is fit fresh on the candidate corpus each run, no pretrained weights downloaded.
- Full 100,000-candidate run completes in well under 1 minute on a standard machine (measured: ~35–55s wall clock, far inside the 5-minute / 16GB budget).
- No pre-computation step is required — the single `rank.py` command above is the entire pipeline.

## Honeypot handling

~48 of the 100,000 candidates fail the internal-consistency checks in `honeypot_flags()` (e.g. "expert" proficiency with zero months used on 2+ skills, current-job-date inconsistencies, contradictory education years) and are excluded before scoring. Verified: 0 honeypots appear in the final top-100 output.

## AI tool usage

Built with Claude as a development/engineering assistant — architecture discussion, code authoring, and debugging. No candidate data was sent to any external LLM API; the ranking pipeline itself makes zero hosted-API calls, consistent with the offline compute constraint. See `submission_metadata.yaml` for the full declaration.
