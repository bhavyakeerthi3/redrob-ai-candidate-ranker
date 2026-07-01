# Redrob Intelligent Candidate Discovery Ranker

This repository contains a reproducible, CPU-only candidate ranking system for
Track 01: The Data & AI Challenge.

The target role is a Senior AI Engineer on Redrob AI's founding team. The ranker
is built to avoid simple keyword matching: it scores career evidence, product
shipping context, trusted skills, behavioral availability, and trap indicators.

## Why this approach

The job description asks for someone who has shipped retrieval, ranking, or
recommendation systems in production, can work hands-on in Python, understands
evaluation, and is actually reachable for recruiting. The candidate data also
contains keyword stuffers and honeypots. Because the scoring step must finish
within 5 minutes on CPU with no network, this solution uses a deterministic
feature ranker instead of per-candidate LLM calls.

## Scoring architecture

The final score combines:

- Title fit: senior/applied AI, ML, NLP, search, and recommendation titles are
  weighted above generic engineering titles; non-target business titles are
  strongly down-weighted.
- Career semantic evidence: career summaries and role descriptions are scanned
  for production retrieval/ranking signals such as embeddings, vector search,
  semantic search, recommendation systems, evaluation frameworks, NDCG/MRR, A/B
  tests, deployed systems, and real users.
- Trusted skill evidence: AI/retrieval skills are weighted by proficiency,
  duration, endorsements, and Redrob assessment scores where available.
- Experience band: the JD's preferred 5-9 year band is favored while still
  allowing exceptional adjacent candidates outside the band.
- Product context: product-company and AI/SaaS/fintech/e-commerce exposure is
  favored over pure services-only careers.
- Behavioral availability: recency, open-to-work, recruiter response rate,
  response speed, notice period, interview completion, GitHub activity, profile
  completeness, verification, relocation, and preferred locations.
- Trap penalties: keyword/title mismatch, thin expert skills, CV/speech-only
  profiles, inconsistent career dates, experience mismatch, inactivity, low
  recruiter response, and framework-only junior AI signals.

Every output row includes fact-grounded reasoning derived from the same features.

## Setup

Use Python 3.11 or newer. The ranker has no runtime dependencies beyond the
standard library.

```bash
python --version
```

Place the released challenge file at `data/candidates.jsonl`, or pass its path
with `--candidates`.

## Reproduce the submission

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/team_redrob_ranker.csv --diagnostics ./outputs/top100_diagnostics.jsonl
python data/validate_submission.py ./outputs/team_redrob_ranker.csv
```

The command streams the JSONL file, keeps only the top 100 candidates in memory,
and writes the required CSV columns:

```text
candidate_id,rank,score,reasoning
```

## Small-sample smoke test

```bash
python rank.py --candidates ./data/sample_candidates.json --out ./outputs/sample_ranked.csv --top-k 10
```

## Docker sandbox

```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" redrob-ranker python rank.py --candidates ./data/candidates.jsonl --out ./outputs/team_redrob_ranker.csv
```

## Repository contents

- `rank.py`: CLI entry point.
- `src/redrob_ranker/ranker.py`: scoring, ranking, reasoning, and output logic.
- `data/sample_candidates.json`: small sample from the challenge bundle.
- `data/validate_submission.py`: provided validator.
- `submission_metadata.yaml`: portal metadata template filled with the approach
  summary. Replace contact details and GitHub URL before final upload.

## Notes for reviewers

No hosted LLMs, GPUs, or network calls are used during ranking. The code is
deterministic: ties are broken by candidate ID and scores are emitted in
non-increasing rank order.

