# Redrob Intelligent Candidate Discovery and Ranking Engine

This repository contains the complete implementation of a reproducible, CPU-only candidate ranking system designed for Track 01: The Data and AI Challenge. 

The primary target of this engine is to identify and rank the top 100 candidates for a founding Senior AI Engineer role at Redrob AI, streaming a pool of 100,000 candidates under strict compute constraints. The system has zero runtime package dependencies, runs entirely on the CPU with no network access, and executes in approximately 37 seconds on a standard local machine.

## Architecture and System Philosophy

Real-world recruiting systems operating on candidate pools of over 100,000 profiles face a severe latency-cost trade-off. Running per-candidate Large Language Model (LLM) API calls is slow and expensive. To satisfy the 5-minute wall-clock constraint on a single 16 GB CPU-only sandbox, this solution employs a deterministic feature streaming ranker. 

The engine streams candidates one-by-one from the JSONL dataset, extracts profile attributes, parses work history and skills, calculates an explainable composite score, and manages a top-K min-heap in memory to output the final ranked shortlist.

## Scoring Formulation and Weights

The final candidate score is computed using a weighted linear combination of raw suitability factors, modified by product company multipliers, and reduced by standard or terminal penalties:

Score = clamp(RawScore - Penalty, 0.0, 1.0)

Where RawScore is defined as:

RawScore = 0.24 * TitleScore + 0.25 * CareerScore + 0.18 * SkillScore + 0.10 * ExperienceScore + 0.10 * ProductScore + 0.13 * AvailabilityScore

### 1. Title Fit (24% Weight)
Current titles are parsed and mapped to target bands:
* Senior/Lead AI Engineer: 1.0
* Senior/Staff Machine Learning Engineer: 0.95
* Senior NLP / Senior Software Engineer (ML): 0.90
* Search / Recommendation Systems Engineer: 0.90
* Applied ML Engineer / AI Engineer: 0.84
* Data Scientist: 0.58
* AI Research Scientist: 0.45 (slightly down-weighted due to the job description's preference for shipping over academic research)
* Software/Backend/Data Engineer: 0.38
* Non-target business titles (e.g. Marketing, HR, Finance): -0.35

### 2. Career Semantic Evidence (25% Weight)
The candidate's headline, summary, and past role descriptions are concatenated and scanned for production-grade retrieval, ranking, and search patterns. Weight is distributed among keywords representing:
* Information retrieval, semantic search, vector search, embeddings, sentence transformers.
* Search ranking, recommendation systems, hybrid search.
* Production systems, deployed systems, real user scaling, A/B testing.
* Evaluation metrics like NDCG, MRR, MAP, and offline evaluation frameworks.

### 3. Trusted Skill Evidence (18% Weight)
Skills are evaluated by combining stated proficiency, duration of use (months), skill endorsements, and platform-verified Redrob assessment scores:
* Trust Factor = 0.48 * ProficiencyMultiplier + 0.28 * DurationMultiplier + 0.24 * EndorsementLogScale
* If a platform assessment score exists for the skill, the trust factor is updated: 0.7 * TrustFactor + 0.3 * (AssessmentScore / 100.0)
* Stated core skills (embeddings, semantic search, FAISS, Milvus, Pinecone, MLOps, NLP) are scaled by their trust factors to prevent keyword stuffing.

### 4. Experience Band (10% Weight)
The job description seeks candidates in the 5-9 years of experience sweet spot, while remaining open to adjacent profiles. The score combines:
* Overall Experience: Evaluates total years of experience, peaking at 1.0 for 5-9 years, 0.75-0.78 for 4-5 and 9-11 years, and dropping outside those ranges.
* Machine Learning Experience: Calculates the duration (in years) spent explicitly in AI/ML/NLP/Search roles. Candidates with 4+ years of actual ML-specific roles receive a maximum boost, ensuring that total experience represents actual domain depth rather than general engineering.

### 5. Product vs. Services Company Context (10% Weight)
The job description explicitly flags candidates who have only worked at consulting/services firms (e.g. TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) as poor fits, while prioritizing product-company experience. Substring matching maps company names to identify:
* Product-company exposure: 1.0
* Relevant industries (SaaS, AI/ML, Fintech, E-commerce): 0.65
* Services-company only background: 0.15
* Neutral industry background: 0.35

### 6. Availability and Recruitability (13% Weight)
Weights are applied to candidate engagement metrics to ensure they are reachable:
* Recency of active logins (highest weight for active in past 30 days).
* Recruiter message response rate and average response speed.
* Stated notice period (under 30 days is prioritized).
* Interview completion rates, offer acceptance rates, profile completeness, verified email/phone, and willingness to relocate to Pune or Noida.

---

## Honeypots and Impossible Profile Detection

The challenge dataset contains subtle honeypots and trap candidates. The ranking engine applies a terminal penalty of 1.0 if any of the following impossible profile states are detected, dropping the candidate's final score to 0.0 and removing them from the shortlist:

### 1. Company Founding Date Verification
The engine inspects the start dates of employment at newly founded or established companies. If a candidate claims to work at a company before its founding year, they are flagged:
* Krutrim: founded in 2023
* Sarvam AI: founded in 2023
* Pinecone: founded in 2019
* Cred: founded in 2018
* Glance: founded in 2019
* OpenAI: founded in 2015
* PhonePe / Meesho: founded in 2015
* Swiggy / Razorpay / Postman: founded in 2014

### 2. Zero-Duration Expert Skills
If a candidate claims "expert" or "advanced" proficiency in any skill but specifies their duration of using that skill as 0 months, they are identified as a keyword stuffer.

### 3. Chronological Role Duration Violations
* Stated overall years of experience is smaller than the duration of any single work history role (with a 6-month buffer).
* Stated role duration (months) differs from the chronological span between start and end dates by more than 3 months.

### 4. Job Hopper / Title-Chaser Penalties
Candidates who average less than 18 months per role across 3+ roles while repeatedly climbing titles (e.g. changing companies every 12-15 months to jump from Senior to Lead to Staff) receive a standard penalty of 0.08, reflecting the job description's demand for team stability.

---

## Repository Contents

* `rank.py`: Command-line interface entry point.
* `src/redrob_ranker/ranker.py`: Core scoring, honeypot detection, tie-breaking, and reasoning generation logic.
* `requirements.txt`: Specified dependencies (empty, as the ranker runs on standard Python libraries).
* `Dockerfile`: Container definition for sandbox testing.
* `submission_metadata.yaml`: Participant metadata containing approach summary, contact information, and declarations.
* `data/sample_candidates.json`: The small 50-candidate sample from the challenge bundle.
* `data/validate_submission.py`: The official validator script.
* `outputs/team_redrob_ranker.csv`: The final, validated top 100 candidate shortlist.
* `outputs/top100_diagnostics.jsonl`: The diagnostic parameters used to score each of the top 100 candidates.
* `outputs/redrob_candidate_ranker_deck.pdf`: The PDF presentation explaining the methodology.
* `outputs/redrob_candidate_ranker_deck.pptx`: The raw PowerPoint slides.

---

## Setup and Reproducibility

### Setup
Ensure you have Python 3.11 or newer installed. The ranking code does not require any external package installations.

```bash
python --version
```

### Reproduce the Shortlist
To run the ranker on the full candidate pool and output the CSV file, run the following command:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/team_redrob_ranker.csv --diagnostics ./outputs/top100_diagnostics.jsonl
```

### Validate the Output
Run the official validator script on the generated CSV file to confirm it meets format, column, sorting, and tie-breaking requirements:

```bash
python data/validate_submission.py outputs/team_redrob_ranker.csv
```

### Small Sample Test
To run a quick test on the 50-candidate sample:

```bash
python rank.py --candidates ./data/sample_candidates.json --out ./outputs/sample_ranked.csv --top-k 10
```

### Docker Execution
To build and run the ranker inside a sandboxed container matching the evaluation environment constraints:

```bash
docker build -t redrob-ranker .
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/outputs:/app/outputs" redrob-ranker python rank.py --candidates ./data/candidates.jsonl --out ./outputs/team_redrob_ranker.csv
```
