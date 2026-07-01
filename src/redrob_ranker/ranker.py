from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 7, 1)

NONTECH_TITLES = {
    "accountant",
    "business analyst",
    "civil engineer",
    "content writer",
    "customer support",
    "graphic designer",
    "hr manager",
    "marketing manager",
    "mechanical engineer",
    "operations manager",
    "project manager",
    "sales executive",
}

SERVICES_COMPANIES = {
    "accenture",
    "capgemini",
    "cognizant",
    "hcl",
    "infosys",
    "mindtree",
    "mphasis",
    "tcs",
    "tech mahindra",
    "wipro",
}

PRODUCT_COMPANIES = {
    "amazon",
    "apple",
    "browserstack",
    "cred",
    "dream11",
    "flipkart",
    "freshworks",
    "glance",
    "google",
    "haptik",
    "inmobi",
    "krutrim",
    "linkedin",
    "meesho",
    "meta",
    "microsoft",
    "netflix",
    "nykaa",
    "observe.ai",
    "ola",
    "paytm",
    "phonepe",
    "pinecone",
    "policybazaar",
    "postman",
    "razorpay",
    "sarvam ai",
    "swiggy",
    "uber",
    "yellow.ai",
    "zomato",
    "zoho",
}

PREFERRED_LOCATIONS = {
    "pune": 1.0,
    "noida": 1.0,
    "delhi": 0.75,
    "gurgaon": 0.75,
    "hyderabad": 0.75,
    "mumbai": 0.65,
    "bangalore": 0.55,
}

CORE_SKILLS = {
    "embeddings": 1.2,
    "semantic search": 1.3,
    "vector search": 1.35,
    "information retrieval": 1.35,
    "recommendation systems": 1.25,
    "sentence transformers": 1.1,
    "faiss": 1.0,
    "pinecone": 1.0,
    "milvus": 0.95,
    "llms": 0.85,
    "rag": 0.9,
    "fine-tuning llms": 0.8,
    "lora": 0.55,
    "hugging face transformers": 0.65,
    "machine learning": 0.7,
    "nlp": 0.9,
    "mlops": 0.7,
    "mlflow": 0.45,
    "python": 0.9,
}

INFRA_SKILLS = {
    "airflow",
    "apache beam",
    "apache flink",
    "aws",
    "azure",
    "databricks",
    "docker",
    "fastapi",
    "gcp",
    "kafka",
    "kubernetes",
    "postgresql",
    "pyspark",
    "redis",
    "snowflake",
    "spark",
    "sql",
}

CV_SPEECH_SKILLS = {
    "computer vision",
    "gans",
    "image classification",
    "object detection",
    "speech recognition",
    "tts",
    "yolo",
}

CAREER_PATTERNS = {
    "retrieval": 1.3,
    "semantic search": 1.4,
    "vector search": 1.4,
    "search ranking": 1.35,
    "recommendation": 1.15,
    "ranking": 1.1,
    "embedd": 1.0,
    "rag": 0.8,
    "llm": 0.7,
    "fine-tun": 0.65,
    "nlp": 0.75,
    "production": 0.9,
    "deployed": 0.8,
    "real users": 0.85,
    "a/b": 0.8,
    "ndcg": 1.0,
    "mrr": 0.9,
    "offline benchmark": 0.8,
    "evaluation framework": 0.95,
    "eval framework": 0.95,
    "hybrid search": 1.2,
    "faiss": 0.8,
    "pinecone": 0.8,
    "milvus": 0.8,
    "opensearch": 0.7,
    "elasticsearch": 0.7,
}


@dataclass(order=True)
class CandidateScore:
    sort_key: tuple[float, str]
    candidate_id: str
    score: float
    reasoning: str
    diagnostics: dict[str, Any]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        y, m, d = (int(part) for part in value[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            yield from payload
            return
        raise ValueError(f"{path} must contain a JSON array of candidates")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def text_blob(candidate: dict[str, Any]) -> str:
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.extend(
            [
                role.get("title", ""),
                role.get("industry", ""),
                role.get("description", ""),
            ]
        )
    return norm(" ".join(parts))


def title_score(title: str) -> tuple[float, str]:
    t = norm(title)
    if "lead ai engineer" in t or "senior ai engineer" in t:
        return 1.0, "senior AI title"
    if "staff machine learning" in t or "senior machine learning" in t:
        return 0.95, "senior ML title"
    if "senior nlp" in t or "senior software engineer (ml)" in t:
        return 0.9, "senior ML/NLP title"
    if "search engineer" in t or "recommendation systems engineer" in t:
        return 0.9, "retrieval/recommendation title"
    if "applied ml" in t or "machine learning engineer" in t or t == "ai engineer":
        return 0.84, "applied ML title"
    if t in {"ml engineer", "nlp engineer"}:
        return 0.76, "ML/NLP title"
    if "data scientist" in t:
        return 0.58, "data science title"
    if "ai research" in t:
        return 0.45, "research-oriented AI title"
    if "software engineer" in t or "backend engineer" in t or "data engineer" in t:
        return 0.38, "adjacent engineering title"
    if t in NONTECH_TITLES:
        return -0.35, "non-target title"
    return 0.0, "neutral title"


def experience_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0:
        return 0.78
    if 9.0 < years <= 11.0:
        return 0.75
    if 3.0 <= years < 4.0:
        return 0.42
    if 11.0 < years <= 13.0:
        return 0.38
    return 0.12


def career_signal_score(blob: str) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for pattern, weight in CAREER_PATTERNS.items():
        if pattern in blob:
            score += weight
            hits.append(pattern)
    score = clamp(score / 8.5)
    return score, hits[:6]


def skill_signal(candidate: dict[str, Any]) -> tuple[float, list[str], dict[str, int]]:
    prof_mult = {"beginner": 0.35, "intermediate": 0.65, "advanced": 0.9, "expert": 1.0}
    total = 0.0
    matched: list[tuple[float, str]] = []
    counts = {"core": 0, "infra": 0, "cv_speech": 0, "thin_expert": 0}
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        key = norm(name)
        proficiency = prof_mult.get(skill.get("proficiency", "beginner"), 0.4)
        endorsements = math.log1p(skill.get("endorsements", 0)) / math.log(101)
        duration = skill.get("duration_months", 0) or 0
        duration_mult = clamp(duration / 36.0, 0.15, 1.0)
        trust = 0.48 * proficiency + 0.28 * duration_mult + 0.24 * endorsements

        if key in assessments:
            trust = 0.7 * trust + 0.3 * clamp(float(assessments[key]) / 100.0)
        elif name in assessments:
            trust = 0.7 * trust + 0.3 * clamp(float(assessments[name]) / 100.0)

        if key in CORE_SKILLS:
            contribution = CORE_SKILLS[key] * trust
            total += contribution
            matched.append((contribution, name))
            counts["core"] += 1
        if key in INFRA_SKILLS:
            total += 0.18 * trust
            counts["infra"] += 1
        if key in CV_SPEECH_SKILLS:
            counts["cv_speech"] += 1
        if skill.get("proficiency") in {"advanced", "expert"} and duration <= 3 and skill.get("endorsements", 0) <= 1:
            counts["thin_expert"] += 1

    matched.sort(reverse=True)
    return clamp(total / 7.5), [name for _, name in matched[:6]], counts


def product_company_score(candidate: dict[str, Any]) -> tuple[float, bool, bool]:
    companies = [norm(role.get("company", "")) for role in candidate.get("career_history", [])]
    industries = [norm(role.get("industry", "")) for role in candidate.get("career_history", [])]
    current_industry = norm(candidate["profile"].get("current_industry", ""))
    has_product = any(c in PRODUCT_COMPANIES for c in companies) or current_industry in {
        "ai/ml",
        "conversational ai",
        "e-commerce",
        "fintech",
        "healthtech ai",
        "internet",
        "saas",
        "software",
    }
    services_only = bool(companies) and all(c in SERVICES_COMPANIES for c in companies)
    if has_product:
        score = 1.0
    elif any(ind in {"software", "saas", "ai/ml", "fintech", "e-commerce"} for ind in industries):
        score = 0.65
    elif services_only:
        score = 0.15
    else:
        score = 0.35
    return score, has_product, services_only


def availability_score(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    reasons: list[str] = []

    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        inactive_days = max(0, (REFERENCE_DATE - last_active).days)
    else:
        inactive_days = 365

    recency = 1.0 if inactive_days <= 30 else 0.75 if inactive_days <= 75 else 0.45 if inactive_days <= 150 else 0.15
    response = clamp(float(signals.get("recruiter_response_rate", 0.0)))
    response_time = float(signals.get("avg_response_time_hours", 240.0))
    response_speed = 1.0 if response_time <= 12 else 0.85 if response_time <= 36 else 0.6 if response_time <= 96 else 0.25
    notice = int(signals.get("notice_period_days", 180) or 180)
    notice_score = 1.0 if notice <= 30 else 0.68 if notice <= 60 else 0.35 if notice <= 90 else 0.12
    interview = clamp(float(signals.get("interview_completion_rate", 0.0)))
    offer = float(signals.get("offer_acceptance_rate", -1.0))
    offer_score = 0.55 if offer < 0 else clamp(offer)
    github = float(signals.get("github_activity_score", -1.0))
    github_score = 0.45 if github < 0 else clamp(github / 100.0)
    completeness = clamp(float(signals.get("profile_completeness_score", 0.0)) / 100.0)

    available = (
        0.20 * recency
        + 0.18 * response
        + 0.12 * response_speed
        + 0.16 * notice_score
        + 0.12 * interview
        + 0.08 * offer_score
        + 0.07 * github_score
        + 0.07 * completeness
    )
    if signals.get("open_to_work_flag"):
        available += 0.06
        reasons.append("open to work")
    if signals.get("verified_email") and signals.get("verified_phone"):
        available += 0.02

    loc = norm(profile.get("location", ""))
    country = norm(profile.get("country", ""))
    loc_score = 0.0
    for key, value in PREFERRED_LOCATIONS.items():
        if key in loc:
            loc_score = max(loc_score, value)
    if not loc_score and country == "india":
        loc_score = 0.35
    if signals.get("willing_to_relocate"):
        loc_score = max(loc_score, 0.62 if country == "india" else 0.35)
        reasons.append("relocation ok")
    if inactive_days <= 75:
        reasons.append(f"active {inactive_days}d ago")
    if response >= 0.65:
        reasons.append(f"{response:.0%} recruiter response")
    if notice <= 30:
        reasons.append(f"{notice}d notice")

    return clamp(0.72 * available + 0.28 * loc_score), reasons


def trap_penalty(candidate: dict[str, Any], blob: str, skill_counts: dict[str, int], career_hits: list[str]) -> tuple[float, list[str]]:
    profile = candidate["profile"]
    title = norm(profile.get("current_title", ""))
    years = float(profile.get("years_of_experience", 0.0))
    roles = candidate.get("career_history", [])
    penalties = 0.0
    reasons: list[str] = []

    # --- Honeypot / Impossible Profile Detection ---
    is_honeypot = False

    # 1. Company founding date anomalies
    founding_years = {
        "krutrim": 2023,
        "sarvam ai": 2023,
        "pinecone": 2019,
        "cred": 2018,
        "glance": 2019,
        "phonepe": 2015,
        "meesho": 2015,
        "swiggy": 2014,
        "razorpay": 2014,
        "postman": 2014,
        "haptik": 2013,
        "nykaa": 2012,
        "browserstack": 2011,
        "freshworks": 2010,
        "ola": 2010,
        "paytm": 2010,
        "zomato": 2008,
        "dream11": 2008,
        "observe.ai": 2017,
        "yellow.ai": 2016,
        "openai": 2015
    }
    for role in roles:
        company = norm(role.get("company", ""))
        start_date = role.get("start_date")
        if start_date:
            try:
                year = int(start_date.split("-")[0])
                for comp, f_year in founding_years.items():
                    if comp in company and year < f_year:
                        is_honeypot = True
                        reasons.append(f"impossible company founding date ({role.get('company')})")
                        break
            except Exception:
                pass
        if is_honeypot:
            break

    # 2. Expert/Advanced skill with 0 months duration
    for s in candidate.get("skills", []):
        if s.get("proficiency") in {"expert", "advanced"} and s.get("duration_months", -1) == 0:
            is_honeypot = True
            reasons.append(f"expert skill with zero duration ({s.get('name')})")
            break

    # 3. Role duration exceeds total years of experience
    for role in roles:
        dur_months = role.get("duration_months", 0)
        if dur_months > (years * 12 + 6):
            is_honeypot = True
            reasons.append("role duration exceeds total experience")
            break

    # 4. Role span mismatch with duration_months
    for role in roles:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date")) or REFERENCE_DATE
        dur_months = role.get("duration_months", 0)
        if start and end:
            span_months = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(span_months - dur_months) > 3:
                is_honeypot = True
                reasons.append("role duration timeline discrepancy")
                break

    if is_honeypot:
        return 1.0, reasons

    # --- Standard Penalties ---
    if title in NONTECH_TITLES and (skill_counts["core"] >= 4 or "rag" in blob or "llm" in blob):
        penalties += 0.22
        reasons.append("AI keyword/title mismatch")

    if skill_counts["thin_expert"] >= 4:
        penalties += 0.12
        reasons.append("thin expert skills")

    if skill_counts["cv_speech"] >= 3 and skill_counts["core"] <= 2 and not any(hit in career_hits for hit in ["retrieval", "ranking", "recommendation", "semantic search"]):
        penalties += 0.14
        reasons.append("CV/speech heavy")

    total_months = sum(int(role.get("duration_months", 0) or 0) for role in roles)
    if total_months and abs(total_months / 12.0 - years) > 5.0:
        penalties += 0.08
        reasons.append("experience mismatch")

    future_roles = 0
    very_short_roles = 0
    for role in roles:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        if start and start > REFERENCE_DATE:
            future_roles += 1
        if end and start and end < start:
            future_roles += 1
        if int(role.get("duration_months", 0) or 0) < 9:
            very_short_roles += 1
    if future_roles:
        penalties += 0.25
        reasons.append("invalid dates")
    if len(roles) >= 3 and very_short_roles >= 2:
        penalties += 0.06
        reasons.append("short tenures")

    signals = candidate.get("redrob_signals", {})
    last_active = parse_date(signals.get("last_active_date"))
    inactive_days = 365 if last_active is None else max(0, (REFERENCE_DATE - last_active).days)
    if inactive_days > 180:
        penalties += 0.08
        reasons.append("inactive")
    if float(signals.get("recruiter_response_rate", 0.0)) < 0.12:
        penalties += 0.07
        reasons.append("low response")

    if "langchain" in blob and len(career_hits) <= 2 and years < 4:
        penalties += 0.08
        reasons.append("framework-only signal")

    return clamp(penalties, 0.0, 0.65), reasons


def score_candidate(candidate: dict[str, Any]) -> CandidateScore:
    profile = candidate["profile"]
    candidate_id = candidate["candidate_id"]
    blob = text_blob(candidate)

    t_score, title_reason = title_score(profile.get("current_title", ""))
    exp = experience_score(float(profile.get("years_of_experience", 0.0)))
    career, career_hits = career_signal_score(blob)
    skills, matched_skills, skill_counts = skill_signal(candidate)
    product, has_product, services_only = product_company_score(candidate)
    availability, availability_reasons = availability_score(candidate)
    penalty, penalty_reasons = trap_penalty(candidate, blob, skill_counts, career_hits)

    if t_score < 0:
        title_component = 0.0
    else:
        title_component = t_score

    raw = (
        0.24 * title_component
        + 0.25 * career
        + 0.18 * skills
        + 0.10 * exp
        + 0.10 * product
        + 0.13 * availability
    )

    # Skill keywords should not let a non-target title outrank real ML/search builders.
    if t_score < 0 and career < 0.45:
        raw *= 0.62
    if services_only and not has_product and career < 0.6:
        raw *= 0.88
    if "research" in title_reason and "production" not in career_hits and "deployed" not in career_hits:
        raw *= 0.88

    score = clamp(raw - penalty)

    reasoning = build_reasoning(
        candidate,
        score,
        title_reason,
        career_hits,
        matched_skills,
        availability_reasons,
        penalty_reasons,
        has_product,
        services_only,
    )

    diagnostics = {
        "title": round(title_component, 3),
        "career": round(career, 3),
        "skill_score": round(skills, 3),
        "experience": round(exp, 3),
        "product": round(product, 3),
        "availability": round(availability, 3),
        "penalty": round(penalty, 3),
        "career_hits": career_hits,
        "matched_skills": matched_skills,
    }
    return CandidateScore((score, candidate_id), candidate_id, score, reasoning, diagnostics)


def build_reasoning(
    candidate: dict[str, Any],
    score: float,
    title_reason: str,
    career_hits: list[str],
    matched_skills: list[str],
    availability_reasons: list[str],
    penalty_reasons: list[str],
    has_product: bool,
    services_only: bool,
) -> str:
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "candidate")
    years = float(profile.get("years_of_experience", 0.0))
    location = profile.get("location", "unknown location")
    notice = signals.get("notice_period_days", "unknown")
    response = float(signals.get("recruiter_response_rate", 0.0))
    skill_text = ", ".join(matched_skills[:3]) if matched_skills else "limited explicit retrieval skills"
    career_text = ", ".join(career_hits[:3]) if career_hits else "adjacent engineering evidence"

    company_note = "product-company exposure" if has_product else "mostly services background" if services_only else "some relevant industry exposure"
    availability_note = "; ".join(availability_reasons[:2]) if availability_reasons else f"{notice}d notice and {response:.0%} recruiter response"

    concern = ""
    if penalty_reasons:
        concern = f" Concern: {', '.join(penalty_reasons[:2])}."
    elif score < 0.52:
        concern = " Concern: weaker direct retrieval evidence than higher-ranked profiles."

    return (
        f"{title} with {years:.1f} years in {location}; {career_text} plus {skill_text}. "
        f"Fits the JD through {title_reason}, {company_note}, and availability signals ({availability_note}).{concern}"
    )


def rank_candidates(candidates_path: Path, top_k: int = 100) -> list[CandidateScore]:
    heap: list[CandidateScore] = []
    for candidate in iter_jsonl(candidates_path):
        scored = score_candidate(candidate)
        if len(heap) < top_k:
            heapq.heappush(heap, scored)
        elif scored.sort_key > heap[0].sort_key:
            heapq.heapreplace(heap, scored)
    return sorted(heap, key=lambda item: (-item.score, item.candidate_id))


def write_submission(rows: list[CandidateScore], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_score = rows[0].score if rows else 1.0
    min_score = rows[-1].score if rows else 0.0
    span = max(max_score - min_score, 1e-9)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(rows, start=1):
            calibrated = 0.52 + 0.46 * ((row.score - min_score) / span)
            calibrated -= rank * 0.000001
            writer.writerow([row.candidate_id, rank, f"{calibrated:.6f}", row.reasoning])


def write_diagnostics(rows: list[CandidateScore], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rank, row in enumerate(rows, start=1):
            payload = {
                "rank": rank,
                "candidate_id": row.candidate_id,
                "score": row.score,
                "reasoning": row.reasoning,
                "diagnostics": row.diagnostics,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--top-k", default=100, type=int, help="Number of rows to emit")
    parser.add_argument("--diagnostics", type=Path, help="Optional JSONL file with feature diagnostics for the top rows")
    args = parser.parse_args()

    rows = rank_candidates(args.candidates, args.top_k)
    write_submission(rows, args.out)
    if args.diagnostics:
        write_diagnostics(rows, args.diagnostics)


if __name__ == "__main__":
    main()
