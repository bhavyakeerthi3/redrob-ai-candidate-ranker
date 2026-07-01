FROM python:3.11-slim

WORKDIR /app
COPY . /app

CMD ["python", "rank.py", "--candidates", "./data/sample_candidates.json", "--out", "./outputs/sample_submission.csv"]

