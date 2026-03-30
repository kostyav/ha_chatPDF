# Orchestrator — lightweight coordinator; no ML deps, CPU-only.
# Build context: src/part2/
FROM python:3.12-slim

WORKDIR /app

COPY services/orchestrator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/orchestrator/main.py main.py

ENV PYTHONPATH=/app

CMD ["python", "main.py"]
