# Text indexer — sentence-transformers + FAISS; no conflicting deps, CPU-only.
# Build context: src/part2/
FROM python:3.12-slim

WORKDIR /app

COPY services/text_indexer/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/text_indexer/main.py main.py

ENV PYTHONPATH=/app

CMD ["python", "main.py"]
