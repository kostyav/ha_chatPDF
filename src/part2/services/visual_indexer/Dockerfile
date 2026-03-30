# Visual indexer — Byaldi/ColQwen2 requires transformers>=4.47,<4.48 and a GPU.
# Build context: src/part2/
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/visual_indexer/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/visual_indexer/main.py main.py

ENV PYTHONPATH=/app

CMD ["python3", "main.py"]
