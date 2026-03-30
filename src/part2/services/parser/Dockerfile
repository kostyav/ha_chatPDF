# Parser — Docling needs transformers>=4.48 (rt_detr_v2 layout model) and a GPU.
# Build context: src/part2/
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        libxcb1 libxcb-shm0 libxcb-xfixes0 \
        libgl1 libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/parser/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/parser/main.py main.py

ENV PYTHONPATH=/app

CMD ["python3", "main.py"]
