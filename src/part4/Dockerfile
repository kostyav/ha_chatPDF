FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir openai>=1.0 redis>=5.0 fastapi>=0.111 uvicorn[standard]>=0.29

COPY pyproject.toml .
COPY src/part2/shared/ src/part2/shared/
COPY src/part3/ src/part3/
COPY src/part4/ src/part4/

RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "src.part4.main:app", "--host", "0.0.0.0", "--port", "8080"]
