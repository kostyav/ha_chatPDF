FROM python:3.12-slim

WORKDIR /app

# Only the deps the agent needs
RUN pip install --no-cache-dir openai>=1.0 redis>=5.0 pyyaml>=6.0

# Copy the package root so `src.*` imports work
COPY pyproject.toml .
COPY src/part2/shared/ src/part2/shared/
COPY src/part3/ src/part3/

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "src.part3.agent"]
