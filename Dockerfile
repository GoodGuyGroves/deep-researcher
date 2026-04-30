FROM python:3.13-slim

LABEL org.opencontainers.image.title="deep-researcher" \
      org.opencontainers.image.description="Iterative web research pipeline with LLM summarization and optional OpenViking ingestion" \
      org.opencontainers.image.source="https://github.com/GoodGuyGroves/deep-researcher" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install uv for fast, reproducible dependency management
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

# Install dependencies from lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY engine.py server.py research.py ./
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8001"]
