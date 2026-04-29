FROM python:3.12-slim

WORKDIR /app

# git is needed to install ollama-deep-researcher from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies from lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY server.py research.py ./
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8001"]
