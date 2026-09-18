FROM ghcr.io/astral-sh/uv:0.11-python3.12-trixie-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH" DB_PATH=/data/voicistant.db
WORKDIR /app

# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY app ./app
RUN useradd --system --uid 10001 app && install -d -o app -g app /data
USER app

# Host networking (WebRTC, SPEC §3.7): bind to localhost only, Caddy is the public entry point.
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8080", "--proxy-headers"]
