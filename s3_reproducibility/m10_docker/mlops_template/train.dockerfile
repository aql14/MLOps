# 1. Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. Install essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 3. Setup caching for faster builds
ENV UV_LINK_MODE=copy

# 4. Copy application files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

# 5. Install dependencies using Cache Mount
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-cache --no-install-project

    
# 6. Set the entrypoint (Fixing the import path)
ENV PYTHONPATH="/src:/src/docker_project"
RUN mkdir -p models reports/figures
ENTRYPOINT ["uv", "run", "src/docker_project/train.py"]