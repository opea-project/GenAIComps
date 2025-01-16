# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY comps /home/user/comps
WORKDIR /home/user/comps/mcp_google_search

# Install all dependencies in pyproject.toml
RUN uv venv
# equivalent to source .venv/bin/activate
ENV PATH="/home/user/comps/mcp_google_search/.venv/bin:${PATH}"
RUN uv sync

ENTRYPOINT ["sleep", "infinity"]