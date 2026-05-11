# syntax=docker/dockerfile:1.6

# ----- Stage 1: build the TypeScript MCP server -----
FROM node:20-bookworm-slim AS mcp-builder
WORKDIR /build/mcp-server

# Install with the lockfile when it exists, otherwise fall back to a fresh install.
COPY mcp-server/package.json mcp-server/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY mcp-server/tsconfig.json ./
COPY mcp-server/src ./src
RUN npm run build


# ----- Stage 2: runtime (Python + Node) -----
FROM python:3.12-slim AS runtime

# Node.js 20 — the Python app spawns the MCP server as a stdio child process.
# fonts-dejavu gives the Download-PDF action a Unicode-capable font family
# with all four variants (regular/bold/italic/bold-italic). Markdown
# emphasis (*text*) requires the italic variant, which lives in the
# fonts-dejavu-extra sub-package — fonts-dejavu pulls in both.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs fonts-dejavu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies first (cached on every code-only change).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# MCP server: production deps + prebuilt JS from the builder stage.
COPY mcp-server/package.json mcp-server/package-lock.json* mcp-server/
RUN cd mcp-server \
    && (if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi) \
    && npm cache clean --force
COPY --from=mcp-builder /build/mcp-server/dist mcp-server/dist

# Streamlit app
COPY app.py mcp_client.py orchestrator.py reports.py ./

EXPOSE 8501

# Bind to 0.0.0.0 so Docker can publish the port; disable telemetry & CORS check
# for a clean local demo.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
