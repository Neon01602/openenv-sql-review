# ── Base image ────────────────────────────────────────────────────────────────
# Use python:3.11-slim WITHOUT a digest pin so the registry can serve any
# cached layer. The digest in the previous build caused the validator's
# Docker daemon to request a specific manifest that wasn't cached, triggering
# the httpReadSeeker 401/timeout error.
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HF Spaces) ────────────────────────────────────
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so this layer is cached between code-only changes
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY --chown=user . /app

# ── Switch to non-root user ───────────────────────────────────────────────────
USER user

# ── Expose port 7860 (mandatory for HF Spaces) ───────────────────────────────
EXPOSE 7860

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]