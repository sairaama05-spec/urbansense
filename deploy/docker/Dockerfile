# ── UrbanSense – Production Docker Image ──────────────────────────────────────
# Base: PyTorch 2.2 + CUDA 12.1 + cuDNN 8 (runtime flavour keeps image slim)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="UrbanSense" \
      description="Automotive perception pipeline — DETR · Mask2Former · ByteTrack · Streamlit"

# ── system dependencies ───────────────────────────────────────────────────────
# libgl1 + libglib2.0-0  → OpenCV headless runtime
# curl                   → HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# Install GPU-compatible requirements; skip re-installing torch (already in base)
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements.txt

# ── application source ────────────────────────────────────────────────────────
COPY . .

# ── runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# ── health check (polls Streamlit's /healthz endpoint) ───────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app/streamlit/app.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.headless=true"]
