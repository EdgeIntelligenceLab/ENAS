# ENAS reproducibility image (CPU-only)
FROM python:3.10-slim
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . || true
CMD ["python", "scripts/build_camera_ready_tables.py"]
