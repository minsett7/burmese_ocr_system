FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY services/insurance-vlm /service
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements-gpu.txt
RUN useradd --create-home --uid 10001 vlm \
    && mkdir -p /var/lib/insurance-vlm/jobs /models/huggingface \
    && chown -R vlm:vlm /service /var/lib/insurance-vlm /models
USER vlm
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "insurance_vlm_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
