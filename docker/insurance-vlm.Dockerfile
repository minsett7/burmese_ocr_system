FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY services/insurance-vlm/pyproject.toml services/insurance-vlm/README.md ./
COPY services/insurance-vlm/src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 vlm \
    && mkdir -p /var/lib/insurance-vlm/jobs \
    && chown -R vlm:vlm /service /var/lib/insurance-vlm
USER vlm
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["uvicorn", "insurance_vlm_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
