FROM python:3.12-slim-bookworm
WORKDIR /mocks
COPY integration-tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY integration-tests/mock_services.py ./mock_services.py
CMD ["uvicorn", "mock_services:app", "--host", "0.0.0.0", "--port", "8000"]
