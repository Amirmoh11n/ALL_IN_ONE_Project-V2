FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY webapplication/backend/requirements.txt /tmp/web-requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/web-requirements.txt
COPY configs ./configs
COPY src ./src
COPY webapplication ./webapplication
COPY artifacts/exports ./artifacts/exports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["uvicorn", "webapplication.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
