# Tamil handwriting recogniser - web UI (public demo).
# Build:  docker build -t tamil-ui .        Run:  docker run -p 7860:7860 tamil-ui
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MPLCONFIGDIR=/tmp/mpl TAMIL_UI_PUBLIC=1 PORT=7860 OMP_NUM_THREADS=2

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# only what serving needs: code, config, the exported model, the support set, class names, comparison table
COPY --chown=app config.yaml ./
COPY --chown=app src/ src/
COPY --chown=app ui/ ui/
COPY --chown=app models/ models/
COPY --chown=app data/support_set/ data/support_set/
COPY --chown=app data/processed/class_names.json data/processed/
COPY --chown=app reports/model_comparison.csv reports/

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if b'ready' in urllib.request.urlopen('http://127.0.0.1:%s/api/health' % __import__('os').environ.get('PORT','7860')).read() else 1)"
CMD ["sh", "-c", "uvicorn server:create_app --factory --app-dir ui --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
