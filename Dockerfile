FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r <(python -c "import yaml;d=yaml.safe_load(open('openenv.yaml'));print('\n'.join(d['dependencies']))")
ENV PYTHONPATH=/app
CMD ["python", "scripts/run_baseline.py"]
