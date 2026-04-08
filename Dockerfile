FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app
CMD ["python", "inference.py"]
