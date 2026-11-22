FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 7860
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120 app:app"