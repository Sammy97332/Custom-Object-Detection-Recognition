FROM python:3.11-slim

ENV YOLO_CONFIG_DIR=/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_TRIM_THRESHOLD_=65536

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt gunicorn && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

COPY . .

EXPOSE 7860
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 1 --timeout 120 app:app"