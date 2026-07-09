FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --no-compile --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && sed -E '/^(kaggle|albumentations)([<>=!~].*)?$/d' requirements.txt > /tmp/requirements-runtime.txt \
    && pip install --no-cache-dir --no-compile -r /tmp/requirements-runtime.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
