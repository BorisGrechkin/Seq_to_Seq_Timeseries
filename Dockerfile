FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DOCKER_BUILDKIT=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y \
    gcc \
    && apt-get clean \
    && apt-get autoremove

WORKDIR /app
COPY Model/ Model/
COPY Scripts/ Scripts/
COPY config.env /

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118

CMD ["python", "main.py"]