FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py evaluate.py ./

ENTRYPOINT ["python", "/opt/program/train.py"]

