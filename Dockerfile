FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "src/pipeline.py"]