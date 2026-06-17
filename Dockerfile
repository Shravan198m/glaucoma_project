FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl ca-certificates \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model weights from Google Drive using gdown to bypass the virus scan warning screen
RUN pip install --no-cache-dir gdown && \
    mkdir -p outputs/models && \
    gdown --id 1fIjbsob9aomFU2zKNg3vCNwH_IWdLjW4 -O outputs/models/best_model.pth

ENV GLAUCOMA_DEVICE=cpu
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
