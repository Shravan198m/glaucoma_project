FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including OpenCV and ReportLab pre-requisites)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl ca-certificates \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security and Hugging Face compatibility
RUN useradd -m -u 1000 user && \
    mkdir -p /app/outputs && \
    chown -R user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements and install them
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Download model weights and install gdown
RUN pip install --no-cache-dir --user gdown && \
    mkdir -p outputs/models && \
    gdown 1fIjbsob9aomFU2zKNg3vCNwH_IWdLjW4 -O outputs/models/best_model.pth

ENV GLAUCOMA_DEVICE=cpu

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]