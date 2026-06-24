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

# Copy the application (includes the model weights from the LFS repository!)
COPY --chown=user:user . .

ENV GLAUCOMA_DEVICE=cpu

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]