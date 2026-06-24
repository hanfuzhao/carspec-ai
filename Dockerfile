FROM python:3.9-slim

WORKDIR /app

# System dependencies (libgl1 replaces libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (lightweight, without torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/processed data/outputs

# Port
ENV PORT=7860
EXPOSE 7860

# Start
CMD ["python", "main.py"]
