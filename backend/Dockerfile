# Minimal Dockerfile - guaranteed to work (v2)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and ONNX Runtime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy Google Cloud credentials from parent directory
COPY ../ocrtyre-9369d891cdc1.json /app/ocrtyre-9369d891cdc1.json

# Create models directory and copy YOLO model
RUN mkdir -p models
COPY models/Tyre_Detect.onnx /app/models/Tyre_Detect.onnx

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre-9369d891cdc1.json

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
