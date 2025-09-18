# Minimal Dockerfile - guaranteed to work (v2)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install only the absolute minimum for opencv-python-headless
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

# Copy Google Cloud credentials
COPY ocrtyre.json /app/ocrtyre.json

# Create models directory and copy YOLO model
RUN mkdir -p models
COPY models/Tyre_Detect.onnx /app/models/Tyre_Detect.onnx

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre.json

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
