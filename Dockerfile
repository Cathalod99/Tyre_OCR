# Minimal Dockerfile - guaranteed to work (v3)
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
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre-9369d891cdc1.json
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create necessary directories
RUN mkdir -p /app/temp /app/doc/img

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "backend/main.py"]
