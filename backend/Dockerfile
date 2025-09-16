# Use Python 3.10 slim as base image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies required for ML libraries and data science packages
RUN apt-get update && apt-get install -y \
    # Build tools for Python packages
    gcc \
    g++ \
    build-essential \
    # Java for H2O and other ML libraries
    openjdk-17-jdk \
    # System utilities
    curl \
    wget \
    unzip \
    # For PDF processing
    poppler-utils \
    # For database connections
    libpq-dev \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set Java environment for H2O
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set working directory
WORKDIR /app

# Install Python dependencies
# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/logs /app/output

# Set proper permissions
RUN chmod +x /app

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
