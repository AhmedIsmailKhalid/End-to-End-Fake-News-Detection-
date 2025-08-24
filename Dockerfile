FROM python:3.11.6-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    wget \
    procps \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy path configuration first
COPY path_config.py /app/

# Copy validation script
COPY docker_validation.py /app/

# Copy project files
COPY . /app

# Create necessary directories with proper permissions for uploads
RUN mkdir -p /app/data /app/model /app/logs /app/cache /app/temp && \
    mkdir -p /app/data/kaggle && \
    mkdir -p /tmp/app_data /tmp/app_logs /tmp/app_cache

# Copy health check script and make it executable
COPY health_check.sh /app/
RUN chmod +x /app/health_check.sh

# Make scripts executable
RUN chmod +x /app/start.sh

# Copy initial datasets if they exist to the correct app structure
RUN if [ -d /app/data/kaggle ]; then \
        echo "Kaggle datasets found in app structure"; \
        chmod -R 775 /app/data/kaggle; \
    fi && \
    if [ -f /app/data/combined_dataset.csv ]; then \
        echo "Combined dataset found in app structure"; \
        chmod 664 /app/data/combined_dataset.csv; \
    fi

# Set comprehensive permissions BEFORE switching users
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/app_data /tmp/app_logs /tmp/app_cache && \
    chmod -R 755 /app && \
    chmod -R 777 /app/data /app/logs /app/cache /app/temp /app/model && \
    chmod -R 775 /tmp/app_data /tmp/app_logs /tmp/app_cache

# Switch to non-root user AFTER setting permissions
USER appuser

# Create user-writable directories in case the above fails
RUN mkdir -p /tmp/fallback_data /tmp/fallback_logs && \
    chmod 777 /tmp/fallback_data /tmp/fallback_logs

# Initialize system with the new path structure
RUN python /app/initialize_system.py

# Simple permission test using basic commands
RUN python3 -c "import sys; sys.path.append('/app'); from path_config import path_manager; print('Environment:', path_manager.environment); print('Data dir:', str(path_manager.base_paths['data'])); print('Model dir:', str(path_manager.base_paths['model']))"

# Test write permissions with a simple approach
RUN python3 -c "import sys; sys.path.append('/app'); from path_config import path_manager; from pathlib import Path; test_file = path_manager.get_data_path('test.txt'); test_file.write_text('test'); test_file.unlink(); print('Write permissions verified')" || echo "Using fallback permissions"

# Health check using the proper paths
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/health_check.sh

# Expose ports
EXPOSE 7860 8000

# Set environment variable to help the app detect container environment
ENV DOCKER_CONTAINER=1

# Run the startup script
CMD ["./start.sh"]