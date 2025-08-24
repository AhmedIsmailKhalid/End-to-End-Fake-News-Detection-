#!/bin/bash

# Robust startup script with error handling and health checks
set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Function to detect environment
detect_environment() {
    if [ -n "$SPACE_ID" ] || [ -f "/app/app.py" ] || [ -f "/app/streamlit_app.py" ]; then
        echo "huggingface_spaces"
    elif [ -f "/.dockerenv" ] || [ -n "$DOCKER_CONTAINER" ]; then
        echo "docker"
    elif [[ "$PWD" == /app* ]]; then
        echo "container"
    else
        echo "local"
    fi
}

# Function to check if port is available
check_port() {
    local port=$1
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln | grep -q ":$port "; then
            warning "Port $port is already in use"
            return 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln | grep -q ":$port "; then
            warning "Port $port is already in use"
            return 1
        fi
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        
        info "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    error "$service_name failed to start within expected time"
    return 1
}

# Function to test Python imports
test_python_imports() {
    log "Testing Python imports..."
    
    python3 -c "
import sys
import os
sys.path.append('/app')

try:
    from path_config import path_manager
    print(f'✅ Path manager loaded: {path_manager.environment}')
    
    import pandas as pd
    print('✅ Pandas imported')
    
    import sklearn
    print('✅ Scikit-learn imported') 
    
    import streamlit
    print('✅ Streamlit imported')
    
    import fastapi
    print('✅ FastAPI imported')
    
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        error "Python import test failed"
        return 1
    fi
    
    return 0
}

# Function to check system initialization 
check_system_initialization() {
    log "Checking system initialization..."
    
    python3 -c "
import sys
import os
sys.path.append('/app')

try:
    from path_config import path_manager
    
    # Check critical files
    critical_files = [
        path_manager.get_combined_dataset_path(),
        path_manager.get_model_file_path(),
        path_manager.get_vectorizer_path(),
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f'Missing files: {missing_files}')
        sys.exit(1)
    else:
        print('✅ All critical files present')
        
except Exception as e:
    print(f'❌ System check failed: {e}')
    sys.exit(1)
"
    
    return $?
}

# Function to handle shutdown gracefully
cleanup() {
    log "Shutting down services gracefully..."
    
    # Kill background processes
    if [ ! -z "$SCHEDULER_PID" ]; then
        kill $SCHEDULER_PID 2>/dev/null || true
        wait $SCHEDULER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null || true
        wait $FASTAPI_PID 2>/dev/null || true
    fi
    
    log "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main startup sequence
main() {
    local environment=$(detect_environment)
    
    log "🚀 Starting Fake News Detection System"
    log "🌍 Environment: $environment"
    log "📁 Working directory: $(pwd)"
    log "🐍 Python path: $(which python3)"
    
    # Test Python imports first
    if ! test_python_imports; then
        error "Python environment check failed"
        exit 1
    fi
    
    # Check and initialize system if needed
    if ! check_system_initialization; then
        log "System not initialized, running initialization..."
        python3 /app/initialize_system.py
        
        if [ $? -ne 0 ]; then
            error "System initialization failed"
            exit 1
        fi
        
        # Verify initialization
        if ! check_system_initialization; then
            error "System initialization verification failed"
            exit 1
        fi
    else
        log "✅ System already initialized"
    fi
    
    # Set PYTHONPATH to ensure imports work
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # Start FastAPI server
    log "🚀 Starting FastAPI server..."
    
    # Choose appropriate host based on environment
    if [ "$environment" = "huggingface_spaces" ]; then
        FASTAPI_HOST="0.0.0.0"
    else
        FASTAPI_HOST="127.0.0.1"
    fi
    
    uvicorn app.fastapi_server:app \
        --host $FASTAPI_HOST \
        --port 8000 \
        --log-level info \
        --access-log \
        --workers 1 &
    
    FASTAPI_PID=$!
    
    # Wait for FastAPI to be ready
    if ! wait_for_service "FastAPI" "http://127.0.0.1:8000/health"; then
        error "FastAPI failed to start"
        exit 1
    fi
    
    # Start background services only if not in HuggingFace Spaces
    if [ "$environment" != "huggingface_spaces" ]; then
        log "🔄 Starting background services..."
        
        # Start scheduler
        log "Starting scheduler..."
        python3 scheduler/schedule_tasks.py &> /app/logs/scheduler.log &
        SCHEDULER_PID=$!
        
        # Start drift monitor
        log "Starting drift monitor..."
        python3 monitor/monitor_drift.py &> /app/logs/monitor.log &
        MONITOR_PID=$!
    else
        log "ℹ️ Skipping background services in HuggingFace Spaces environment"
    fi
    
    # Verify system health before starting Streamlit
    log "🏥 Performing final health check..."
    python3 -c "
import sys
import requests
sys.path.append('/app')

try:
    response = requests.get('http://127.0.0.1:8000/health', timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f'✅ API Health: {data.get(\"status\", \"unknown\")}')
        print(f'✅ Environment: {data.get(\"environment_info\", {}).get(\"environment\", \"unknown\")}')
        print(f'✅ Model Available: {data.get(\"model_health\", {}).get(\"model_available\", False)}')
    else:
        print(f'❌ Health check failed: {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'❌ Health check error: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        error "Health check failed"
        exit 1
    fi
    
    # Start Streamlit (foreground)
    log "🎨 Starting Streamlit interface..."
    
    # Choose appropriate Streamlit configuration based on environment
    if [ "$environment" = "huggingface_spaces" ]; then
        # HuggingFace Spaces configuration
        exec streamlit run app/streamlit_app.py \
            --server.port=7860 \
            --server.address=0.0.0.0 \
            --server.enableCORS false \
            --server.enableXsrfProtection false \
            --server.maxUploadSize 200 \
            --server.enableStaticServing true \
            --logger.level info \
            --server.headless true
    else
        # Local/Docker configuration
        exec streamlit run app/streamlit_app.py \
            --server.port=7860 \
            --server.address=0.0.0.0 \
            --server.enableCORS false \
            --server.enableXsrfProtection false \
            --server.maxUploadSize 200 \
            --server.enableStaticServing true \
            --logger.level info
    fi
}

# Preliminary checks
preliminary_checks() {
    log "🔍 Running preliminary checks..."
    
    # Check if we're in the right directory
    if [ ! -f "/app/requirements.txt" ] && [ ! -f "requirements.txt" ]; then
        error "requirements.txt not found. Are we in the right directory?"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1)
    log "🐍 Python version: $python_version"
    
    # Check if we can import basic modules
    if ! python3 -c "import sys, os, json, pathlib" 2>/dev/null; then
        error "Basic Python modules not available"
        exit 1
    fi
    
    # Change to /app directory if we're not already there
    if [ "$(pwd)" != "/app" ] && [ -d "/app" ]; then
        log "📁 Changing to /app directory"
        cd /app
    fi
    
    log "✅ Preliminary checks passed"
}

# Environment-specific setup
setup_environment() {
    local environment=$(detect_environment)
    
    log "⚙️ Setting up environment: $environment"
    
    case $environment in
        "huggingface_spaces")
            log "🤗 Configuring for HuggingFace Spaces"
            export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
            export STREAMLIT_SERVER_PORT="7860"
            export FASTAPI_HOST="0.0.0.0"
            # Disable background services for HF Spaces
            export DISABLE_BACKGROUND_SERVICES="true"
            ;;
        "docker"|"container")
            log "🐳 Configuring for Docker/Container"
            export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
            export STREAMLIT_SERVER_PORT="7860"
            export FASTAPI_HOST="127.0.0.1"
            ;;
        "local")
            log "💻 Configuring for Local Development"
            export STREAMLIT_SERVER_ADDRESS="127.0.0.1"
            export STREAMLIT_SERVER_PORT="7860"
            export FASTAPI_HOST="127.0.0.1"
            ;;
    esac
    
    # Set up logging
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="/app:$PYTHONPATH"
    
    log "✅ Environment setup complete"
}

# Entry point
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    # Script is being executed directly
    preliminary_checks
    setup_environment
    main "$@"
fi