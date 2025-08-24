#!/bin/bash

# Health check script for Docker container with dynamic path support
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[HEALTH]${NC} $1"
}

error() {
    echo -e "${RED}[HEALTH ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[HEALTH WARNING]${NC} $1"
}

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-10}
    
    if curl -s -f --max-time "$timeout" "$url" > /dev/null 2>&1; then
        log "$service_name is responding"
        return 0
    else
        error "$service_name health check failed"
        return 1
    fi
}

# Function to check file using Python path manager
check_files_with_python() {
    python3 -c "
import sys
import os
sys.path.append('/app')

try:
    from path_config import path_manager
    
    # Check critical files using the path manager
    critical_files = [
        (path_manager.get_combined_dataset_path(), 'Combined Dataset'),
        (path_manager.get_model_file_path(), 'Model File'),
        (path_manager.get_vectorizer_path(), 'Vectorizer File'),
        (path_manager.get_metadata_path(), 'Metadata File')
    ]
    
    missing_files = []
    for file_path, description in critical_files:
        if file_path.exists():
            print(f'✅ {description}: {file_path}')
        else:
            print(f'❌ {description}: {file_path}')
            missing_files.append(description)
    
    if missing_files:
        print(f'Missing files: {missing_files}')
        sys.exit(1)
    else:
        print('✅ All critical files present')
        
except ImportError as e:
    print(f'❌ Failed to import path_config: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Health check failed: {e}')
    sys.exit(1)
"
    return $?
}

# Function to check process
check_process() {
    local process_name=$1
    if pgrep -f "$process_name" > /dev/null 2>&1; then
        log "$process_name process is running"
        return 0
    else
        warning "$process_name process not found"
        return 1
    fi
}

# Main health check
main() {
    log "Starting comprehensive health check..."
    
    local exit_code=0
    
    # Check FastAPI service
    if ! check_service "FastAPI" "http://127.0.0.1:8000/health" 10; then
        exit_code=1
    fi
    
    # Check Streamlit service
    if ! check_service "Streamlit" "http://127.0.0.1:7860/_stcore/health" 10; then
        # Streamlit health endpoint might not always be available, try alternative
        if ! curl -s -f --max-time 10 "http://127.0.0.1:7860" > /dev/null 2>&1; then
            error "Streamlit health check failed"
            exit_code=1
        else
            log "Streamlit is responding (alternative check)"
        fi
    fi
    
    # Check required files using Python path manager
    log "Checking required files..."
    if ! check_files_with_python; then
        exit_code=1
    fi
    
    # Check critical processes (optional in some environments)
    log "Checking processes..."
    
    # Always check if Python processes are running
    if ! pgrep -f "streamlit" > /dev/null 2>&1; then
        warning "Streamlit process not found"
        # Don't fail on this as the service might still be responding
    fi
    
    if ! pgrep -f "uvicorn" > /dev/null 2>&1; then
        warning "FastAPI/Uvicorn process not found"
        # Don't fail on this as the service might still be responding
    fi
    
    # Check if scheduler is running (optional for some environments)
    if ! check_process "schedule_tasks.py"; then
        warning "Scheduler process not running (may be normal in some environments)"
        # Don't fail the health check for this
    fi
    
    # Check Python environment
    log "Checking Python environment..."
    if ! python3 -c "
import sys
sys.path.append('/app')

# Test critical imports
try:
    import pandas
    import sklearn
    import streamlit
    import fastapi
    from path_config import path_manager
    print(f'✅ Python environment OK (Environment: {path_manager.environment})')
except ImportError as e:
    print(f'❌ Python import failed: {e}')
    sys.exit(1)
"; then
        error "Python environment check failed"
        exit_code=1
    fi
    
    # Final status
    if [ $exit_code -eq 0 ]; then
        log "All health checks passed ✅"
    else
        error "Health check failed ❌"
    fi
    
    return $exit_code
}

# Execute main function
main "$@"