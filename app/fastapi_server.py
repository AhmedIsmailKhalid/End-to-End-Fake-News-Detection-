import json
import time
import joblib
import logging
import hashlib
import uvicorn
import asyncio
import aiofiles
import traceback
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status

from data.data_validator import (
    DataValidationPipeline, validate_text, validate_articles_list, 
    get_validation_stats, generate_quality_report
)
from data.validation_schemas import ValidationLevel, ValidationResultSchema
 
from model.retrain import AutomatedRetrainingManager
from monitor.metrics_collector import MetricsCollector
from monitor.prediction_monitor import PredictionMonitor
from monitor.alert_system import AlertSystem, console_notification_handler

from deployment.traffic_router import TrafficRouter
from deployment.model_registry import ModelRegistry
from deployment.blue_green_manager import BlueGreenDeploymentManager


# Import the new path manager
try:
    from path_config import path_manager
except ImportError:
    # Fallback for development environments
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from path_config import path_manager

# Configure logging with fallback for permission issues
def setup_logging():
    """Setup logging with fallback for environments with restricted file access"""
    handlers = [logging.StreamHandler()]  # Always include console output
    
    try:
        # Try to create log file in the logs directory
        log_file_path = path_manager.get_logs_path('fastapi_server.log')
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test if we can write to the file
        test_handler = logging.FileHandler(log_file_path)
        test_handler.close()
        
        # If successful, add file handler
        handlers.append(logging.FileHandler(log_file_path))
        print(f"Logging to file: {log_file_path}")  # Use print instead of logger
        
    except (PermissionError, OSError) as e:
        # If file logging fails, just use console logging
        print(f"Cannot create log file, using console only: {e}")
        
        # Try alternative locations for file logging
        try:
            import tempfile
            temp_log = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, prefix='fastapi_')
            temp_log.close()
            handlers.append(logging.FileHandler(temp_log.name))
            print(f"Using temporary log file: {temp_log.name}")
        except Exception as temp_e:
            print(f"Temporary file logging also failed: {temp_e}")
    
    return handlers

# Setup logging with error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=setup_logging()
)
logger = logging.getLogger(__name__)

# Now that logger is defined, log the environment info
try:
    path_manager.log_environment_info()
except Exception as e:
    logger.warning(f"Could not log environment info: {e}")

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage
rate_limit_storage = defaultdict(list)


class ModelManager:
    """Manages model loading and health checks with dynamic paths"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.model_metadata = {}
        self.last_health_check = None
        self.health_status = "unknown"
        self.load_model()

    def load_model(self):
        """Load model with comprehensive error handling and dynamic paths"""
        try:
            logger.info("Loading ML model...")

            # Initialize all to None first
            self.model = None
            self.vectorizer = None
            self.pipeline = None

            # Try to load pipeline first (preferred)
            pipeline_path = path_manager.get_pipeline_path()
            logger.info(f"Checking for pipeline at: {pipeline_path}")
            
            if pipeline_path.exists():
                try:
                    self.pipeline = joblib.load(pipeline_path)
                    # Extract components from pipeline
                    if hasattr(self.pipeline, 'named_steps'):
                        self.model = self.pipeline.named_steps.get('model')
                        self.vectorizer = (self.pipeline.named_steps.get('vectorizer') or 
                                         self.pipeline.named_steps.get('vectorize'))
                    logger.info("Loaded model pipeline successfully")
                    logger.info(f"Pipeline steps: {list(self.pipeline.named_steps.keys()) if hasattr(self.pipeline, 'named_steps') else 'No named_steps'}")
                except Exception as e:
                    logger.warning(f"Failed to load pipeline: {e}, falling back to individual components")
                    self.pipeline = None
            else:
                logger.info(f"Pipeline file not found at {pipeline_path}")

            # If pipeline loading failed or doesn't exist, load individual components
            if self.pipeline is None:
                model_path = path_manager.get_model_file_path()
                vectorizer_path = path_manager.get_vectorizer_path()
                
                logger.info(f"Checking for model at: {model_path}")
                logger.info(f"Checking for vectorizer at: {vectorizer_path}")

                if model_path.exists() and vectorizer_path.exists():
                    try:
                        self.model = joblib.load(model_path)
                        self.vectorizer = joblib.load(vectorizer_path)
                        logger.info("Loaded model components successfully")
                    except Exception as e:
                        logger.error(f"Failed to load individual components: {e}")
                        raise e
                else:
                    raise FileNotFoundError(f"No model files found. Checked:\n- {pipeline_path}\n- {model_path}\n- {vectorizer_path}")

            # Verify we have what we need for predictions
            if self.pipeline is None and (self.model is None or self.vectorizer is None):
                raise ValueError("Neither complete pipeline nor individual model components are available")

            # Load metadata
            metadata_path = path_manager.get_metadata_path()
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {self.model_metadata.get('model_version', 'Unknown')}")
            else:
                logger.warning(f"Metadata file not found at: {metadata_path}")
                self.model_metadata = {"model_version": "unknown"}

            self.health_status = "healthy"
            self.last_health_check = datetime.now()

            # Log what was successfully loaded
            logger.info(f"Model loading summary:")
            logger.info(f"  Pipeline available: {self.pipeline is not None}")
            logger.info(f"  Model available: {self.model is not None}")
            logger.info(f"  Vectorizer available: {self.vectorizer is not None}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.health_status = "unhealthy"
            self.model = None
            self.vectorizer = None
            self.pipeline = None

    def predict(self, text: str) -> tuple[str, float]:
        """Make prediction with error handling"""
        try:
            if self.pipeline:
                # Use pipeline for prediction
                prediction = self.pipeline.predict([text])[0]
                probabilities = self.pipeline.predict_proba([text])[0]
                logger.debug("Used pipeline for prediction")
            elif self.model and self.vectorizer:
                # Use individual components
                X = self.vectorizer.transform([text])
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0]
                logger.debug("Used individual components for prediction")
            else:
                raise ValueError("No model available for prediction")

            # Get confidence score
            confidence = float(max(probabilities))

            # Convert prediction to readable format
            label = "Fake" if prediction == 1 else "Real"

            return label, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test prediction with sample text
            test_text = "This is a test article for health check purposes."
            label, confidence = self.predict(test_text)

            self.health_status = "healthy"
            self.last_health_check = datetime.now()

            return {
                "status": "healthy",
                "last_check": self.last_health_check.isoformat(),
                "model_available": self.model is not None,
                "vectorizer_available": self.vectorizer is not None,
                "pipeline_available": self.pipeline is not None,
                "test_prediction": {"label": label, "confidence": confidence},
                "environment": path_manager.environment,
                "model_path": str(path_manager.get_model_file_path()),
                "vectorizer_path": str(path_manager.get_vectorizer_path()),
                "pipeline_path": str(path_manager.get_pipeline_path()),
                "data_path": str(path_manager.get_data_path()),
                "file_exists": {
                    "model": path_manager.get_model_file_path().exists(),
                    "vectorizer": path_manager.get_vectorizer_path().exists(),
                    "pipeline": path_manager.get_pipeline_path().exists(),
                    "metadata": path_manager.get_metadata_path().exists()
                }
            }

        except Exception as e:
            self.health_status = "unhealthy"
            self.last_health_check = datetime.now()

            return {
                "status": "unhealthy",
                "last_check": self.last_health_check.isoformat(),
                "error": str(e),
                "model_available": self.model is not None,
                "vectorizer_available": self.vectorizer is not None,
                "pipeline_available": self.pipeline is not None,
                "environment": path_manager.environment,
                "model_path": str(path_manager.get_model_file_path()),
                "vectorizer_path": str(path_manager.get_vectorizer_path()),
                "pipeline_path": str(path_manager.get_pipeline_path()),
                "data_path": str(path_manager.get_data_path()),
                "file_exists": {
                    "model": path_manager.get_model_file_path().exists(),
                    "vectorizer": path_manager.get_vectorizer_path().exists(),
                    "pipeline": path_manager.get_pipeline_path().exists(),
                    "metadata": path_manager.get_metadata_path().exists()
                }
            }


# Background task functions
async def log_prediction(text: str, prediction: str, confidence: float, client_ip: str, processing_time: float):
    """Log prediction details with error handling for file access"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "text_length": len(text),
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "text_hash": hashlib.md5(text.encode()).hexdigest()
        }

        # Try to save to log file
        try:
            log_file = path_manager.get_logs_path("prediction_log.json")

            # Load existing logs
            logs = []
            if log_file.exists():
                try:
                    async with aiofiles.open(log_file, 'r') as f:
                        content = await f.read()
                        logs = json.loads(content)
                except:
                    logs = []

            # Add new log
            logs.append(log_entry)

            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]

            # Save logs
            async with aiofiles.open(log_file, 'w') as f:
                await f.write(json.dumps(logs, indent=2))
                
        except (PermissionError, OSError) as e:
            # If file logging fails, just log to console
            logger.warning(f"Cannot write prediction log to file: {e}")
            logger.info(f"Prediction logged: {json.dumps(log_entry)}")

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


async def log_batch_prediction(total_texts: int, successful_predictions: int, client_ip: str, processing_time: float):
    """Log batch prediction details"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "batch_prediction",
            "client_ip": client_ip,
            "total_texts": total_texts,
            "successful_predictions": successful_predictions,
            "processing_time": processing_time,
            "success_rate": successful_predictions / total_texts if total_texts > 0 else 0
        }

        logger.info(f"Batch prediction logged: {json.dumps(log_entry)}")

    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")


# Global variables
model_manager = ModelManager()

# Initialize automation manager
automation_manager = None

# Initialize deployment components
deployment_manager = None
traffic_router = None
model_registry = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with deployment system"""
    global deployment_manager, traffic_router, model_registry
    
    logger.info("Starting FastAPI application...")
    
    # Startup tasks
    model_manager.load_model()
    
    # Initialize deployment components
    try:
        deployment_manager = BlueGreenDeploymentManager()
        traffic_router = TrafficRouter()
        model_registry = ModelRegistry()
        logger.info("Deployment system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize deployment system: {e}")
    
    # Initialize monitoring and automation...
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down FastAPI application...")

# Initialize monitoring components
prediction_monitor = PredictionMonitor(base_dir=Path("/tmp"))
metrics_collector = MetricsCollector(base_dir=Path("/tmp"))
alert_system = AlertSystem(base_dir=Path("/tmp"))

# Start monitoring
prediction_monitor.start_monitoring()

alert_system.add_notification_handler("console", console_notification_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("Starting FastAPI application...")

    # Startup tasks
    model_manager.load_model()

    # Schedule periodic health checks
    asyncio.create_task(periodic_health_check())

    yield

    # Shutdown tasks
    logger.info("Shutting down FastAPI application...")


# Background tasks
async def periodic_health_check():
    """Periodic health check"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            health_status = model_manager.health_check()

            if health_status["status"] == "unhealthy":
                logger.warning(
                    "Model health check failed, attempting to reload...")
                model_manager.load_model()

        except Exception as e:
            logger.error(f"Periodic health check failed: {e}")


# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="Production-ready API for fake news detection with comprehensive monitoring and security features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom OpenAPI setup - RIGHT AFTER app creation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Fake News Detection API",
        version="2.0.0",
        description="Production-ready API for fake news detection with comprehensive monitoring and security features",
        routes=app.routes,
    )

    # Add security definitions
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Set the custom OpenAPI function
app.openapi = custom_openapi


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to analyze for fake news detection")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')

        # Basic content validation
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters long')

        # Check for suspicious patterns
        suspicious_patterns = ['<script', 'javascript:', 'data:']
        if any(pattern in v.lower() for pattern in suspicious_patterns):
            raise ValueError('Text contains suspicious content')

        return v.strip()


class PredictionResponse(BaseModel):
    prediction: str = Field(...,
                            description="Prediction result: 'Real' or 'Fake'")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Confidence score between 0 and 1")
    model_version: str = Field(...,
                               description="Version of the model used for prediction")
    timestamp: str = Field(..., description="Timestamp of the prediction")
    processing_time: float = Field(...,
                                   description="Time taken for processing in seconds")


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10,
                             description="List of texts to analyze")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')

        for text in v:
            if not text or not text.strip():
                raise ValueError('All texts must be non-empty')

            if len(text.strip()) < 10:
                raise ValueError(
                    'All texts must be at least 10 characters long')

        return [text.strip() for text in v]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_health: Dict[str, Any]
    system_health: Dict[str, Any]
    api_health: Dict[str, Any]
    environment_info: Dict[str, Any]


# Rate limiting
async def rate_limit_check(request: Request):
    """Check rate limits"""
    client_ip = request.client.host
    current_time = time.time()

    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if current_time - timestamp < 3600  # 1 hour window
    ]

    # Check rate limit (100 requests per hour)
    if len(rate_limit_storage[client_ip]) >= 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 100 requests per hour."
        )

    # Add current request
    rate_limit_storage[client_ip].append(current_time)


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    log_data = {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "status_code": response.status_code,
        "process_time": process_time,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"Request: {json.dumps(log_data)}")

    return response


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    error_data = {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }

    logger.error(f"HTTP Exception: {json.dumps(error_data)}")

    return JSONResponse(
        status_code=exc.status_code,
        content=error_data
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_data = {
        "error": True,
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }

    logger.error(f"General Exception: {str(exc)}\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content=error_data
    )


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Fake News Detection API",
        "version": "2.0.0",
        "environment": path_manager.environment,
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    _: None = Depends(rate_limit_check)
    ):
    """
    Predict whether a news article is fake or real using blue-green deployment routing
    - **text**: The news article text to analyze
    - **returns**: Prediction result with confidence score
    """
    start_time = time.time()
    client_ip = http_request.client.host
    user_agent = http_request.headers.get("user-agent")

    try:
        # Check model health
        if model_manager.health_status != "healthy":
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )

        # Prepare request data for routing
        request_data = {
            'client_id': client_ip,
            'user_agent': user_agent,
            'timestamp': datetime.now().isoformat()
        }
        
        # Use traffic router if available, otherwise fallback to model manager
        if traffic_router and (traffic_router.blue_model or traffic_router.green_model):
            try:
                environment, result = traffic_router.make_prediction(request.text, request_data)
                
                # Extract results from traffic router response
                label = result['prediction']
                confidence = result['confidence']
                processing_time = result['processing_time']
                
                logger.debug(f"Used {environment} environment for prediction")
                
            except Exception as e:
                logger.warning(f"Traffic router failed, falling back to model manager: {e}")
                # Fallback to original model manager
                label, confidence = model_manager.predict(request.text)
                processing_time = time.time() - start_time
                environment = "blue"  # Default fallback
        else:
            # Fallback to original model manager
            label, confidence = model_manager.predict(request.text)
            processing_time = time.time() - start_time
            environment = "blue"  # Default when no traffic router

        # Record prediction for monitoring
        prediction_monitor.record_prediction(
            prediction=label,
            confidence=confidence,
            processing_time=processing_time,
            text=request.text,
            model_version=model_manager.model_metadata.get('model_version', 'unknown'),
            client_id=client_ip,
            user_agent=user_agent
        )

        # Record API request metrics
        metrics_collector.record_api_request(
            endpoint="/predict",
            method="POST",
            response_time=processing_time,
            status_code=200,
            client_ip=client_ip
        )

        # Create response
        response = PredictionResponse(
            prediction=label,
            confidence=confidence,
            model_version=model_manager.model_metadata.get('model_version', 'unknown'),
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )

        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            request.text,
            label,
            confidence,
            client_ip,
            processing_time
        )

        return response

    except HTTPException:
        # Record error for failed requests
        processing_time = time.time() - start_time
        prediction_monitor.record_error(
            error_type="http_error",
            error_message="Service unavailable",
            context={"status_code": 503}
        )
        metrics_collector.record_api_request(
            endpoint="/predict",
            method="POST",
            response_time=processing_time,
            status_code=503,
            client_ip=client_ip
        )
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Record error
        prediction_monitor.record_error(
            error_type="prediction_error",
            error_message=str(e),
            context={"text_length": len(request.text)}
        )
        
        metrics_collector.record_api_request(
            endpoint="/predict",
            method="POST",
            response_time=processing_time,
            status_code=500,
            client_ip=client_ip
        )
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    _: None = Depends(rate_limit_check)
):
    """
    Predict multiple news articles in batch
    - **texts**: List of news article texts to analyze
    - **returns**: List of prediction results
    """
    start_time = time.time()

    try:
        # Check model health
        if model_manager.health_status != "healthy":
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )

        predictions = []

        for text in request.texts:
            try:
                label, confidence = model_manager.predict(text)

                prediction = PredictionResponse(
                    prediction=label,
                    confidence=confidence,
                    model_version=model_manager.model_metadata.get(
                        'model_version', 'unknown'),
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0  # Will be updated with total time
                )

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Batch prediction failed for text: {e}")
                # Continue with other texts
                continue

        # Calculate total processing time
        total_processing_time = time.time() - start_time

        # Update processing time for all predictions
        for prediction in predictions:
            prediction.processing_time = total_processing_time / \
                len(predictions)

        response = BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time=total_processing_time
        )

        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction,
            len(request.texts),
            len(predictions),
            http_request.client.host,
            total_processing_time
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    - **returns**: Detailed health status of the API and model
    """
    try:
        # Model health
        model_health = model_manager.health_check()

        # System health
        import psutil
        system_health = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - psutil.boot_time()
        }

        # API health
        api_health = {
            "rate_limit_active": len(rate_limit_storage) > 0,
            "active_connections": len(rate_limit_storage)
        }

        # Environment info
        environment_info = path_manager.get_environment_info()

        # Overall status
        overall_status = "healthy" if model_health["status"] == "healthy" else "unhealthy"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            model_health=model_health,
            system_health=system_health,
            api_health=api_health,
            environment_info=environment_info
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_health={"status": "unhealthy", "error": str(e)},
            system_health={"error": str(e)},
            api_health={"error": str(e)},
            environment_info={"error": str(e)}
        )


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint with comprehensive CV results
    - **returns**: Detailed health status including cross-validation metrics
    """
    try:
        # Get basic health information
        basic_health = await health_check()
        
        # Load metadata to get CV results
        metadata_path = path_manager.get_metadata_path()
        cv_details = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract cross-validation information
                cv_info = metadata.get('cross_validation', {})
                if cv_info:
                    cv_details = {
                        'cross_validation_available': True,
                        'n_splits': cv_info.get('n_splits', 'Unknown'),
                        'test_scores': cv_info.get('test_scores', {}),
                        'train_scores': cv_info.get('train_scores', {}),
                        'overfitting_score': cv_info.get('overfitting_score', 'Unknown'),
                        'stability_score': cv_info.get('stability_score', 'Unknown'),
                        'individual_fold_results': cv_info.get('individual_fold_results', [])
                    }
                    
                    # Add summary statistics
                    test_scores = cv_info.get('test_scores', {})
                    if 'f1' in test_scores:
                        cv_details['cv_f1_summary'] = {
                            'mean': test_scores['f1'].get('mean', 'Unknown'),
                            'std': test_scores['f1'].get('std', 'Unknown'),
                            'min': test_scores['f1'].get('min', 'Unknown'),
                            'max': test_scores['f1'].get('max', 'Unknown'),
                            'scores': test_scores['f1'].get('scores', [])
                        }
                    
                    if 'accuracy' in test_scores:
                        cv_details['cv_accuracy_summary'] = {
                            'mean': test_scores['accuracy'].get('mean', 'Unknown'),
                            'std': test_scores['accuracy'].get('std', 'Unknown'),
                            'min': test_scores['accuracy'].get('min', 'Unknown'),
                            'max': test_scores['accuracy'].get('max', 'Unknown'),
                            'scores': test_scores['accuracy'].get('scores', [])
                        }
                
                # Add model comparison results if available
                statistical_validation = metadata.get('statistical_validation', {})
                if statistical_validation:
                    cv_details['statistical_validation'] = statistical_validation
                
                promotion_validation = metadata.get('promotion_validation', {})
                if promotion_validation:
                    cv_details['promotion_validation'] = promotion_validation
                
                # Add model version and training info
                cv_details['model_info'] = {
                    'model_version': metadata.get('model_version', 'Unknown'),
                    'model_type': metadata.get('model_type', 'Unknown'),
                    'training_timestamp': metadata.get('timestamp', 'Unknown'),
                    'promotion_timestamp': metadata.get('promotion_timestamp'),
                    'cv_f1_mean': metadata.get('cv_f1_mean'),
                    'cv_f1_std': metadata.get('cv_f1_std'),
                    'cv_accuracy_mean': metadata.get('cv_accuracy_mean'),
                    'cv_accuracy_std': metadata.get('cv_accuracy_std')
                }
            
            except Exception as e:
                cv_details = {
                    'cross_validation_available': False,
                    'error': f"Failed to load CV details: {str(e)}"
                }
        else:
            cv_details = {
                'cross_validation_available': False,
                'error': "No metadata file found"
            }
        
        # Combine basic health with detailed CV information
        detailed_response = {
            'basic_health': basic_health,
            'cross_validation_details': cv_details,
            'detailed_check_timestamp': datetime.now().isoformat()
        }
        
        return detailed_response
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            'basic_health': {'status': 'unhealthy', 'error': str(e)},
            'cross_validation_details': {
                'cross_validation_available': False,
                'error': f"Detailed health check failed: {str(e)}"
            },
            'detailed_check_timestamp': datetime.now().isoformat()
        }


@app.get("/cv/results")
async def get_cv_results():
    """
    Get detailed cross-validation results for the current model
    - **returns**: Comprehensive CV metrics and fold-by-fold results
    """
    try:
        metadata_path = path_manager.get_metadata_path()
        
        if not metadata_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Model metadata not found. Train a model first."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        cv_info = metadata.get('cross_validation', {})
        
        if not cv_info:
            raise HTTPException(
                status_code=404,
                detail="No cross-validation results found. Model may not have been trained with CV."
            )
        
        # Structure the CV results for API response
        cv_response = {
            'model_version': metadata.get('model_version', 'Unknown'),
            'model_type': metadata.get('model_type', 'Unknown'),
            'training_timestamp': metadata.get('timestamp', 'Unknown'),
            'cross_validation': {
                'methodology': {
                    'n_splits': cv_info.get('n_splits', 'Unknown'),
                    'cv_type': 'StratifiedKFold',
                    'random_state': 42
                },
                'test_scores': cv_info.get('test_scores', {}),
                'train_scores': cv_info.get('train_scores', {}),
                'performance_indicators': {
                    'overfitting_score': cv_info.get('overfitting_score', 'Unknown'),
                    'stability_score': cv_info.get('stability_score', 'Unknown')
                },
                'individual_fold_results': cv_info.get('individual_fold_results', [])
            },
            'statistical_validation': metadata.get('statistical_validation', {}),
            'promotion_validation': metadata.get('promotion_validation', {}),
            'summary_statistics': {
                'cv_f1_mean': metadata.get('cv_f1_mean'),
                'cv_f1_std': metadata.get('cv_f1_std'),
                'cv_accuracy_mean': metadata.get('cv_accuracy_mean'),
                'cv_accuracy_std': metadata.get('cv_accuracy_std')
            }
        }
        
        return cv_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV results retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve CV results: {str(e)}"
        )


@app.get("/cv/comparison")
async def get_model_comparison_results():
    """
    Get latest model comparison results from retraining
    - **returns**: Statistical comparison results between models
    """
    try:
        # Load comparison logs
        comparison_log_path = path_manager.get_logs_path("model_comparison.json")
        
        if not comparison_log_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No model comparison results found."
            )
        
        with open(comparison_log_path, 'r') as f:
            comparison_logs = json.load(f)
        
        if not comparison_logs:
            raise HTTPException(
                status_code=404,
                detail="No comparison entries found."
            )
        
        # Get the most recent comparison
        latest_comparison = comparison_logs[-1]
        comparison_details = latest_comparison.get('comparison_details', {})
        
        # Structure the response
        comparison_response = {
            'comparison_timestamp': latest_comparison.get('timestamp', 'Unknown'),
            'session_id': latest_comparison.get('session_id', 'Unknown'),
            'models_compared': {
                'model1_name': comparison_details.get('model1_name', 'Production'),
                'model2_name': comparison_details.get('model2_name', 'Candidate')
            },
            'cv_methodology': {
                'cv_folds': comparison_details.get('cv_folds', 'Unknown')
            },
            'model_performance': {
                'production_model': comparison_details.get('model1_cv_results', {}),
                'candidate_model': comparison_details.get('model2_cv_results', {})
            },
            'metric_comparisons': comparison_details.get('metric_comparisons', {}),
            'statistical_tests': comparison_details.get('statistical_tests', {}),
            'promotion_decision': comparison_details.get('promotion_decision', {}),
            'summary': {
                'decision': comparison_details.get('promotion_decision', {}).get('promote_candidate', False),
                'reason': comparison_details.get('promotion_decision', {}).get('reason', 'Unknown'),
                'confidence': comparison_details.get('promotion_decision', {}).get('confidence', 0)
            }
        }
        
        return comparison_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison results retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model comparison results: {str(e)}"
        )
        

@app.get("/metrics")
async def get_metrics():
    """
    Get comprehensive API metrics including CV results
    - **returns**: Usage statistics, performance metrics, and CV information
    """
    try:
        # Calculate metrics from rate limiting storage
        total_requests = sum(len(requests)
                             for requests in rate_limit_storage.values())
        unique_clients = len(rate_limit_storage)

        # Load metadata for CV information
        metadata_path = path_manager.get_metadata_path()
        cv_summary = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract CV summary
                cv_info = metadata.get('cross_validation', {})
                if cv_info:
                    test_scores = cv_info.get('test_scores', {})
                    cv_summary = {
                        'cv_available': True,
                        'cv_folds': cv_info.get('n_splits', 'Unknown'),
                        'cv_f1_mean': test_scores.get('f1', {}).get('mean'),
                        'cv_f1_std': test_scores.get('f1', {}).get('std'),
                        'cv_accuracy_mean': test_scores.get('accuracy', {}).get('mean'),
                        'cv_accuracy_std': test_scores.get('accuracy', {}).get('std'),
                        'overfitting_score': cv_info.get('overfitting_score'),
                        'stability_score': cv_info.get('stability_score')
                    }
                else:
                    cv_summary = {'cv_available': False}
                    
            except Exception as e:
                cv_summary = {'cv_available': False, 'cv_error': str(e)}
        else:
            cv_summary = {'cv_available': False, 'cv_error': 'No metadata file'}

        metrics = {
            'api_metrics': {
                'total_requests': total_requests,
                'unique_clients': unique_clients,
                'timestamp': datetime.now().isoformat()
            },
            'model_info': {
                'model_version': model_manager.model_metadata.get('model_version', 'unknown'),
                'model_health': model_manager.health_status,
                'last_health_check': model_manager.last_health_check.isoformat() if model_manager.last_health_check else None
            },
            'cross_validation_summary': cv_summary,
            'environment_info': {
                'environment': path_manager.environment,
                'available_datasets': path_manager.list_available_datasets(),
                'available_models': path_manager.list_available_models()
            }
        }

        return metrics

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics retrieval failed: {str(e)}"
        )

@app.get("/validation/statistics")
async def get_validation_statistics():
    """Get comprehensive validation statistics"""
    try:
        stats = get_validation_stats()
        
        if not stats:
            return {
                'statistics_available': False,
                'message': 'No validation statistics available yet',
                'timestamp': datetime.now().isoformat()
            }
        
        enhanced_stats = {
            'statistics_available': True,
            'last_updated': stats.get('last_updated'),
            'overall_metrics': {
                'total_validations': stats.get('total_validations', 0),
                'total_articles_processed': stats.get('total_articles', 0),
                'overall_success_rate': (stats.get('total_valid_articles', 0) / 
                                       max(stats.get('total_articles', 1), 1)),
                'average_quality_score': stats.get('average_quality_score', 0.0)
            },
            'source_breakdown': stats.get('source_statistics', {}),
            'recent_performance': {
                'validation_history': stats.get('validation_history', [])[-10:],
                'quality_trends': stats.get('quality_trends', [])[-10:]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_stats
        
    except Exception as e:
        logger.error(f"Failed to get validation statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve validation statistics: {str(e)}"
        )

@app.get("/validation/quality-report")
async def get_quality_report():
    """Get comprehensive data quality report"""
    try:
        report = generate_quality_report()
        
        if 'error' in report:
            raise HTTPException(
                status_code=404,
                detail=report['error']
            )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quality report: {str(e)}"
        )

@app.get("/validation/health")
async def get_validation_health():
    """Get validation system health status"""
    try:
        stats = get_validation_stats()
        
        health_indicators = {
            'validation_system_active': True,
            'statistics_available': bool(stats),
            'recent_activity': False,
            'quality_status': 'unknown'
        }
        
        if stats:
            last_updated = stats.get('last_updated')
            if last_updated:
                try:
                    last_update_time = datetime.fromisoformat(last_updated)
                    hours_since_update = (datetime.now() - last_update_time).total_seconds() / 3600
                    health_indicators['recent_activity'] = hours_since_update <= 24
                    health_indicators['hours_since_last_validation'] = hours_since_update
                except:
                    pass
            
            avg_quality = stats.get('average_quality_score', 0)
            success_rate = stats.get('total_valid_articles', 0) / max(stats.get('total_articles', 1), 1)
            
            if avg_quality >= 0.7 and success_rate >= 0.8:
                health_indicators['quality_status'] = 'excellent'
            elif avg_quality >= 0.5 and success_rate >= 0.6:
                health_indicators['quality_status'] = 'good'
            elif avg_quality >= 0.3 and success_rate >= 0.4:
                health_indicators['quality_status'] = 'fair'
            else:
                health_indicators['quality_status'] = 'poor'
            
            health_indicators['average_quality_score'] = avg_quality
            health_indicators['validation_success_rate'] = success_rate
        
        overall_healthy = (
            health_indicators['validation_system_active'] and
            health_indicators['statistics_available'] and
            health_indicators['quality_status'] not in ['poor', 'unknown']
        )
        
        return {
            'validation_health': {
                'overall_status': 'healthy' if overall_healthy else 'degraded',
                'health_indicators': health_indicators,
                'last_check': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Validation health check failed: {e}")
        return {
            'validation_health': {
                'overall_status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
        }


# New monitoring endpoints
@app.get("/monitor/metrics/current")
async def get_current_metrics():
    """Get current real-time metrics"""
    try:
        prediction_metrics = prediction_monitor.get_current_metrics()
        system_metrics = metrics_collector.collect_system_metrics()
        api_metrics = metrics_collector.collect_api_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "prediction_metrics": asdict(prediction_metrics),
            "system_metrics": asdict(system_metrics),
            "api_metrics": asdict(api_metrics)
        }
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitor/metrics/historical")
async def get_historical_metrics(hours: int = 24):
    """Get historical metrics"""
    try:
        return {
            "prediction_metrics": [asdict(m) for m in prediction_monitor.get_historical_metrics(hours)],
            "aggregated_metrics": metrics_collector.get_aggregated_metrics(hours)
        }
    except Exception as e:
        logger.error(f"Failed to get historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitor/alerts")
async def get_alerts():
    """Get active alerts and statistics"""
    try:
        return {
            "active_alerts": [asdict(alert) for alert in alert_system.get_active_alerts()],
            "alert_statistics": alert_system.get_alert_statistics()
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitor/health")
async def get_monitoring_health():
    """Get monitoring system health"""
    try:
        dashboard_data = metrics_collector.get_real_time_dashboard_data()
        confidence_analysis = prediction_monitor.get_confidence_analysis()
        
        return {
            "monitoring_status": "active",
            "dashboard_data": dashboard_data,
            "confidence_analysis": confidence_analysis,
            "total_predictions": prediction_monitor.total_predictions
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitor/patterns")
async def get_prediction_patterns(hours: int = 24):
    """Get prediction patterns and anomaly analysis"""
    try:
        return prediction_monitor.get_prediction_patterns(hours)
    except Exception as e:
        logger.error(f"Failed to get prediction patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    try:
        success = alert_system.acknowledge_alert(alert_id, "api_user")
        if success:
            return {"message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolution_note: str = ""):
    """Resolve an alert"""
    try:
        success = alert_system.resolve_alert(alert_id, "api_user", resolution_note)
        if success:
            return {"message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automation/status")
async def get_automation_status():
    """Get automation system status"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        # Get automation status
        automation_status = automation_manager.get_automation_status()
        
        # Get drift monitoring status
        drift_status = automation_manager.drift_monitor.get_automation_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "automation_system": automation_status,
            "drift_monitoring": drift_status,
            "system_health": "active" if automation_manager.retraining_active else "inactive"
        }
        
    except Exception as e:
        logger.error(f"Failed to get automation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automation/triggers/check")
async def check_retraining_triggers():
    """Check current retraining triggers"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        trigger_results = automation_manager.drift_monitor.check_retraining_triggers()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "trigger_evaluation": trigger_results,
            "recommendation": "Retraining recommended" if trigger_results.get('should_retrain') else "No retraining needed"
        }
        
    except Exception as e:
        logger.error(f"Failed to check triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automation/retrain/trigger")
async def trigger_manual_retraining(reason: str = "manual_api_trigger"):
    """Manually trigger retraining"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        result = automation_manager.trigger_manual_retraining(reason)
        
        if result['success']:
            return {
                "message": "Retraining triggered successfully",
                "timestamp": datetime.now().isoformat(),
                "reason": reason
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automation/queue")
async def get_retraining_queue():
    """Get current retraining queue"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        queue = automation_manager.load_retraining_queue()
        recent_logs = automation_manager.get_recent_automation_logs(hours=24)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "queued_jobs": queue,
            "recent_activity": recent_logs,
            "queue_length": len(queue)
        }
        
    except Exception as e:
        logger.error(f"Failed to get retraining queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automation/drift/status")
async def get_drift_monitoring_status():
    """Get drift monitoring status"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        # Get recent drift results
        drift_logs = automation_manager.get_recent_automation_logs(hours=48)
        drift_checks = [log for log in drift_logs if 'drift' in log.get('event', '')]
        
        # Get current drift status
        drift_status = automation_manager.drift_monitor.get_automation_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "drift_monitoring_active": True,
            "recent_drift_checks": drift_checks[-10:],  # Last 10 checks
            "drift_status": drift_status
        }
        
    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automation/settings/update")
async def update_automation_settings(settings: Dict[str, Any]):
    """Update automation settings"""
    try:
        if automation_manager is None:
            raise HTTPException(status_code=503, detail="Automation system not available")
        
        # Update settings
        automation_manager.automation_config.update(settings)
        automation_manager.save_automation_config()
        
        return {
            "message": "Automation settings updated",
            "timestamp": datetime.now().isoformat(),
            "updated_settings": settings
        }
        
    except Exception as e:
        logger.error(f"Failed to update automation settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Deployment endpoints
@app.get("/deployment/status")
async def get_deployment_status():
    """Get deployment system status"""
    try:
        if not deployment_manager:
            raise HTTPException(status_code=503, detail="Deployment system not available")
        
        return deployment_manager.get_deployment_status()
        
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployment/prepare")
async def prepare_deployment(target_version: str, strategy: str = "blue_green"):
    """Prepare a new deployment"""
    try:
        if not deployment_manager:
            raise HTTPException(status_code=503, detail="Deployment system not available")
        
        deployment_id = deployment_manager.prepare_deployment(target_version, strategy)
        
        return {
            "message": "Deployment prepared",
            "deployment_id": deployment_id,
            "target_version": target_version,
            "strategy": strategy
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployment/start/{deployment_id}")
async def start_deployment(deployment_id: str):
    """Start a prepared deployment"""
    try:
        if not deployment_manager:
            raise HTTPException(status_code=503, detail="Deployment system not available")
        
        success = deployment_manager.start_deployment(deployment_id)
        
        if success:
            return {"message": "Deployment started successfully", "deployment_id": deployment_id}
        else:
            raise HTTPException(status_code=500, detail="Deployment failed to start")
            
    except Exception as e:
        logger.error(f"Failed to start deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployment/rollback")
async def rollback_deployment(reason: str = "Manual rollback"):
    """Rollback current deployment"""
    try:
        if not deployment_manager:
            raise HTTPException(status_code=503, detail="Deployment system not available")
        
        success = deployment_manager.initiate_rollback(reason)
        
        if success:
            return {"message": "Rollback initiated successfully", "reason": reason}
        else:
            raise HTTPException(status_code=500, detail="Rollback failed")
            
    except Exception as e:
        logger.error(f"Failed to rollback deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deployment/traffic")
async def get_traffic_status():
    """Get traffic routing status"""
    try:
        if not traffic_router:
            raise HTTPException(status_code=503, detail="Traffic router not available")
        
        return traffic_router.get_routing_status()
        
    except Exception as e:
        logger.error(f"Failed to get traffic status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployment/traffic/weights")
async def set_traffic_weights(blue_weight: int, green_weight: int):
    """Set traffic routing weights"""
    try:
        if not traffic_router:
            raise HTTPException(status_code=503, detail="Traffic router not available")
        
        success = traffic_router.set_routing_weights(blue_weight, green_weight)
        
        if success:
            return {
                "message": "Traffic weights updated",
                "blue_weight": blue_weight,
                "green_weight": green_weight
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update traffic weights")
            
    except Exception as e:
        logger.error(f"Failed to set traffic weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deployment/performance")
async def get_deployment_performance(window_minutes: int = 60):
    """Get deployment performance comparison"""
    try:
        if not traffic_router:
            raise HTTPException(status_code=503, detail="Traffic router not available")
        
        return traffic_router.compare_environment_performance(window_minutes)
        
    except Exception as e:
        logger.error(f"Failed to get deployment performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/registry/models")
async def list_registry_models(status: str = None, limit: int = 10):
    """List models in registry"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not available")
        
        models = model_registry.list_models(status=status, limit=limit)
        return {"models": [asdict(model) for model in models]}
        
    except Exception as e:
        logger.error(f"Failed to list registry models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/registry/stats")
async def get_registry_stats():
    """Get model registry statistics"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not available")
        
        return model_registry.get_registry_stats()
        
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))