import json
import joblib
import logging
import hashlib
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    STAGED = "staged"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    version_id: str
    name: str
    description: str
    created_at: str
    created_by: str
    status: str
    
    # Model files
    model_path: str
    vectorizer_path: str
    pipeline_path: Optional[str]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    cross_validation_results: Dict[str, Any]
    
    # Training details
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    feature_info: Dict[str, Any]
    
    # Deployment info
    deployment_history: List[Dict[str, Any]]
    performance_history: List[Dict[str, Any]]
    
    # Model signature
    model_signature: str
    dependencies: Dict[str, str]
    
    # Tags and labels
    tags: List[str]
    labels: Dict[str, str]

class ModelRegistry:
    """Central registry for managing model versions and metadata"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/tmp")
        self.setup_registry_paths()
        self.setup_registry_config()
        
        # Model storage
        self.models = {}  # version_id -> ModelMetadata
        self.load_registry()
    
    def setup_registry_paths(self):
        """Setup model registry paths"""
        self.registry_dir = self.base_dir / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry files
        self.registry_index_path = self.registry_dir / "model_index.json"
        self.registry_metadata_path = self.registry_dir / "registry_metadata.json"
        self.registry_log_path = self.registry_dir / "registry_log.json"
        
        # Model storage directory
        self.models_storage_dir = self.registry_dir / "models"
        self.models_storage_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_registry_config(self):
        """Setup registry configuration"""
        self.registry_config = {
            'max_versions_per_model': 10,
            'auto_cleanup_enabled': True,
            'cleanup_after_days': 30,
            'backup_enabled': True,
            'backup_interval_hours': 24,
            'validation_required': True,
            'signature_verification': True
        }
    
    def register_model(self, model_path: str, vectorizer_path: str, 
                      metadata: Dict[str, Any], version_id: str = None) -> str:
        """Register a new model version"""
        try:
            # Generate version ID if not provided
            if not version_id:
                version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate model files exist
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not Path(vectorizer_path).exists():
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            
            # Create model storage directory
            model_storage_dir = self.models_storage_dir / version_id
            model_storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files to registry storage
            import shutil
            registry_model_path = model_storage_dir / "model.pkl"
            registry_vectorizer_path = model_storage_dir / "vectorizer.pkl"
            
            shutil.copy2(model_path, registry_model_path)
            shutil.copy2(vectorizer_path, registry_vectorizer_path)
            
            # Generate model signature
            model_signature = self.generate_model_signature(registry_model_path, registry_vectorizer_path)
            
            # Create comprehensive metadata
            model_metadata = ModelMetadata(
                version_id=version_id,
                name=metadata.get('name', f'model_{version_id}'),
                description=metadata.get('description', 'No description provided'),
                created_at=datetime.now().isoformat(),
                created_by=metadata.get('created_by', 'system'),
                status=ModelStatus.VALIDATING.value,
                
                # File paths
                model_path=str(registry_model_path),
                vectorizer_path=str(registry_vectorizer_path),
                pipeline_path=metadata.get('pipeline_path'),
                
                # Performance metrics
                training_metrics=metadata.get('training_metrics', {}),
                validation_metrics=metadata.get('validation_metrics', {}),
                cross_validation_results=metadata.get('cross_validation_results', {}),
                
                # Training details
                training_config=metadata.get('training_config', {}),
                dataset_info=metadata.get('dataset_info', {}),
                feature_info=metadata.get('feature_info', {}),
                
                # Deployment info
                deployment_history=[],
                performance_history=[],
                
                # Model signature
                model_signature=model_signature,
                dependencies=metadata.get('dependencies', {}),
                
                # Tags and labels
                tags=metadata.get('tags', []),
                labels=metadata.get('labels', {})
            )
            
            # Validate model if required
            if self.registry_config['validation_required']:
                validation_result = self.validate_model(model_metadata)
                if not validation_result['valid']:
                    model_metadata.status = ModelStatus.FAILED.value
                    self.log_registry_event("model_validation_failed", 
                                           f"Model validation failed: {validation_result['errors']}")
                else:
                    model_metadata.status = ModelStatus.STAGED.value
            else:
                model_metadata.status = ModelStatus.STAGED.value
            
            # Save metadata to file
            metadata_file = model_storage_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(model_metadata), f, indent=2)
            
            # Register in memory
            self.models[version_id] = model_metadata
            
            # Update registry index
            self.update_registry_index()
            
            # Log registration
            self.log_registry_event("model_registered", f"Registered model version {version_id}", {
                'version_id': version_id,
                'model_signature': model_signature,
                'status': model_metadata.status
            })
            
            logger.info(f"Successfully registered model version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise e
    
    def get_model(self, version_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by version ID"""
        return self.models.get(version_id)
    
    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get currently active model"""
        for model in self.models.values():
            if model.status == ModelStatus.ACTIVE.value:
                return model
        return None
    
    def list_models(self, status: str = None, limit: int = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        # Filter by status
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        if limit:
            models = models[:limit]
        
        return models
    
    def promote_model(self, version_id: str) -> bool:
        """Promote a model to active status"""
        try:
            model = self.get_model(version_id)
            if not model:
                raise ValueError(f"Model {version_id} not found")
            
            if model.status != ModelStatus.STAGED.value:
                raise ValueError(f"Model {version_id} is not staged for promotion")
            
            # Demote current active model
            current_active = self.get_active_model()
            if current_active:
                current_active.status = ModelStatus.RETIRED.value
                self.log_registry_event("model_retired", f"Retired model {current_active.version_id}")
            
            # Promote new model
            model.status = ModelStatus.ACTIVE.value
            
            # Record deployment
            deployment_record = {
                'promoted_at': datetime.now().isoformat(),
                'promoted_by': 'system',
                'previous_active': current_active.version_id if current_active else None
            }
            model.deployment_history.append(deployment_record)
            
            # Update registry
            self.update_registry_index()
            self.save_model_metadata(model)
            
            self.log_registry_event("model_promoted", f"Promoted model {version_id} to active", {
                'version_id': version_id,
                'previous_active': current_active.version_id if current_active else None
            })
            
            logger.info(f"Successfully promoted model {version_id} to active")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {version_id}: {e}")
            return False
    
    def retire_model(self, version_id: str) -> bool:
        """Retire a model version"""
        try:
            model = self.get_model(version_id)
            if not model:
                raise ValueError(f"Model {version_id} not found")
            
            old_status = model.status
            model.status = ModelStatus.RETIRED.value
            
            # Update registry
            self.update_registry_index()
            self.save_model_metadata(model)
            
            self.log_registry_event("model_retired", f"Retired model {version_id}", {
                'version_id': version_id,
                'previous_status': old_status
            })
            
            logger.info(f"Successfully retired model {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retire model {version_id}: {e}")
            return False
    
    def delete_model(self, version_id: str, force: bool = False) -> bool:
        """Delete a model version"""
        try:
            model = self.get_model(version_id)
            if not model:
                raise ValueError(f"Model {version_id} not found")
            
            # Prevent deletion of active model unless forced
            if model.status == ModelStatus.ACTIVE.value and not force:
                raise ValueError("Cannot delete active model without force=True")
            
            # Remove from memory
            del self.models[version_id]
            
            # Remove model storage directory
            model_storage_dir = self.models_storage_dir / version_id
            if model_storage_dir.exists():
                import shutil
                shutil.rmtree(model_storage_dir)
            
            # Update registry index
            self.update_registry_index()
            
            self.log_registry_event("model_deleted", f"Deleted model {version_id}", {
                'version_id': version_id,
                'forced': force
            })
            
            logger.info(f"Successfully deleted model {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {version_id}: {e}")
            return False
    
    def validate_model(self, model_metadata: ModelMetadata) -> Dict[str, Any]:
        """Validate a registered model"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if model files exist
            if not Path(model_metadata.model_path).exists():
                validation_result['errors'].append("Model file not found")
                validation_result['valid'] = False
            
            if not Path(model_metadata.vectorizer_path).exists():
                validation_result['errors'].append("Vectorizer file not found")
                validation_result['valid'] = False
            
            # Try to load model
            try:
                model = joblib.load(model_metadata.model_path)
                vectorizer = joblib.load(model_metadata.vectorizer_path)
                
                # Check if model has required methods
                if not hasattr(model, 'predict'):
                    validation_result['errors'].append("Model missing predict method")
                    validation_result['valid'] = False
                
                if not hasattr(vectorizer, 'transform'):
                    validation_result['errors'].append("Vectorizer missing transform method")
                    validation_result['valid'] = False
                
                # Test prediction with dummy data
                try:
                    test_text = ["This is a test article for validation"]
                    X = vectorizer.transform(test_text)
                    prediction = model.predict(X)
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)
                except Exception as e:
                    validation_result['errors'].append(f"Model prediction test failed: {str(e)}")
                    validation_result['valid'] = False
                
            except Exception as e:
                validation_result['errors'].append(f"Failed to load model: {str(e)}")
                validation_result['valid'] = False
            
            # Check performance metrics
            if not model_metadata.training_metrics:
                validation_result['warnings'].append("No training metrics available")
            
            # Verify signature if enabled
            if self.registry_config['signature_verification']:
                current_signature = self.generate_model_signature(
                    model_metadata.model_path, 
                    model_metadata.vectorizer_path
                )
                if current_signature != model_metadata.model_signature:
                    validation_result['errors'].append("Model signature verification failed")
                    validation_result['valid'] = False
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['valid'] = False
        
        return validation_result
    
    def generate_model_signature(self, model_path: str, vectorizer_path: str) -> str:
        """Generate a signature for model files"""
        try:
            hasher = hashlib.sha256()
            
            # Hash model file
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            # Hash vectorizer file
            with open(vectorizer_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate model signature: {e}")
            return ""
    
    def record_performance(self, version_id: str, performance_metrics: Dict[str, float]):
        """Record performance metrics for a model"""
        try:
            model = self.get_model(version_id)
            if not model:
                raise ValueError(f"Model {version_id} not found")
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': performance_metrics
            }
            
            model.performance_history.append(performance_record)
            
            # Keep only last 100 performance records
            if len(model.performance_history) > 100:
                model.performance_history = model.performance_history[-100:]
            
            # Save updated metadata
            self.save_model_metadata(model)
            
            logger.info(f"Recorded performance for model {version_id}")
            
        except Exception as e:
            logger.error(f"Failed to record performance for model {version_id}: {e}")
    
    def get_model_comparison(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            model1 = self.get_model(version_id1)
            model2 = self.get_model(version_id2)
            
            if not model1 or not model2:
                raise ValueError("One or both models not found")
            
            comparison = {
                'model1': {
                    'version_id': model1.version_id,
                    'created_at': model1.created_at,
                    'status': model1.status,
                    'training_metrics': model1.training_metrics,
                    'validation_metrics': model1.validation_metrics
                },
                'model2': {
                    'version_id': model2.version_id,
                    'created_at': model2.created_at,
                    'status': model2.status,
                    'training_metrics': model2.training_metrics,
                    'validation_metrics': model2.validation_metrics
                },
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            # Calculate metric differences
            metric_diffs = {}
            for metric in model1.training_metrics:
                if metric in model2.training_metrics:
                    diff = model2.training_metrics[metric] - model1.training_metrics[metric]
                    metric_diffs[metric] = {
                        'difference': diff,
                        'improvement': diff > 0,
                        'percentage_change': (diff / model1.training_metrics[metric]) * 100 if model1.training_metrics[metric] != 0 else 0
                    }
            
            comparison['metric_differences'] = metric_diffs
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {'error': str(e)}
    
    def cleanup_old_models(self):
        """Clean up old retired models"""
        try:
            if not self.registry_config['auto_cleanup_enabled']:
                return
            
            cleanup_date = datetime.now() - timedelta(days=self.registry_config['cleanup_after_days'])
            
            models_to_cleanup = []
            for model in self.models.values():
                if (model.status == ModelStatus.RETIRED.value and 
                    datetime.fromisoformat(model.created_at) < cleanup_date):
                    models_to_cleanup.append(model.version_id)
            
            for version_id in models_to_cleanup:
                self.delete_model(version_id, force=True)
                logger.info(f"Cleaned up old model: {version_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
    
    def update_registry_index(self):
        """Update the registry index file"""
        try:
            index = {
                'last_updated': datetime.now().isoformat(),
                'total_models': len(self.models),
                'models_by_status': {},
                'model_versions': []
            }
            
            # Count models by status
            for model in self.models.values():
                status = model.status
                index['models_by_status'][status] = index['models_by_status'].get(status, 0) + 1
            
            # Add model summaries
            for model in self.models.values():
                index['model_versions'].append({
                    'version_id': model.version_id,
                    'name': model.name,
                    'status': model.status,
                    'created_at': model.created_at,
                    'signature': model.model_signature
                })
            
            # Save index
            with open(self.registry_index_path, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update registry index: {e}")
    
    def save_model_metadata(self, model: ModelMetadata):
        """Save model metadata to file"""
        try:
            model_storage_dir = self.models_storage_dir / model.version_id
            metadata_file = model_storage_dir / "metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(asdict(model), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def load_registry(self):
        """Load registry from storage"""
        try:
            # Load from individual model metadata files
            if self.models_storage_dir.exists():
                for model_dir in self.models_storage_dir.iterdir():
                    if model_dir.is_dir():
                        metadata_file = model_dir / "metadata.json"
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata_dict = json.load(f)
                                
                                model_metadata = ModelMetadata(**metadata_dict)
                                self.models[model_metadata.version_id] = model_metadata
                                
                            except Exception as e:
                                logger.warning(f"Failed to load model metadata from {metadata_file}: {e}")
            
            logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def log_registry_event(self, event: str, message: str, details: Dict = None):
        """Log registry events"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'message': message,
                'details': details or {}
            }
            
            # Load existing logs
            logs = []
            if self.registry_log_path.exists():
                try:
                    with open(self.registry_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.registry_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log registry event: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            stats = {
                'total_models': len(self.models),
                'models_by_status': {},
                'active_model': None,
                'latest_model': None,
                'storage_info': {},
                'recent_activity': []
            }
            
            # Count by status
            for model in self.models.values():
                status = model.status
                stats['models_by_status'][status] = stats['models_by_status'].get(status, 0) + 1
            
            # Get active model
            active_model = self.get_active_model()
            if active_model:
                stats['active_model'] = {
                    'version_id': active_model.version_id,
                    'created_at': active_model.created_at,
                    'training_metrics': active_model.training_metrics
                }
            
            # Get latest model
            models_by_date = sorted(self.models.values(), key=lambda x: x.created_at, reverse=True)
            if models_by_date:
                latest = models_by_date[0]
                stats['latest_model'] = {
                    'version_id': latest.version_id,
                    'created_at': latest.created_at,
                    'status': latest.status
                }
            
            # Storage information
            if self.models_storage_dir.exists():
                total_size = sum(f.stat().st_size for f in self.models_storage_dir.rglob('*') if f.is_file())
                stats['storage_info'] = {
                    'total_size_mb': total_size / (1024 * 1024),
                    'model_count': len(list(self.models_storage_dir.iterdir()))
                }
            
            # Recent activity
            if self.registry_log_path.exists():
                try:
                    with open(self.registry_log_path, 'r') as f:
                        logs = json.load(f)
                    stats['recent_activity'] = logs[-10:]  # Last 10 events
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {'error': str(e)}