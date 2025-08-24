import os
import sys
from pathlib import Path
from typing import Dict, Optional
import logging

# Get logger, but handle case where it might not be configured yet
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # If no handlers, add a basic one
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
except:
    # Fallback - create a simple logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class EnvironmentPathManager:
    """Dynamic path management for different deployment environments"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.base_paths = self._configure_paths()
        self._ensure_directories()
        
    def _detect_environment(self) -> str:
        """Detect the current deployment environment"""
        # Check for HuggingFace Spaces
        if os.environ.get('SPACE_ID') or os.path.exists('/app/app.py') or os.path.exists('/app/streamlit_app.py'):
            return 'huggingface_spaces'
        
        # Check for Docker container
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            return 'docker'
        
        # Check if running from /app directory (likely container)
        if str(Path.cwd()).startswith('/app'):
            return 'container'
            
        # Default to local development
        return 'local'
    
    def _configure_paths(self) -> Dict[str, Path]:
        """Configure paths based on environment"""
        if self.environment == 'huggingface_spaces':
            # HuggingFace Spaces: Use /app structure
            base_dir = Path('/app')
            return {
                'base': base_dir,
                'data': base_dir / 'data',
                'model': base_dir / 'model', 
                'logs': base_dir / 'logs',
                'cache': base_dir / 'cache',
                'temp': base_dir / 'temp'
            }
            
        elif self.environment in ['docker', 'container']:
            # Docker/Container: Use /app structure with /tmp for temporary files
            base_dir = Path('/app')
            return {
                'base': base_dir,
                'data': base_dir / 'data',
                'model': base_dir / 'model',
                'logs': base_dir / 'logs', 
                'cache': Path('/tmp/cache'),
                'temp': Path('/tmp/temp')
            }
            
        else:
            # Local development: Use project structure
            # Find project root (where this file is located)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent
            
            # Navigate up to find the actual project root
            while project_root.parent != project_root:
                if (project_root / 'requirements.txt').exists():
                    break
                project_root = project_root.parent
            
            return {
                'base': project_root,
                'data': project_root / 'data',
                'model': project_root / 'model',
                'logs': project_root / 'logs',
                'cache': project_root / 'cache',
                'temp': project_root / 'temp'
            }
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist with proper error handling"""
        for path_name, path in self.base_paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = path / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                    # Use print for critical startup messages to avoid logging dependency issues
                    if self.environment == 'huggingface_spaces':
                        print(f"Directory {path} created with write access")
                except (PermissionError, OSError):
                    print(f"Directory exists but no write access: {path}")
                    
            except PermissionError:
                print(f"Cannot create directory {path}, using fallback")
                if path_name in ['cache', 'temp']:
                    # Fallback to user's home directory for cache/temp
                    fallback_path = Path.home() / f'.fake_news_detector/{path_name}'
                    try:
                        fallback_path.mkdir(parents=True, exist_ok=True)
                        self.base_paths[path_name] = fallback_path
                        print(f"Using fallback directory: {fallback_path}")
                    except Exception as e:
                        print(f"Fallback directory creation failed: {e}")
                elif path_name == 'logs':
                    # For logs, try using a temporary directory
                    try:
                        import tempfile
                        temp_logs = Path(tempfile.mkdtemp(prefix='logs_'))
                        self.base_paths[path_name] = temp_logs
                        print(f"Using temporary logs directory: {temp_logs}")
                    except Exception as e:
                        print(f"Cannot create temporary logs directory: {e}")
            except Exception as e:
                print(f"Failed to create directory {path}: {e}")
    
    def get_data_path(self, filename: str = '') -> Path:
        """Get data directory path"""
        return self.base_paths['data'] / filename if filename else self.base_paths['data']
    
    def get_model_path(self, filename: str = '') -> Path:
        """Get model directory path"""
        return self.base_paths['model'] / filename if filename else self.base_paths['model']
    
    def get_logs_path(self, filename: str = '') -> Path:
        """Get logs directory path"""  
        return self.base_paths['logs'] / filename if filename else self.base_paths['logs']
    
    def get_cache_path(self, filename: str = '') -> Path:
        """Get cache directory path"""
        return self.base_paths['cache'] / filename if filename else self.base_paths['cache']
    
    def get_temp_path(self, filename: str = '') -> Path:
        """Get temporary directory path"""
        return self.base_paths['temp'] / filename if filename else self.base_paths['temp']
    
    def get_activity_log_path(self) -> Path:
        """Get activity log file path"""
        return self.get_logs_path('activity_log.json')
    
    def get_metadata_path(self) -> Path:
        """Get model metadata file path"""
        return self.get_model_path('metadata.json')
    
    def get_combined_dataset_path(self) -> Path:
        """Get combined dataset path"""
        return self.get_data_path('combined_dataset.csv')
    
    def get_scraped_data_path(self) -> Path:
        """Get scraped data path"""
        return self.get_data_path('scraped_real.csv')
    
    def get_generated_data_path(self) -> Path:
        """Get generated fake data path"""
        return self.get_data_path('generated_fake.csv')
    
    def get_model_file_path(self) -> Path:
        """Get main model file path"""
        return self.get_model_path('model.pkl')
    
    def get_vectorizer_path(self) -> Path:
        """Get vectorizer file path"""
        return self.get_model_path('vectorizer.pkl')
    
    def get_pipeline_path(self) -> Path:
        """Get pipeline file path"""
        return self.get_model_path('pipeline.pkl')
    
    def get_candidate_model_path(self) -> Path:
        """Get candidate model file path"""
        return self.get_model_path('model_candidate.pkl')
    
    def get_candidate_vectorizer_path(self) -> Path:
        """Get candidate vectorizer file path"""
        return self.get_model_path('vectorizer_candidate.pkl')
    
    def get_candidate_pipeline_path(self) -> Path:
        """Get candidate pipeline file path"""
        return self.get_model_path('pipeline_candidate.pkl')
    
    def list_available_datasets(self) -> Dict[str, bool]:
        """List available datasets and their existence status"""
        datasets = {
            'combined_dataset.csv': self.get_combined_dataset_path().exists(),
            'scraped_real.csv': self.get_scraped_data_path().exists(), 
            'generated_fake.csv': self.get_generated_data_path().exists(),
            'kaggle/Fake.csv': (self.get_data_path() / 'kaggle' / 'Fake.csv').exists(),
            'kaggle/True.csv': (self.get_data_path() / 'kaggle' / 'True.csv').exists(),
        }
        return datasets
    
    def list_available_models(self) -> Dict[str, bool]:
        """List available models and their existence status"""
        models = {
            'model.pkl': self.get_model_file_path().exists(),
            'vectorizer.pkl': self.get_vectorizer_path().exists(),
            'pipeline.pkl': self.get_pipeline_path().exists(),
            'model_candidate.pkl': self.get_candidate_model_path().exists(),
            'vectorizer_candidate.pkl': self.get_candidate_vectorizer_path().exists(),
            'pipeline_candidate.pkl': self.get_candidate_pipeline_path().exists(),
            'metadata.json': self.get_metadata_path().exists()
        }
        return models
    
    def get_environment_info(self) -> Dict:
        """Get comprehensive environment information"""
        return {
            'environment': self.environment,
            'base_dir': str(self.base_paths['base']),
            'data_dir': str(self.base_paths['data']),
            'model_dir': str(self.base_paths['model']),
            'logs_dir': str(self.base_paths['logs']),
            'available_datasets': self.list_available_datasets(),
            'available_models': self.list_available_models(),
            'current_working_directory': str(Path.cwd()),
            'python_path': sys.path[0],
            'space_id': os.environ.get('SPACE_ID', 'Not HF Spaces'),
            'docker_env': os.path.exists('/.dockerenv')
        }
    
    def log_environment_info(self):
        """Log detailed environment information"""
        info = self.get_environment_info()
        logger.info(f"🌍 Environment: {info['environment']}")
        logger.info(f"📁 Base directory: {info['base_dir']}")
        logger.info(f"📊 Data directory: {info['data_dir']}")
        logger.info(f"🤖 Model directory: {info['model_dir']}")
        logger.info(f"📝 Logs directory: {info['logs_dir']}")
        
        # Log available files
        datasets = info['available_datasets']
        models = info['available_models']
        
        logger.info(f"📈 Available datasets: {sum(datasets.values())}/{len(datasets)}")
        for name, exists in datasets.items():
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {name}")
            
        logger.info(f"🎯 Available models: {sum(models.values())}/{len(models)}")
        for name, exists in models.items():
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {name}")

# Global instance
path_manager = EnvironmentPathManager()

# Convenience functions for backward compatibility
def get_data_path(filename: str = '') -> Path:
    return path_manager.get_data_path(filename)

def get_model_path(filename: str = '') -> Path:
    return path_manager.get_model_path(filename)

def get_logs_path(filename: str = '') -> Path:
    return path_manager.get_logs_path(filename)

def get_environment_info() -> Dict:
    return path_manager.get_environment_info()

def log_environment_info():
    path_manager.log_environment_info()

# For debugging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_environment_info()