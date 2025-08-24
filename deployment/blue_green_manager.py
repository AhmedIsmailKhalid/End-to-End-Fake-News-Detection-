import json
import time
import logging
import threading
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple


logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    INACTIVE = "inactive"
    PREPARING = "preparing"
    STAGING = "staging"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    COMPLETED = "completed"

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_path: str
    vectorizer_path: str
    metadata_path: str
    created_at: str
    status: str
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]

@dataclass
class DeploymentPlan:
    """Deployment plan configuration"""
    deployment_id: str
    source_version: str
    target_version: str
    strategy: str  # 'blue_green', 'canary', 'rolling'
    traffic_stages: List[Dict[str, Any]]
    health_checks: Dict[str, Any]
    rollback_conditions: Dict[str, Any]
    created_at: str
    status: str

class BlueGreenDeploymentManager:
    """Manages blue-green deployments with traffic routing and health monitoring"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/tmp")
        self.setup_deployment_paths()
        self.setup_deployment_config()
        
        # Current deployment state
        self.current_deployment = None
        self.active_version = None
        self.staging_version = None
        self.traffic_split = {"blue": 100, "green": 0}
        
        # Monitoring
        self.deployment_monitor = None
        self.monitor_thread = None
        self.monitoring_active = False
        
        # Load existing state
        self.load_deployment_state()
    
    def setup_deployment_paths(self):
        """Setup deployment-specific paths"""
        self.deployment_dir = self.base_dir / "deployment"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models_dir = self.deployment_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment logs
        self.deployment_log_path = self.deployment_dir / "deployment_log.json"
        self.deployment_state_path = self.deployment_dir / "deployment_state.json"
        self.traffic_log_path = self.deployment_dir / "traffic_log.json"
        
        # Blue-Green specific directories
        self.blue_dir = self.models_dir / "blue"
        self.green_dir = self.models_dir / "green"
        self.blue_dir.mkdir(parents=True, exist_ok=True)
        self.green_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_deployment_config(self):
        """Setup deployment configuration"""
        self.deployment_config = {
            'traffic_stages': [
                {'percentage': 10, 'duration_minutes': 15, 'success_threshold': 0.95},
                {'percentage': 25, 'duration_minutes': 15, 'success_threshold': 0.95},
                {'percentage': 50, 'duration_minutes': 30, 'success_threshold': 0.95},
                {'percentage': 75, 'duration_minutes': 30, 'success_threshold': 0.95},
                {'percentage': 100, 'duration_minutes': 0, 'success_threshold': 0.95}
            ],
            'health_checks': {
                'response_time_threshold': 5.0,  # seconds
                'error_rate_threshold': 0.05,    # 5%
                'confidence_threshold': 0.6,     # minimum confidence
                'check_interval': 30,            # seconds
                'failure_threshold': 3           # consecutive failures
            },
            'rollback_conditions': {
                'error_rate_spike': 0.15,        # 15% error rate
                'response_time_spike': 10.0,     # 10 seconds
                'confidence_drop': 0.4,          # below 40% confidence
                'health_check_failures': 5       # consecutive failures
            },
            'deployment_timeouts': {
                'stage_timeout_minutes': 60,
                'total_deployment_hours': 6,
                'rollback_timeout_minutes': 15
            }
        }
    
    def create_model_version(self, model_path: str, vectorizer_path: str, 
                           metadata: Dict) -> str:
        """Create a new model version"""
        try:
            version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create version directory
            version_dir = self.models_dir / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files to version directory
            import shutil
            model_dest = version_dir / "model.pkl"
            vectorizer_dest = version_dir / "vectorizer.pkl"
            metadata_dest = version_dir / "metadata.json"
            
            shutil.copy2(model_path, model_dest)
            shutil.copy2(vectorizer_path, vectorizer_dest)
            
            # Save metadata
            version_metadata = {
                **metadata,
                'version_id': version_id,
                'created_at': datetime.now().isoformat(),
                'model_path': str(model_dest),
                'vectorizer_path': str(vectorizer_dest),
                'status': 'created'
            }
            
            with open(metadata_dest, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Create ModelVersion object
            model_version = ModelVersion(
                version_id=version_id,
                model_path=str(model_dest),
                vectorizer_path=str(vectorizer_dest),
                metadata_path=str(metadata_dest),
                created_at=version_metadata['created_at'],
                status='created',
                performance_metrics=metadata.get('performance_metrics', {}),
                deployment_config={}
            )
            
            # Log version creation
            self.log_deployment_event("version_created", f"Created model version {version_id}", {
                'version_id': version_id,
                'performance_metrics': model_version.performance_metrics
            })
            
            logger.info(f"Created model version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise e
    
    def prepare_deployment(self, target_version_id: str, 
                         deployment_strategy: str = "blue_green") -> str:
        """Prepare a new deployment"""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get current active version
            current_version = self.get_active_version()
            
            # Validate target version exists
            if not self.version_exists(target_version_id):
                raise ValueError(f"Target version {target_version_id} does not exist")
            
            # Create deployment plan
            deployment_plan = DeploymentPlan(
                deployment_id=deployment_id,
                source_version=current_version['version_id'] if current_version else None,
                target_version=target_version_id,
                strategy=deployment_strategy,
                traffic_stages=self.deployment_config['traffic_stages'].copy(),
                health_checks=self.deployment_config['health_checks'].copy(),
                rollback_conditions=self.deployment_config['rollback_conditions'].copy(),
                created_at=datetime.now().isoformat(),
                status=DeploymentStatus.PREPARING.value
            )
            
            # Stage the new version
            self.stage_version(target_version_id, deployment_plan)
            
            # Update deployment state
            self.current_deployment = deployment_plan
            self.save_deployment_state()
            
            self.log_deployment_event("deployment_prepared", f"Prepared deployment {deployment_id}", {
                'deployment_plan': asdict(deployment_plan)
            })
            
            logger.info(f"Prepared deployment: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to prepare deployment: {e}")
            raise e
    
    def stage_version(self, version_id: str, deployment_plan: DeploymentPlan):
        """Stage a model version for deployment"""
        try:
            # Determine staging environment (blue or green)
            staging_env = self.determine_staging_environment()
            
            # Copy version to staging directory
            version_dir = self.models_dir / version_id
            staging_dir = self.blue_dir if staging_env == "blue" else self.green_dir
            
            # Clear staging directory
            import shutil
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            for file_name in ["model.pkl", "vectorizer.pkl", "metadata.json"]:
                source_file = version_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, staging_dir / file_name)
            
            # Update staging version
            self.staging_version = {
                'version_id': version_id,
                'environment': staging_env,
                'staged_at': datetime.now().isoformat(),
                'status': 'staged'
            }
            
            # Update deployment status
            deployment_plan.status = DeploymentStatus.STAGING.value
            
            logger.info(f"Staged version {version_id} in {staging_env} environment")
            
        except Exception as e:
            logger.error(f"Failed to stage version: {e}")
            raise e
    
    def start_deployment(self, deployment_id: str) -> bool:
        """Start the deployment process"""
        try:
            if not self.current_deployment or self.current_deployment.deployment_id != deployment_id:
                raise ValueError(f"Deployment {deployment_id} not found or not current")
            
            # Update status
            self.current_deployment.status = DeploymentStatus.DEPLOYING.value
            
            # Start monitoring
            self.start_deployment_monitoring()
            
            # Begin traffic shifting
            success = self.execute_traffic_stages()
            
            if success:
                self.current_deployment.status = DeploymentStatus.COMPLETED.value
                self.finalize_deployment()
            else:
                self.current_deployment.status = DeploymentStatus.FAILED.value
                self.initiate_rollback("Deployment failed during traffic shifting")
            
            self.save_deployment_state()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start deployment: {e}")
            self.initiate_rollback(f"Deployment error: {str(e)}")
            return False
    
    def execute_traffic_stages(self) -> bool:
        """Execute gradual traffic shifting"""
        try:
            stages = self.current_deployment.traffic_stages
            
            for i, stage in enumerate(stages):
                logger.info(f"Starting stage {i+1}/{len(stages)}: {stage['percentage']}% traffic")
                
                # Update traffic split
                self.update_traffic_split(stage['percentage'])
                
                # Wait for stage duration
                if stage['duration_minutes'] > 0:
                    stage_success = self.monitor_stage_health(
                        stage['duration_minutes'], 
                        stage['success_threshold']
                    )
                    
                    if not stage_success:
                        logger.error(f"Stage {i+1} failed health checks")
                        return False
                
                self.log_deployment_event("stage_completed", f"Stage {i+1} completed", {
                    'stage': stage,
                    'traffic_split': self.traffic_split
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Traffic stage execution failed: {e}")
            return False
    
    def update_traffic_split(self, green_percentage: int):
        """Update traffic routing split"""
        self.traffic_split = {
            "blue": 100 - green_percentage,
            "green": green_percentage
        }
        
        # Log traffic change
        self.log_traffic_change(self.traffic_split)
        
        logger.info(f"Updated traffic split: Blue {self.traffic_split['blue']}%, Green {self.traffic_split['green']}%")
    
    def monitor_stage_health(self, duration_minutes: int, success_threshold: float) -> bool:
        """Monitor health during a deployment stage"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            check_interval = self.deployment_config['health_checks']['check_interval']
            
            consecutive_failures = 0
            max_failures = self.deployment_config['health_checks']['failure_threshold']
            
            while datetime.now() < end_time:
                # Perform health check
                health_result = self.perform_health_check()
                
                if health_result['healthy']:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"Health check failed: {health_result['issues']}")
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive failures: {consecutive_failures}")
                        return False
                
                # Check for immediate rollback conditions
                if self.should_trigger_immediate_rollback(health_result):
                    logger.error("Immediate rollback conditions met")
                    return False
                
                time.sleep(check_interval)
            
            return True
            
        except Exception as e:
            logger.error(f"Stage health monitoring failed: {e}")
            return False
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_result = {
                'healthy': True,
                'issues': [],
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check response times
            avg_response_time = self.get_average_response_time()
            threshold = self.deployment_config['health_checks']['response_time_threshold']
            
            health_result['metrics']['response_time'] = avg_response_time
            
            if avg_response_time > threshold:
                health_result['healthy'] = False
                health_result['issues'].append(f"High response time: {avg_response_time:.2f}s")
            
            # Check error rates
            error_rate = self.get_current_error_rate()
            error_threshold = self.deployment_config['health_checks']['error_rate_threshold']
            
            health_result['metrics']['error_rate'] = error_rate
            
            if error_rate > error_threshold:
                health_result['healthy'] = False
                health_result['issues'].append(f"High error rate: {error_rate:.2%}")
            
            # Check prediction confidence
            avg_confidence = self.get_average_confidence()
            confidence_threshold = self.deployment_config['health_checks']['confidence_threshold']
            
            health_result['metrics']['confidence'] = avg_confidence
            
            if avg_confidence < confidence_threshold:
                health_result['healthy'] = False
                health_result['issues'].append(f"Low confidence: {avg_confidence:.2f}")
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'issues': [f"Health check error: {str(e)}"],
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def should_trigger_immediate_rollback(self, health_result: Dict) -> bool:
        """Check if immediate rollback should be triggered"""
        rollback_conditions = self.deployment_config['rollback_conditions']
        metrics = health_result['metrics']
        
        # Check error rate spike
        if metrics.get('error_rate', 0) > rollback_conditions['error_rate_spike']:
            return True
        
        # Check response time spike
        if metrics.get('response_time', 0) > rollback_conditions['response_time_spike']:
            return True
        
        # Check confidence drop
        if metrics.get('confidence', 1) < rollback_conditions['confidence_drop']:
            return True
        
        return False
    
    def initiate_rollback(self, reason: str) -> bool:
        """Initiate deployment rollback"""
        try:
            logger.warning(f"Initiating rollback: {reason}")
            
            if self.current_deployment:
                self.current_deployment.status = DeploymentStatus.ROLLING_BACK.value
            
            # Immediately route all traffic to blue (current production)
            self.update_traffic_split(0)  # 0% to green, 100% to blue
            
            # Clear staging environment
            self.clear_staging_environment()
            
            # Update deployment state
            if self.current_deployment:
                self.current_deployment.status = DeploymentStatus.FAILED.value
            
            self.save_deployment_state()
            
            self.log_deployment_event("rollback_initiated", f"Rollback initiated: {reason}", {
                'reason': reason,
                'traffic_split': self.traffic_split
            })
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def finalize_deployment(self):
        """Finalize successful deployment"""
        try:
            if not self.staging_version:
                raise ValueError("No staging version to finalize")
            
            # Move staging to active
            staging_env = self.staging_version['environment']
            
            # Update active version
            self.active_version = {
                **self.staging_version,
                'activated_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Clear staging
            self.staging_version = None
            
            # Update traffic split to 100% green if that's where new version is
            if staging_env == "green":
                self.update_traffic_split(100)
            else:
                self.update_traffic_split(0)
            
            # Archive old version if exists
            self.archive_old_version()
            
            self.log_deployment_event("deployment_finalized", "Deployment successfully finalized", {
                'active_version': self.active_version,
                'traffic_split': self.traffic_split
            })
            
            logger.info("Deployment finalized successfully")
            
        except Exception as e:
            logger.error(f"Failed to finalize deployment: {e}")
            raise e
    
    def get_active_version(self) -> Optional[Dict]:
        """Get currently active model version"""
        return self.active_version
    
    def get_staging_version(self) -> Optional[Dict]:
        """Get currently staged model version"""
        return self.staging_version
    
    def get_traffic_split(self) -> Dict[str, int]:
        """Get current traffic split configuration"""
        return self.traffic_split.copy()
    
    def determine_staging_environment(self) -> str:
        """Determine which environment to use for staging"""
        if not self.active_version:
            return "blue"  # Default to blue if no active version
        
        current_env = self.active_version.get('environment', 'blue')
        return "green" if current_env == "blue" else "blue"
    
    def version_exists(self, version_id: str) -> bool:
        """Check if a version exists"""
        version_dir = self.models_dir / version_id
        return version_dir.exists()
    
    def clear_staging_environment(self):
        """Clear the staging environment"""
        if self.staging_version:
            staging_env = self.staging_version['environment']
            staging_dir = self.blue_dir if staging_env == "blue" else self.green_dir
            
            import shutil
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
                staging_dir.mkdir(parents=True, exist_ok=True)
            
            self.staging_version = None
    
    def archive_old_version(self):
        """Archive the previously active version"""
        # Implementation for archiving old versions
        # This could move old versions to an archive directory
        pass
    
    def start_deployment_monitoring(self):
        """Start background deployment monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self.deployment_monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_deployment_monitoring(self):
        """Stop deployment monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
    
    def deployment_monitoring_loop(self):
        """Background monitoring loop for deployments"""
        while self.monitoring_active:
            try:
                if (self.current_deployment and 
                    self.current_deployment.status == DeploymentStatus.DEPLOYING.value):
                    
                    # Perform periodic health checks
                    health_result = self.perform_health_check()
                    
                    if self.should_trigger_immediate_rollback(health_result):
                        self.initiate_rollback("Automated rollback due to health check failures")
                        break
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Deployment monitoring error: {e}")
                time.sleep(60)
    
    def get_average_response_time(self) -> float:
        """Get average response time from recent requests"""
        # This would integrate with your monitoring system
        # For now, return a simulated value
        return np.random.normal(2.0, 0.5)
    
    def get_current_error_rate(self) -> float:
        """Get current error rate"""
        # This would integrate with your monitoring system
        # For now, return a simulated value
        return np.random.beta(1, 20)  # Typically low error rate
    
    def get_average_confidence(self) -> float:
        """Get average prediction confidence"""
        # This would integrate with your monitoring system
        # For now, return a simulated value
        return np.random.beta(8, 2)  # Typically high confidence
    
    def log_deployment_event(self, event: str, message: str, details: Dict = None):
        """Log deployment events"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'message': message,
                'details': details or {}
            }
            
            # Load existing logs
            logs = []
            if self.deployment_log_path.exists():
                try:
                    with open(self.deployment_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.deployment_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")
    
    def log_traffic_change(self, traffic_split: Dict[str, int]):
        """Log traffic routing changes"""
        try:
            traffic_entry = {
                'timestamp': datetime.now().isoformat(),
                'traffic_split': traffic_split,
                'deployment_id': self.current_deployment.deployment_id if self.current_deployment else None
            }
            
            # Load existing traffic logs
            logs = []
            if self.traffic_log_path.exists():
                try:
                    with open(self.traffic_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(traffic_entry)
            
            # Keep only last 500 entries
            if len(logs) > 500:
                logs = logs[-500:]
            
            # Save logs
            with open(self.traffic_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log traffic change: {e}")
    
    def save_deployment_state(self):
        """Save current deployment state"""
        try:
            state = {
                'current_deployment': asdict(self.current_deployment) if self.current_deployment else None,
                'active_version': self.active_version,
                'staging_version': self.staging_version,
                'traffic_split': self.traffic_split,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.deployment_state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save deployment state: {e}")
    
    def load_deployment_state(self):
        """Load existing deployment state"""
        try:
            if self.deployment_state_path.exists():
                with open(self.deployment_state_path, 'r') as f:
                    state = json.load(f)
                
                # Restore state
                if state.get('current_deployment'):
                    self.current_deployment = DeploymentPlan(**state['current_deployment'])
                
                self.active_version = state.get('active_version')
                self.staging_version = state.get('staging_version')
                self.traffic_split = state.get('traffic_split', {"blue": 100, "green": 0})
                
                logger.info("Loaded deployment state from file")
            
        except Exception as e:
            logger.warning(f"Failed to load deployment state: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'current_deployment': asdict(self.current_deployment) if self.current_deployment else None,
                'active_version': self.active_version,
                'staging_version': self.staging_version,
                'traffic_split': self.traffic_split,
                'monitoring_active': self.monitoring_active,
                'available_versions': self.list_available_versions(),
                'recent_deployments': self.get_recent_deployments(limit=5)
            }
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e)}
    
    def list_available_versions(self) -> List[str]:
        """List all available model versions"""
        try:
            versions = []
            if self.models_dir.exists():
                for item in self.models_dir.iterdir():
                    if item.is_dir() and item.name.startswith('v'):
                        versions.append(item.name)
            return sorted(versions, reverse=True)
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def get_recent_deployments(self, limit: int = 10) -> List[Dict]:
        """Get recent deployment history"""
        try:
            if not self.deployment_log_path.exists():
                return []
            
            with open(self.deployment_log_path, 'r') as f:
                logs = json.load(f)
            
            # Filter deployment events
            deployment_events = [
                log for log in logs 
                if log.get('event') in ['deployment_prepared', 'deployment_finalized', 'rollback_initiated']
            ]
            
            return deployment_events[-limit:]
            
        except Exception as e:
            logger.error(f"Failed to get recent deployments: {e}")
            return []