import schedule
import time
import logging
import json
import psutil
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager
import subprocess
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustTaskScheduler:
    """Production-ready task scheduler with comprehensive error handling and monitoring"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_scheduler_config()
        self.setup_task_registry()
        self.setup_monitoring()
        self.setup_signal_handlers()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running = True
        self.lock = threading.Lock()
        
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.activity_log_path = Path("/tmp/activity_log.json")
        self.scheduler_log_path = self.logs_dir / "scheduler_execution.json"
        self.error_log_path = self.logs_dir / "scheduler_errors.json"
        self.performance_log_path = self.logs_dir / "scheduler_performance.json"
    
    def setup_scheduler_config(self):
        """Setup scheduler configuration"""
        self.config = {
            'scraping_interval': 'hourly',
            'generation_interval': 'hourly', 
            'retraining_interval': 'hourly',
            'monitoring_interval': 'hourly',
            'health_check_interval': 'every(10).minutes',
            'cleanup_interval': 'daily',
            'max_task_duration': 1800,  # 30 minutes
            'max_retries': 3,
            'retry_delay': 300,  # 5 minutes
            'resource_limits': {
                'max_cpu_percent': 80,
                'max_memory_percent': 85,
                'max_disk_usage_percent': 90
            }
        }
    
    def setup_task_registry(self):
        """Setup task registry with metadata"""
        self.task_registry = {
            'scrape_news': {
                'function': self.scrape_news_task,
                'description': 'Scrape real news articles from various sources',
                'dependencies': [],
                'timeout': 900,  # 15 minutes
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            },
            'generate_fake_news': {
                'function': self.generate_fake_news_task,
                'description': 'Generate synthetic fake news articles',
                'dependencies': [],
                'timeout': 300,  # 5 minutes
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            },
            'retrain_model': {
                'function': self.retrain_model_task,
                'description': 'Retrain ML model with new data',
                'dependencies': ['scrape_news', 'generate_fake_news'],
                'timeout': 1800,  # 30 minutes
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            },
            'monitor_drift': {
                'function': self.monitor_drift_task,
                'description': 'Monitor data and model drift',
                'dependencies': ['retrain_model'],
                'timeout': 600,  # 10 minutes
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            },
            'system_health_check': {
                'function': self.system_health_check_task,
                'description': 'Check system health and resources',
                'dependencies': [],
                'timeout': 60,  # 1 minute
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            },
            'cleanup_old_files': {
                'function': self.cleanup_old_files_task,
                'description': 'Clean up old log files and temporary data',
                'dependencies': [],
                'timeout': 300,  # 5 minutes
                'retry_count': 0,
                'last_run': None,
                'last_success': None,
                'enabled': True
            }
        }
    
    def setup_monitoring(self):
        """Setup monitoring and metrics"""
        self.metrics = {
            'tasks_executed': 0,
            'tasks_succeeded': 0,
            'tasks_failed': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'last_health_check': None,
            'system_status': 'healthy',
            'startup_time': datetime.now().isoformat()
        }
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        # Wait for running tasks to complete
        self.executor.shutdown(wait=True, timeout=60)
        
        # Log shutdown
        self.log_event("Scheduler shutdown completed")
        sys.exit(0)
    
    def log_event(self, event: str, level: str = "INFO", metadata: Dict = None):
        """Log events with timestamps and metadata"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "event": event,
            "level": level,
            "metadata": metadata or {}
        }
        
        try:
            # Load existing logs
            logs = []
            if self.activity_log_path.exists():
                try:
                    with open(self.activity_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            # Add new log
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.activity_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'healthy': (
                    cpu_percent < self.config['resource_limits']['max_cpu_percent'] and
                    memory.percent < self.config['resource_limits']['max_memory_percent'] and
                    disk.percent < self.config['resource_limits']['max_disk_usage_percent']
                )
            }
        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def can_run_task(self, task_name: str) -> Tuple[bool, str]:
        """Check if a task can be run based on system resources and dependencies"""
        # Check if task is enabled
        if not self.task_registry[task_name]['enabled']:
            return False, f"Task {task_name} is disabled"
        
        # Check system resources
        resources = self.check_system_resources()
        if not resources['healthy']:
            return False, f"System resources insufficient: {resources}"
        
        # Check dependencies
        dependencies = self.task_registry[task_name]['dependencies']
        for dep in dependencies:
            dep_task = self.task_registry.get(dep)
            if dep_task is None:
                return False, f"Dependency {dep} not found"
            
            # Check if dependency ran recently and successfully
            if dep_task['last_success'] is None:
                return False, f"Dependency {dep} has never run successfully"
            
            # Check if dependency ran within reasonable time
            last_success = datetime.fromisoformat(dep_task['last_success'])
            if datetime.now() - last_success > timedelta(hours=2):
                return False, f"Dependency {dep} last success too old"
        
        return True, "OK"
    
    @contextmanager
    def task_execution_context(self, task_name: str):
        """Context manager for task execution with timing and error handling"""
        start_time = time.time()
        
        try:
            self.task_registry[task_name]['last_run'] = datetime.now().isoformat()
            self.metrics['tasks_executed'] += 1
            
            logger.info(f"Starting task: {task_name}")
            yield
            
            # Task succeeded
            execution_time = time.time() - start_time
            self.task_registry[task_name]['last_success'] = datetime.now().isoformat()
            self.task_registry[task_name]['retry_count'] = 0
            self.metrics['tasks_succeeded'] += 1
            self.metrics['total_execution_time'] += execution_time
            self.metrics['average_execution_time'] = (
                self.metrics['total_execution_time'] / self.metrics['tasks_executed']
            )
            
            logger.info(f"Task {task_name} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            # Task failed
            execution_time = time.time() - start_time
            self.task_registry[task_name]['retry_count'] += 1
            self.metrics['tasks_failed'] += 1
            
            error_details = {
                'task': task_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': execution_time,
                'retry_count': self.task_registry[task_name]['retry_count']
            }
            
            self.log_error(error_details)
            logger.error(f"Task {task_name} failed after {execution_time:.2f}s: {e}")
            raise
    
    def log_error(self, error_details: Dict):
        """Log error details for debugging"""
        try:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                **error_details
            }
            
            # Load existing errors
            errors = []
            if self.error_log_path.exists():
                try:
                    with open(self.error_log_path, 'r') as f:
                        errors = json.load(f)
                except:
                    errors = []
            
            # Add new error
            errors.append(error_entry)
            
            # Keep only last 100 errors
            if len(errors) > 100:
                errors = errors[-100:]
            
            # Save errors
            with open(self.error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def run_task_with_retry(self, task_name: str):
        """Run a task with retry logic"""
        task_info = self.task_registry[task_name]
        max_retries = self.config['max_retries']
        
        for attempt in range(max_retries + 1):
            try:
                # Check if we can run the task
                can_run, reason = self.can_run_task(task_name)
                if not can_run:
                    logger.warning(f"Cannot run task {task_name}: {reason}")
                    return False
                
                # Execute task
                with self.task_execution_context(task_name):
                    task_info['function']()
                
                return True
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = self.config['retry_delay'] * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Task {task_name} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Task {task_name} failed after {max_retries + 1} attempts")
                    self.log_event(f"Task {task_name} failed permanently", "ERROR", {"attempts": max_retries + 1})
                    return False
        
        return False
    
    # Task implementations
    def scrape_news_task(self):
        """Scrape news articles task"""
        try:
            from data.scrape_real_news import scrape_articles
            success = scrape_articles()
            if not success:
                raise Exception("News scraping failed")
            self.log_event("News scraping completed successfully")
        except Exception as e:
            raise Exception(f"News scraping failed: {e}")
    
    def generate_fake_news_task(self):
        """Generate fake news task"""
        try:
            from data.generate_fake_news import generate_fake_news
            success = generate_fake_news(25)
            if not success:
                raise Exception("Fake news generation failed")
            self.log_event("Fake news generation completed successfully")
        except Exception as e:
            raise Exception(f"Fake news generation failed: {e}")
    
    def retrain_model_task(self):
        """Retrain model task"""
        try:
            from model.retrain import main as retrain_main
            retrain_main()
            self.log_event("Model retraining completed successfully")
        except Exception as e:
            raise Exception(f"Model retraining failed: {e}")
    
    def monitor_drift_task(self):
        """Monitor drift task"""
        try:
            from monitor.monitor_drift import monitor_drift
            drift_score = monitor_drift()
            if drift_score is not None:
                self.log_event(f"Drift monitoring completed", metadata={"drift_score": drift_score})
            else:
                raise Exception("Drift monitoring returned None")
        except Exception as e:
            raise Exception(f"Drift monitoring failed: {e}")
    
    def system_health_check_task(self):
        """System health check task"""
        try:
            resources = self.check_system_resources()
            
            # Check critical files
            critical_files = [
                Path("/tmp/model.pkl"),
                Path("/tmp/vectorizer.pkl"),
                Path("/tmp/data/combined_dataset.csv")
            ]
            
            missing_files = [f for f in critical_files if not f.exists()]
            
            health_status = {
                'resources': resources,
                'missing_files': [str(f) for f in missing_files],
                'healthy': resources['healthy'] and len(missing_files) == 0
            }
            
            self.metrics['last_health_check'] = datetime.now().isoformat()
            self.metrics['system_status'] = 'healthy' if health_status['healthy'] else 'unhealthy'
            
            if not health_status['healthy']:
                self.log_event("System health check failed", "WARNING", health_status)
            
            logger.info(f"System health check completed: {health_status['healthy']}")
            
        except Exception as e:
            raise Exception(f"System health check failed: {e}")
    
    def cleanup_old_files_task(self):
        """Clean up old files task"""
        try:
            cleanup_count = 0
            
            # Clean up old log files
            log_dirs = [Path("/tmp/logs"), Path("/tmp")]
            for log_dir in log_dirs:
                if log_dir.exists():
                    for log_file in log_dir.glob("*.log"):
                        if log_file.stat().st_mtime < time.time() - (7 * 24 * 3600):  # 7 days
                            log_file.unlink()
                            cleanup_count += 1
            
            # Clean up old backup files
            backup_dir = Path("/tmp/backups")
            if backup_dir.exists():
                for backup_file in backup_dir.glob("backup_*"):
                    if backup_file.stat().st_mtime < time.time() - (30 * 24 * 3600):  # 30 days
                        if backup_file.is_dir():
                            import shutil
                            shutil.rmtree(backup_file)
                        else:
                            backup_file.unlink()
                        cleanup_count += 1
            
            self.log_event(f"Cleanup completed: {cleanup_count} files removed")
            
        except Exception as e:
            raise Exception(f"Cleanup failed: {e}")
    
    def run_pipeline_sequence(self):
        """Run the main pipeline sequence"""
        logger.info("Starting pipeline sequence...")
        
        # Define task sequence
        pipeline_tasks = [
            'scrape_news',
            'generate_fake_news',
            'retrain_model',
            'monitor_drift'
        ]
        
        success_count = 0
        for task_name in pipeline_tasks:
            if self.run_task_with_retry(task_name):
                success_count += 1
            else:
                logger.error(f"Pipeline halted due to task failure: {task_name}")
                break
        
        if success_count == len(pipeline_tasks):
            self.log_event("Pipeline sequence completed successfully")
        else:
            self.log_event(f"Pipeline sequence partially completed: {success_count}/{len(pipeline_tasks)} tasks")
    
    def schedule_tasks(self):
        """Schedule all tasks according to configuration"""
        logger.info("Scheduling tasks...")
        
        # Schedule main pipeline
        schedule.every().hour.do(self.run_pipeline_sequence)
        
        # Schedule individual monitoring tasks
        schedule.every(10).minutes.do(self.run_task_with_retry, 'system_health_check')
        schedule.every().day.at("02:00").do(self.run_task_with_retry, 'cleanup_old_files')
        
        logger.info("All tasks scheduled successfully")
    
    def save_performance_metrics(self):
        """Save performance metrics periodically"""
        try:
            metrics_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.copy()
            }
            
            # Load existing metrics
            metrics_log = []
            if self.performance_log_path.exists():
                try:
                    with open(self.performance_log_path, 'r') as f:
                        metrics_log = json.load(f)
                except:
                    metrics_log = []
            
            # Add new metrics
            metrics_log.append(metrics_entry)
            
            # Keep only last 100 entries
            if len(metrics_log) > 100:
                metrics_log = metrics_log[-100:]
            
            # Save metrics
            with open(self.performance_log_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def run(self):
        """Main scheduler loop"""
        logger.info("Starting robust task scheduler...")
        
        # Initial system health check
        self.run_task_with_retry('system_health_check')
        
        # Schedule all tasks
        self.schedule_tasks()
        
        # Run initial pipeline
        self.run_pipeline_sequence()
        
        # Main loop
        last_metrics_save = time.time()
        
        while self.running:
            try:
                schedule.run_pending()
                
                # Save metrics every 10 minutes
                if time.time() - last_metrics_save > 600:
                    self.save_performance_metrics()
                    last_metrics_save = time.time()
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Scheduler shutdown complete")

def main():
    """Main execution function"""
    scheduler = RobustTaskScheduler()
    scheduler.run()

if __name__ == "__main__":
    main()