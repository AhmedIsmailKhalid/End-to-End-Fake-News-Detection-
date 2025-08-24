import json
import time
import psutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: Optional[float] = None

@dataclass
class APIMetrics:
    """API performance metrics"""
    timestamp: str
    total_requests: int
    requests_per_minute: float
    avg_response_time: float
    error_count: int
    error_rate: float
    active_connections: int
    endpoint_stats: Dict[str, Dict[str, Any]]

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: str
    model_version: str
    predictions_made: int
    avg_confidence: float
    confidence_distribution: Dict[str, int]
    prediction_distribution: Dict[str, int]
    processing_time_stats: Dict[str, float]
    model_health_score: float

class MetricsCollector:
    """Comprehensive metrics collection and aggregation system"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.monitor_dir = self.base_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.system_metrics_path = self.monitor_dir / "system_metrics.json"
        self.api_metrics_path = self.monitor_dir / "api_metrics.json"
        self.model_metrics_path = self.monitor_dir / "model_metrics.json"
        self.aggregated_metrics_path = self.monitor_dir / "aggregated_metrics.json"
        
        # In-memory storage
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours
        self.api_metrics_history = deque(maxlen=1440)
        self.model_metrics_history = deque(maxlen=1440)
        
        # Request tracking
        self.request_tracker = defaultdict(list)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_request': None
        })
        
        # Performance baselines
        self.baselines = {
            'response_time': 2.0,
            'cpu_usage': 70.0,
            'memory_usage': 80.0,
            'error_rate': 0.05
        }
        
        self.load_historical_metrics()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Load average (Unix systems)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                # Windows doesn't have getloadavg
                pass
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_total_mb=memory.total / (1024 * 1024),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                load_average=load_avg
            )
            
            # Store in history
            self.system_metrics_history.append(metrics)
            self._append_to_log(self.system_metrics_path, asdict(metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0
            )
    
    def record_api_request(self, 
                          endpoint: str, 
                          method: str, 
                          response_time: float, 
                          status_code: int,
                          client_ip: Optional[str] = None):
        """Record an API request"""
        timestamp = datetime.now()
        
        # Track request
        request_data = {
            'timestamp': timestamp.isoformat(),
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code,
            'client_ip': client_ip
        }
        
        # Add to request tracker for rate calculation
        self.request_tracker[timestamp.minute].append(request_data)
        
        # Update endpoint statistics
        endpoint_key = f"{method} {endpoint}"
        stats = self.endpoint_stats[endpoint_key]
        stats['count'] += 1
        stats['total_time'] += response_time
        stats['last_request'] = timestamp.isoformat()
        
        if status_code >= 400:
            stats['errors'] += 1
        
        # Clean old request data (keep last 5 minutes)
        cutoff_minute = (timestamp - timedelta(minutes=5)).minute
        keys_to_remove = [k for k in self.request_tracker.keys() if k < cutoff_minute]
        for key in keys_to_remove:
            del self.request_tracker[key]
    
    def collect_api_metrics(self) -> APIMetrics:
        """Collect current API performance metrics"""
        now = datetime.now()
        
        # Calculate requests in last minute
        current_minute_requests = self.request_tracker.get(now.minute, [])
        last_minute_requests = self.request_tracker.get((now - timedelta(minutes=1)).minute, [])
        recent_requests = current_minute_requests + last_minute_requests
        
        # Calculate metrics
        total_requests = sum(len(requests) for requests in self.request_tracker.values())
        requests_per_minute = len(recent_requests)
        
        if recent_requests:
            avg_response_time = np.mean([r['response_time'] for r in recent_requests])
            error_count = len([r for r in recent_requests if r['status_code'] >= 400])
            error_rate = error_count / len(recent_requests)
        else:
            avg_response_time = 0.0
            error_count = 0
            error_rate = 0.0
        
        # Endpoint statistics
        endpoint_stats = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats['count'] > 0:
                endpoint_stats[endpoint] = {
                    'count': stats['count'],
                    'avg_response_time': stats['total_time'] / stats['count'],
                    'error_count': stats['errors'],
                    'error_rate': stats['errors'] / stats['count'],
                    'last_request': stats['last_request']
                }
        
        metrics = APIMetrics(
            timestamp=now.isoformat(),
            total_requests=total_requests,
            requests_per_minute=requests_per_minute,
            avg_response_time=avg_response_time,
            error_count=error_count,
            error_rate=error_rate,
            active_connections=0,  # This would need actual connection tracking
            endpoint_stats=endpoint_stats
        )
        
        # Store in history
        self.api_metrics_history.append(metrics)
        self._append_to_log(self.api_metrics_path, asdict(metrics))
        
        return metrics
    
    def collect_model_metrics(self, prediction_monitor) -> ModelMetrics:
        """Collect model performance metrics"""
        try:
            current_metrics = prediction_monitor.get_current_metrics()
            recent_predictions = prediction_monitor._get_recent_predictions(minutes=60)
            
            if recent_predictions:
                processing_times = [p.processing_time for p in recent_predictions]
                processing_time_stats = {
                    'mean': float(np.mean(processing_times)),
                    'std': float(np.std(processing_times)),
                    'min': float(np.min(processing_times)),
                    'max': float(np.max(processing_times)),
                    'p95': float(np.percentile(processing_times, 95))
                }
                
                # Calculate model health score
                health_score = self._calculate_model_health_score(current_metrics, processing_time_stats)
                
                model_version = recent_predictions[0].model_version if recent_predictions else "unknown"
            else:
                processing_time_stats = {}
                health_score = 0.0
                model_version = "unknown"
            
            metrics = ModelMetrics(
                timestamp=datetime.now().isoformat(),
                model_version=model_version,
                predictions_made=current_metrics.total_predictions,
                avg_confidence=current_metrics.avg_confidence,
                confidence_distribution=current_metrics.confidence_distribution,
                prediction_distribution=current_metrics.prediction_distribution,
                processing_time_stats=processing_time_stats,
                model_health_score=health_score
            )
            
            # Store in history
            self.model_metrics_history.append(metrics)
            self._append_to_log(self.model_metrics_path, asdict(metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")
            return ModelMetrics(
                timestamp=datetime.now().isoformat(),
                model_version="unknown",
                predictions_made=0,
                avg_confidence=0.0,
                confidence_distribution={},
                prediction_distribution={},
                processing_time_stats={},
                model_health_score=0.0
            )
    
    def get_aggregated_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get aggregated metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics_history 
                        if datetime.fromisoformat(m.timestamp) > cutoff_time]
        recent_api = [m for m in self.api_metrics_history 
                     if datetime.fromisoformat(m.timestamp) > cutoff_time]
        recent_model = [m for m in self.model_metrics_history 
                       if datetime.fromisoformat(m.timestamp) > cutoff_time]
        
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'time_period_hours': hours,
            'system_metrics': self._aggregate_system_metrics(recent_system),
            'api_metrics': self._aggregate_api_metrics(recent_api),
            'model_metrics': self._aggregate_model_metrics(recent_model),
            'overall_health_score': 0.0,
            'alerts': self._generate_metric_alerts(recent_system, recent_api, recent_model)
        }
        
        # Calculate overall health score
        aggregated['overall_health_score'] = self._calculate_overall_health_score(aggregated)
        
        # Save aggregated metrics
        self._append_to_log(self.aggregated_metrics_path, aggregated)
        
        return aggregated
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get historical data
        recent_system = [m for m in self.system_metrics_history 
                        if datetime.fromisoformat(m.timestamp) > cutoff_time]
        recent_api = [m for m in self.api_metrics_history 
                     if datetime.fromisoformat(m.timestamp) > cutoff_time]
        recent_model = [m for m in self.model_metrics_history 
                       if datetime.fromisoformat(m.timestamp) > cutoff_time]
        
        trends = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'system_trends': self._analyze_system_trends(recent_system),
            'api_trends': self._analyze_api_trends(recent_api),
            'model_trends': self._analyze_model_trends(recent_model),
            'correlation_analysis': self._analyze_correlations(recent_system, recent_api, recent_model)
        }
        
        return trends
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get current data for real-time dashboard"""
        try:
            # Get latest metrics
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            latest_api = self.api_metrics_history[-1] if self.api_metrics_history else None
            latest_model = self.model_metrics_history[-1] if self.model_metrics_history else None
            
            # Get recent trends (last hour)
            recent_aggregated = self.get_aggregated_metrics(hours=1)
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'status': self._determine_system_status(latest_system, latest_api, latest_model),
                'current_metrics': {
                    'system': asdict(latest_system) if latest_system else None,
                    'api': asdict(latest_api) if latest_api else None,
                    'model': asdict(latest_model) if latest_model else None
                },
                'hourly_summary': recent_aggregated,
                'active_alerts': self._get_active_alerts(),
                'key_indicators': self._get_key_indicators(latest_system, latest_api, latest_model)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'unknown',
                'error': str(e)
            }
    
    def _calculate_model_health_score(self, metrics, processing_stats: Dict) -> float:
        """Calculate overall model health score (0-1)"""
        scores = []
        
        # Confidence score
        if metrics.avg_confidence > 0:
            confidence_score = min(metrics.avg_confidence / 0.8, 1.0)  # Target: 80%+
            scores.append(confidence_score)
        
        # Processing time score
        if processing_stats and 'mean' in processing_stats:
            processing_score = max(0, 1.0 - (processing_stats['mean'] / 10.0))  # Target: <10s
            scores.append(processing_score)
        
        # Error rate score
        error_score = max(0, 1.0 - (metrics.error_rate / 0.1))  # Target: <10%
        scores.append(error_score)
        
        # Prediction rate score (activity indicator)
        if metrics.predictions_per_minute > 0:
            activity_score = min(metrics.predictions_per_minute / 10.0, 1.0)  # Normalize to 10 req/min
            scores.append(activity_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _aggregate_system_metrics(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Aggregate system metrics"""
        if not metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        disk_values = [m.disk_usage_percent for m in metrics]
        
        return {
            'cpu_usage': {
                'avg': float(np.mean(cpu_values)),
                'max': float(np.max(cpu_values)),
                'min': float(np.min(cpu_values)),
                'current': cpu_values[-1]
            },
            'memory_usage': {
                'avg': float(np.mean(memory_values)),
                'max': float(np.max(memory_values)),
                'min': float(np.min(memory_values)),
                'current': memory_values[-1]
            },
            'disk_usage': {
                'avg': float(np.mean(disk_values)),
                'max': float(np.max(disk_values)),
                'min': float(np.min(disk_values)),
                'current': disk_values[-1]
            },
            'sample_count': len(metrics)
        }
    
    def _aggregate_api_metrics(self, metrics: List[APIMetrics]) -> Dict[str, Any]:
        """Aggregate API metrics"""
        if not metrics:
            return {}
        
        response_times = [m.avg_response_time for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        request_rates = [m.requests_per_minute for m in metrics]
        
        return {
            'response_time': {
                'avg': float(np.mean(response_times)),
                'max': float(np.max(response_times)),
                'min': float(np.min(response_times)),
                'p95': float(np.percentile(response_times, 95))
            },
            'error_rate': {
                'avg': float(np.mean(error_rates)),
                'max': float(np.max(error_rates)),
                'current': error_rates[-1]
            },
            'request_rate': {
                'avg': float(np.mean(request_rates)),
                'max': float(np.max(request_rates)),
                'current': request_rates[-1]
            },
            'total_requests': sum(m.total_requests for m in metrics),
            'sample_count': len(metrics)
        }
    
    def _aggregate_model_metrics(self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Aggregate model metrics"""
        if not metrics:
            return {}
        
        confidences = [m.avg_confidence for m in metrics]
        health_scores = [m.model_health_score for m in metrics]
        
        return {
            'confidence': {
                'avg': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'current': confidences[-1]
            },
            'health_score': {
                'avg': float(np.mean(health_scores)),
                'min': float(np.min(health_scores)),
                'current': health_scores[-1]
            },
            'total_predictions': sum(m.predictions_made for m in metrics),
            'sample_count': len(metrics)
        }
    
    def _analyze_system_trends(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Analyze system metric trends"""
        if len(metrics) < 2:
            return {}
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        return {
            'cpu_trend': self._calculate_trend(cpu_values),
            'memory_trend': self._calculate_trend(memory_values),
            'stability_score': self._calculate_stability_score(cpu_values, memory_values)
        }
    
    def _analyze_api_trends(self, metrics: List[APIMetrics]) -> Dict[str, Any]:
        """Analyze API metric trends"""
        if len(metrics) < 2:
            return {}
        
        response_times = [m.avg_response_time for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        
        return {
            'response_time_trend': self._calculate_trend(response_times),
            'error_rate_trend': self._calculate_trend(error_rates),
            'performance_score': self._calculate_performance_score(response_times, error_rates)
        }
    
    def _analyze_model_trends(self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Analyze model metric trends"""
        if len(metrics) < 2:
            return {}
        
        confidences = [m.avg_confidence for m in metrics]
        health_scores = [m.model_health_score for m in metrics]
        
        return {
            'confidence_trend': self._calculate_trend(confidences),
            'health_trend': self._calculate_trend(health_scores),
            'model_stability': self._calculate_model_stability(confidences)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        recent_avg = np.mean(values[-5:])  # Last 5 values
        older_avg = np.mean(values[:-5]) if len(values) > 5 else np.mean(values[:-2])
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
        
        if change_percent > 5:
            return 'increasing'
        elif change_percent < -5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_stability_score(self, *value_lists) -> float:
        """Calculate stability score based on coefficient of variation"""
        scores = []
        for values in value_lists:
            if values and len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1
                stability = max(0, 1 - cv)
                scores.append(stability)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _calculate_performance_score(self, response_times: List[float], error_rates: List[float]) -> float:
        """Calculate overall performance score"""
        scores = []
        
        # Response time score
        if response_times:
            avg_response_time = np.mean(response_times)
            response_score = max(0, 1 - (avg_response_time / self.baselines['response_time']))
            scores.append(response_score)
        
        # Error rate score
        if error_rates:
            avg_error_rate = np.mean(error_rates)
            error_score = max(0, 1 - (avg_error_rate / self.baselines['error_rate']))
            scores.append(error_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _calculate_model_stability(self, confidences: List[float]) -> float:
        """Calculate model stability based on confidence consistency"""
        if not confidences or len(confidences) < 2:
            return 0.0
        
        cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 1
        return float(max(0, 1 - cv))
    
    def _analyze_correlations(self, system_metrics, api_metrics, model_metrics) -> Dict[str, Any]:
        """Analyze correlations between different metric types"""
        correlations = {}
        
        try:
            if system_metrics and api_metrics:
                cpu_values = [m.cpu_percent for m in system_metrics]
                response_times = [m.avg_response_time for m in api_metrics]
                
                if len(cpu_values) == len(response_times) and len(cpu_values) > 1:
                    correlation = np.corrcoef(cpu_values, response_times)[0, 1]
                    correlations['cpu_response_time'] = float(correlation)
            
            # Add more correlation analyses as needed
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
        
        return correlations
    
    def _calculate_overall_health_score(self, aggregated: Dict) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # System health
        system_metrics = aggregated.get('system_metrics', {})
        if system_metrics:
            cpu_score = max(0, 1 - (system_metrics['cpu_usage']['current'] / 100))
            memory_score = max(0, 1 - (system_metrics['memory_usage']['current'] / 100))
            scores.extend([cpu_score, memory_score])
        
        # API health
        api_metrics = aggregated.get('api_metrics', {})
        if api_metrics:
            response_score = max(0, 1 - (api_metrics['response_time']['current'] / 10))
            error_score = max(0, 1 - api_metrics['error_rate']['current'])
            scores.extend([response_score, error_score])
        
        # Model health
        model_metrics = aggregated.get('model_metrics', {})
        if model_metrics:
            model_score = model_metrics['health_score']['current']
            scores.append(model_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _generate_metric_alerts(self, system_metrics, api_metrics, model_metrics) -> List[Dict]:
        """Generate alerts based on metric thresholds"""
        alerts = []
        
        # System alerts
        if system_metrics:
            latest_system = system_metrics[-1]
            if latest_system.cpu_percent > self.baselines['cpu_usage']:
                alerts.append({
                    'type': 'warning',
                    'category': 'system',
                    'message': f"High CPU usage: {latest_system.cpu_percent:.1f}%",
                    'timestamp': latest_system.timestamp
                })
            
            if latest_system.memory_percent > self.baselines['memory_usage']:
                alerts.append({
                    'type': 'warning',
                    'category': 'system',
                    'message': f"High memory usage: {latest_system.memory_percent:.1f}%",
                    'timestamp': latest_system.timestamp
                })
        
        # API alerts
        if api_metrics:
            latest_api = api_metrics[-1]
            if latest_api.avg_response_time > self.baselines['response_time']:
                alerts.append({
                    'type': 'warning',
                    'category': 'api',
                    'message': f"Slow response time: {latest_api.avg_response_time:.2f}s",
                    'timestamp': latest_api.timestamp
                })
            
            if latest_api.error_rate > self.baselines['error_rate']:
                alerts.append({
                    'type': 'critical',
                    'category': 'api',
                    'message': f"High error rate: {latest_api.error_rate:.2%}",
                    'timestamp': latest_api.timestamp
                })
        
        return alerts
    
    def _determine_system_status(self, system_metrics, api_metrics, model_metrics) -> str:
        """Determine overall system status"""
        if not system_metrics or not api_metrics or not model_metrics:
            return 'unknown'
        
        # Check for critical issues
        if (system_metrics.cpu_percent > 90 or 
            system_metrics.memory_percent > 95 or
            api_metrics.error_rate > 0.1 or
            api_metrics.avg_response_time > 10):
            return 'critical'
        
        # Check for warnings
        if (system_metrics.cpu_percent > 70 or 
            system_metrics.memory_percent > 80 or
            api_metrics.error_rate > 0.05 or
            api_metrics.avg_response_time > 5 or
            model_metrics.avg_confidence < 0.6):
            return 'warning'
        
        return 'healthy'
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts"""
        # This would typically read from alerts log and filter recent alerts
        return []
    
    def _get_key_indicators(self, system_metrics, api_metrics, model_metrics) -> Dict[str, Any]:
        """Get key performance indicators"""
        indicators = {}
        
        if system_metrics:
            indicators['cpu_usage'] = system_metrics.cpu_percent
            indicators['memory_usage'] = system_metrics.memory_percent
        
        if api_metrics:
            indicators['response_time'] = api_metrics.avg_response_time
            indicators['requests_per_minute'] = api_metrics.requests_per_minute
        
        if model_metrics:
            indicators['model_confidence'] = model_metrics.avg_confidence
            indicators['model_health'] = model_metrics.model_health_score
        
        return indicators
    
    def _append_to_log(self, log_path: Path, data: Dict):
        """Append data to log file"""
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log {log_path}: {e}")
    
    def load_historical_metrics(self):
        """Load historical metrics on startup"""
        try:
            # Load recent metrics (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for log_path, history_deque, metric_class in [
                (self.system_metrics_path, self.system_metrics_history, SystemMetrics),
                (self.api_metrics_path, self.api_metrics_history, APIMetrics),
                (self.model_metrics_path, self.model_metrics_history, ModelMetrics)
            ]:
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if datetime.fromisoformat(data['timestamp']) > cutoff_time:
                                    metric = metric_class(**data)
                                    history_deque.append(metric)
                            except Exception:
                                continue
            
            logger.info(f"Loaded historical metrics: {len(self.system_metrics_history)} system, "
                       f"{len(self.api_metrics_history)} API, {len(self.model_metrics_history)} model")
            
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")