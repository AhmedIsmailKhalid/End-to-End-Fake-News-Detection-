import json
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    """Individual prediction record with metadata"""
    timestamp: str
    text_hash: str
    prediction: str
    confidence: float
    processing_time: float
    model_version: str
    text_length: int
    word_count: int
    client_id: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class MonitoringMetrics:
    """Aggregated monitoring metrics"""
    timestamp: str
    total_predictions: int
    predictions_per_minute: float
    avg_confidence: float
    avg_processing_time: float
    confidence_distribution: Dict[str, int]
    prediction_distribution: Dict[str, int]
    error_rate: float
    response_time_percentiles: Dict[str, float]
    anomaly_score: float

class PredictionMonitor:
    """Real-time prediction monitoring system"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.monitor_dir = self.base_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.predictions_log_path = self.monitor_dir / "predictions.json"
        self.metrics_log_path = self.monitor_dir / "metrics.json"
        self.alerts_log_path = self.monitor_dir / "alerts.json"
        
        # In-memory storage for real-time analysis
        self.recent_predictions = deque(maxlen=10000)  # Last 10k predictions
        self.prediction_buffer = deque(maxlen=1000)    # Buffer for batch processing
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-level metrics
        self.error_count = 0
        self.total_predictions = 0
        
        # Configuration
        self.confidence_thresholds = {
            'very_low': 0.5,
            'low': 0.7,
            'medium': 0.8,
            'high': 0.9
        }
        
        self.performance_thresholds = {
            'response_time_warning': 5.0,  # seconds
            'response_time_critical': 10.0,
            'confidence_warning': 0.6,     # average confidence below this
            'error_rate_warning': 0.05,    # 5% error rate
            'error_rate_critical': 0.10    # 10% error rate
        }
        
        # Background processing
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Load existing data
        self.load_historical_data()
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Prediction monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Prediction monitoring stopped")
    
    def record_prediction(self, 
                         prediction: str,
                         confidence: float,
                         processing_time: float,
                         text: str,
                         model_version: str = "unknown",
                         client_id: Optional[str] = None,
                         user_agent: Optional[str] = None,
                         session_id: Optional[str] = None) -> str:
        """Record a new prediction with comprehensive metadata"""
        
        # Create prediction record
        text_hash = self._hash_text(text)
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            text_hash=text_hash,
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time,
            model_version=model_version,
            text_length=len(text),
            word_count=len(text.split()),
            client_id=client_id,
            user_agent=user_agent,
            session_id=session_id
        )
        
        # Add to in-memory storage
        self.recent_predictions.append(record)
        self.prediction_buffer.append(record)
        self.total_predictions += 1
        
        # Trigger batch processing if buffer is full
        if len(self.prediction_buffer) >= 100:
            self._process_prediction_batch()
        
        return text_hash
    
    def record_error(self, error_type: str, error_message: str, context: Dict = None):
        """Record prediction error"""
        self.error_count += 1
        
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'total_errors': self.error_count,
            'error_rate': self.get_current_error_rate()
        }
        
        # Save error to alerts log
        self._append_to_log(self.alerts_log_path, error_record)
        
        # Check if error rate exceeds thresholds
        self._check_error_rate_alerts()
    
    def get_current_metrics(self) -> MonitoringMetrics:
        """Get current real-time metrics"""
        now = datetime.now()
        recent_predictions = self._get_recent_predictions(minutes=5)
        
        if not recent_predictions:
            return MonitoringMetrics(
                timestamp=now.isoformat(),
                total_predictions=self.total_predictions,
                predictions_per_minute=0.0,
                avg_confidence=0.0,
                avg_processing_time=0.0,
                confidence_distribution={},
                prediction_distribution={},
                error_rate=0.0,
                response_time_percentiles={},
                anomaly_score=0.0
            )
        
        # Calculate metrics
        confidences = [p.confidence for p in recent_predictions]
        processing_times = [p.processing_time for p in recent_predictions]
        predictions = [p.prediction for p in recent_predictions]
        
        return MonitoringMetrics(
            timestamp=now.isoformat(),
            total_predictions=self.total_predictions,
            predictions_per_minute=len(recent_predictions) / 5.0,
            avg_confidence=float(np.mean(confidences)),
            avg_processing_time=float(np.mean(processing_times)),
            confidence_distribution=self._calculate_confidence_distribution(confidences),
            prediction_distribution=self._calculate_prediction_distribution(predictions),
            error_rate=self.get_current_error_rate(),
            response_time_percentiles=self._calculate_percentiles(processing_times),
            anomaly_score=self._calculate_anomaly_score(recent_predictions)
        )
    
    def get_historical_metrics(self, hours: int = 24) -> List[MonitoringMetrics]:
        """Get historical metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        historical_metrics = []
        for metrics in self.metrics_history:
            if datetime.fromisoformat(metrics.timestamp) > cutoff_time:
                historical_metrics.append(metrics)
        
        return historical_metrics
    
    def get_prediction_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze prediction patterns for anomaly detection"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = [
            p for p in self.recent_predictions 
            if datetime.fromisoformat(p.timestamp) > cutoff_time
        ]
        
        if not recent_predictions:
            return {'error': 'No recent predictions found'}
        
        # Analyze patterns
        hourly_distribution = defaultdict(int)
        confidence_trends = []
        processing_time_trends = []
        
        for prediction in recent_predictions:
            hour = datetime.fromisoformat(prediction.timestamp).hour
            hourly_distribution[hour] += 1
            confidence_trends.append(prediction.confidence)
            processing_time_trends.append(prediction.processing_time)
        
        return {
            'total_predictions': len(recent_predictions),
            'hourly_distribution': dict(hourly_distribution),
            'confidence_stats': {
                'mean': float(np.mean(confidence_trends)),
                'std': float(np.std(confidence_trends)),
                'min': float(np.min(confidence_trends)),
                'max': float(np.max(confidence_trends))
            },
            'processing_time_stats': {
                'mean': float(np.mean(processing_time_trends)),
                'std': float(np.std(processing_time_trends)),
                'min': float(np.min(processing_time_trends)),
                'max': float(np.max(processing_time_trends))
            },
            'anomaly_indicators': self._detect_anomaly_indicators(recent_predictions)
        }
    
    def get_current_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.total_predictions == 0:
            return 0.0
        return self.error_count / (self.total_predictions + self.error_count)
    
    def get_confidence_analysis(self) -> Dict[str, Any]:
        """Analyze confidence distribution and trends"""
        recent_predictions = self._get_recent_predictions(minutes=60)
        
        if not recent_predictions:
            return {'error': 'No recent predictions found'}
        
        confidences = [p.confidence for p in recent_predictions]
        
        # Confidence distribution
        distribution = self._calculate_confidence_distribution(confidences)
        
        # Confidence trends (last hour in 10-minute windows)
        trends = []
        now = datetime.now()
        for i in range(6):  # 6 ten-minute windows
            window_start = now - timedelta(minutes=(i+1)*10)
            window_end = now - timedelta(minutes=i*10)
            
            window_predictions = [
                p for p in recent_predictions
                if window_start <= datetime.fromisoformat(p.timestamp) < window_end
            ]
            
            if window_predictions:
                avg_confidence = np.mean([p.confidence for p in window_predictions])
                trends.append({
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'avg_confidence': float(avg_confidence),
                    'prediction_count': len(window_predictions)
                })
        
        return {
            'total_predictions': len(recent_predictions),
            'overall_avg_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'distribution': distribution,
            'trends': trends[::-1],  # Reverse to get chronological order
            'low_confidence_alerts': len([c for c in confidences if c < self.confidence_thresholds['low']])
        }
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Process any pending predictions
                if self.prediction_buffer:
                    self._process_prediction_batch()
                
                # Generate and save metrics
                current_metrics = self.get_current_metrics()
                self.metrics_history.append(current_metrics)
                self._append_to_log(self.metrics_log_path, asdict(current_metrics))
                
                # Check for alerts
                self._check_performance_alerts(current_metrics)
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _process_prediction_batch(self):
        """Process batch of predictions and save to log"""
        batch = list(self.prediction_buffer)
        self.prediction_buffer.clear()
        
        # Save batch to log file
        for prediction in batch:
            self._append_to_log(self.predictions_log_path, asdict(prediction))
    
    def _get_recent_predictions(self, minutes: int) -> List[PredictionRecord]:
        """Get predictions from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            p for p in self.recent_predictions 
            if datetime.fromisoformat(p.timestamp) > cutoff_time
        ]
    
    def _calculate_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
        """Calculate confidence distribution buckets"""
        distribution = {
            'very_low': 0,  # < 0.5
            'low': 0,       # 0.5-0.7
            'medium': 0,    # 0.7-0.8
            'high': 0,      # 0.8-0.9
            'very_high': 0  # > 0.9
        }
        
        for confidence in confidences:
            if confidence < 0.5:
                distribution['very_low'] += 1
            elif confidence < 0.7:
                distribution['low'] += 1
            elif confidence < 0.8:
                distribution['medium'] += 1
            elif confidence < 0.9:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
        
        return distribution
    
    def _calculate_prediction_distribution(self, predictions: List[str]) -> Dict[str, int]:
        """Calculate prediction label distribution"""
        distribution = defaultdict(int)
        for prediction in predictions:
            distribution[prediction] += 1
        return dict(distribution)
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not values:
            return {}
        
        return {
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99))
        }
    
    def _calculate_anomaly_score(self, predictions: List[PredictionRecord]) -> float:
        """Calculate anomaly score based on various factors"""
        if not predictions:
            return 0.0
        
        scores = []
        
        # Confidence anomaly (low confidence spike)
        confidences = [p.confidence for p in predictions]
        low_confidence_ratio = len([c for c in confidences if c < 0.6]) / len(confidences)
        scores.append(low_confidence_ratio)
        
        # Processing time anomaly (slow responses)
        processing_times = [p.processing_time for p in predictions]
        slow_response_ratio = len([t for t in processing_times if t > 5.0]) / len(processing_times)
        scores.append(slow_response_ratio)
        
        # Prediction distribution anomaly (extreme skew)
        prediction_dist = self._calculate_prediction_distribution([p.prediction for p in predictions])
        if prediction_dist:
            max_ratio = max(prediction_dist.values()) / len(predictions)
            if max_ratio > 0.9:  # More than 90% same prediction
                scores.append(0.5)
            else:
                scores.append(0.0)
        
        return float(np.mean(scores))
    
    def _detect_anomaly_indicators(self, predictions: List[PredictionRecord]) -> List[str]:
        """Detect specific anomaly indicators"""
        indicators = []
        
        if not predictions:
            return indicators
        
        # Low confidence spike
        low_confidence_count = len([p for p in predictions if p.confidence < 0.6])
        if low_confidence_count > len(predictions) * 0.3:
            indicators.append(f"High low-confidence predictions: {low_confidence_count}/{len(predictions)}")
        
        # Slow response spike
        slow_responses = len([p for p in predictions if p.processing_time > 5.0])
        if slow_responses > len(predictions) * 0.1:
            indicators.append(f"Slow responses detected: {slow_responses}/{len(predictions)}")
        
        # Prediction skew
        prediction_dist = self._calculate_prediction_distribution([p.prediction for p in predictions])
        if prediction_dist:
            max_count = max(prediction_dist.values())
            if max_count > len(predictions) * 0.9:
                dominant_prediction = max(prediction_dist, key=prediction_dist.get)
                indicators.append(f"Extreme prediction skew: {max_count}/{len(predictions)} are '{dominant_prediction}'")
        
        return indicators
    
    def _check_performance_alerts(self, metrics: MonitoringMetrics):
        """Check for performance-based alerts"""
        alerts = []
        
        # Response time alerts
        if metrics.avg_processing_time > self.performance_thresholds['response_time_critical']:
            alerts.append({
                'type': 'critical',
                'category': 'response_time',
                'message': f"Critical response time: {metrics.avg_processing_time:.2f}s",
                'threshold': self.performance_thresholds['response_time_critical']
            })
        elif metrics.avg_processing_time > self.performance_thresholds['response_time_warning']:
            alerts.append({
                'type': 'warning',
                'category': 'response_time',
                'message': f"High response time: {metrics.avg_processing_time:.2f}s",
                'threshold': self.performance_thresholds['response_time_warning']
            })
        
        # Confidence alerts
        if metrics.avg_confidence < self.performance_thresholds['confidence_warning']:
            alerts.append({
                'type': 'warning',
                'category': 'confidence',
                'message': f"Low average confidence: {metrics.avg_confidence:.2f}",
                'threshold': self.performance_thresholds['confidence_warning']
            })
        
        # Error rate alerts
        if metrics.error_rate > self.performance_thresholds['error_rate_critical']:
            alerts.append({
                'type': 'critical',
                'category': 'error_rate',
                'message': f"Critical error rate: {metrics.error_rate:.2%}",
                'threshold': self.performance_thresholds['error_rate_critical']
            })
        elif metrics.error_rate > self.performance_thresholds['error_rate_warning']:
            alerts.append({
                'type': 'warning',
                'category': 'error_rate',
                'message': f"High error rate: {metrics.error_rate:.2%}",
                'threshold': self.performance_thresholds['error_rate_warning']
            })
        
        # Anomaly alerts
        if metrics.anomaly_score > 0.3:
            alerts.append({
                'type': 'warning',
                'category': 'anomaly',
                'message': f"Anomaly detected: score {metrics.anomaly_score:.2f}",
                'threshold': 0.3
            })
        
        # Save alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert['metrics_snapshot'] = asdict(metrics)
            self._append_to_log(self.alerts_log_path, alert)
    
    def _check_error_rate_alerts(self):
        """Check error rate and generate alerts if needed"""
        error_rate = self.get_current_error_rate()
        
        if error_rate > self.performance_thresholds['error_rate_critical']:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'critical',
                'category': 'error_rate',
                'message': f"Critical error rate reached: {error_rate:.2%}",
                'error_count': self.error_count,
                'total_requests': self.total_predictions + self.error_count
            }
            self._append_to_log(self.alerts_log_path, alert)
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _append_to_log(self, log_path: Path, data: Dict):
        """Append data to log file"""
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log {log_path}: {e}")
    
    def load_historical_data(self):
        """Load historical data on startup"""
        try:
            # Load recent predictions
            if self.predictions_log_path.exists():
                with open(self.predictions_log_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            prediction = PredictionRecord(**data)
                            # Only load recent predictions (last 24 hours)
                            if datetime.fromisoformat(prediction.timestamp) > datetime.now() - timedelta(hours=24):
                                self.recent_predictions.append(prediction)
                        except Exception:
                            continue
            
            # Load recent metrics
            if self.metrics_log_path.exists():
                with open(self.metrics_log_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            metrics = MonitoringMetrics(**data)
                            # Only load recent metrics (last 24 hours)
                            if datetime.fromisoformat(metrics.timestamp) > datetime.now() - timedelta(hours=24):
                                self.metrics_history.append(metrics)
                        except Exception:
                            continue
            
            logger.info(f"Loaded {len(self.recent_predictions)} recent predictions and {len(self.metrics_history)} metrics records")
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")