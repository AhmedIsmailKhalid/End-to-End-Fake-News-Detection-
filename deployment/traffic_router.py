import json
import time
import random
import joblib
import logging
import hashlib
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    HASH_BASED = "hash_based"
    CANARY = "canary"
    A_B_TEST = "a_b_test"

@dataclass
class RoutingRule:
    """Traffic routing rule configuration"""
    rule_id: str
    strategy: str
    weights: Dict[str, int]  # environment -> percentage
    conditions: Dict[str, Any]
    active: bool
    created_at: str
    updated_at: str

@dataclass
class RequestMetrics:
    """Metrics for individual requests"""
    request_id: str
    timestamp: str
    environment: str  # blue or green
    response_time: float
    status_code: int
    confidence: Optional[float]
    prediction: Optional[str]
    client_id: Optional[str]
    user_agent: Optional[str]

class TrafficRouter:
    """Intelligent traffic routing for blue-green deployments"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/tmp")
        self.setup_router_paths()
        self.setup_router_config()
        
        # Current routing state
        self.current_routing_rule = None
        self.blue_model = None
        self.green_model = None
        self.blue_vectorizer = None
        self.green_vectorizer = None
        
        # Performance tracking
        self.request_metrics = []
        self.performance_cache = {}
        
        # Load models and routing state
        self.load_routing_state()
        self.load_models()
    
    def setup_router_paths(self):
        """Setup traffic router paths"""
        self.router_dir = self.base_dir / "deployment" / "router"
        self.router_dir.mkdir(parents=True, exist_ok=True)
        
        # Router state files
        self.routing_state_path = self.router_dir / "routing_state.json"
        self.routing_rules_path = self.router_dir / "routing_rules.json"
        self.request_log_path = self.router_dir / "request_log.json"
        self.performance_log_path = self.router_dir / "performance_log.json"
        
        # Model environment paths
        self.blue_model_dir = self.base_dir / "deployment" / "models" / "blue"
        self.green_model_dir = self.base_dir / "deployment" / "models" / "green"
    
    def setup_router_config(self):
        """Setup router configuration"""
        self.router_config = {
            'default_routing': {
                'strategy': RoutingStrategy.WEIGHTED.value,
                'blue_weight': 100,
                'green_weight': 0
            },
            'performance_tracking': {
                'enable_metrics': True,
                'metrics_buffer_size': 10000,
                'performance_window_minutes': 60,
                'cache_performance_seconds': 30
            },
            'routing_decisions': {
                'hash_based_header': 'user-agent',
                'canary_user_percentage': 5,
                'a_b_test_hash_field': 'client_id',
                'sticky_sessions': False
            },
            'health_checks': {
                'enable_health_routing': True,
                'unhealthy_weight': 0,
                'health_check_interval': 30
            }
        }
    
    def set_routing_weights(self, blue_weight: int, green_weight: int) -> bool:
        """Set traffic routing weights"""
        try:
            # Normalize weights to percentages
            total_weight = blue_weight + green_weight
            if total_weight == 0:
                raise ValueError("Total weight cannot be zero")
            
            blue_percentage = int((blue_weight / total_weight) * 100)
            green_percentage = 100 - blue_percentage
            
            # Create or update routing rule
            routing_rule = RoutingRule(
                rule_id=f"weight_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy=RoutingStrategy.WEIGHTED.value,
                weights={'blue': blue_percentage, 'green': green_percentage},
                conditions={},
                active=True,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            self.current_routing_rule = routing_rule
            self.save_routing_state()
            
            self.log_routing_event("weights_updated", f"Updated routing weights: Blue {blue_percentage}%, Green {green_percentage}%", {
                'blue_weight': blue_percentage,
                'green_weight': green_percentage
            })
            
            logger.info(f"Updated routing weights: Blue {blue_percentage}%, Green {green_percentage}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set routing weights: {e}")
            return False
    
    def route_request(self, request_data: Dict[str, Any]) -> str:
        """Route a request to blue or green environment"""
        try:
            if not self.current_routing_rule:
                # Default to blue if no routing rule
                return "blue"
            
            strategy = self.current_routing_rule.strategy
            
            if strategy == RoutingStrategy.WEIGHTED.value:
                return self._route_weighted(request_data)
            elif strategy == RoutingStrategy.ROUND_ROBIN.value:
                return self._route_round_robin(request_data)
            elif strategy == RoutingStrategy.HASH_BASED.value:
                return self._route_hash_based(request_data)
            elif strategy == RoutingStrategy.CANARY.value:
                return self._route_canary(request_data)
            elif strategy == RoutingStrategy.A_B_TEST.value:
                return self._route_a_b_test(request_data)
            else:
                return "blue"  # Default fallback
                
        except Exception as e:
            logger.error(f"Routing decision failed: {e}")
            return "blue"  # Safe fallback
    
    def _route_weighted(self, request_data: Dict[str, Any]) -> str:
        """Route based on weighted distribution"""
        weights = self.current_routing_rule.weights
        blue_weight = weights.get('blue', 100)
        green_weight = weights.get('green', 0)
        
        # Generate random number 0-99
        random_num = random.randint(0, 99)
        
        # Route to green if random number is less than green weight
        if random_num < green_weight:
            return "green"
        else:
            return "blue"
    
    def _route_round_robin(self, request_data: Dict[str, Any]) -> str:
        """Route using round-robin algorithm"""
        # Simple counter-based round robin
        request_count = len(self.request_metrics)
        weights = self.current_routing_rule.weights
        
        blue_weight = weights.get('blue', 50)
        green_weight = weights.get('green', 50)
        
        # Calculate cycle length based on weights
        cycle_length = blue_weight + green_weight
        position_in_cycle = request_count % cycle_length
        
        if position_in_cycle < blue_weight:
            return "blue"
        else:
            return "green"
    
    def _route_hash_based(self, request_data: Dict[str, Any]) -> str:
        """Route based on hash of request characteristics"""
    def _route_hash_based(self, request_data: Dict[str, Any]) -> str:
        """Route based on hash of request characteristics"""
        hash_field = self.router_config['routing_decisions']['hash_based_header']
        hash_value = request_data.get(hash_field, 'default')
        
        # Generate hash
        hash_digest = hashlib.md5(str(hash_value).encode()).hexdigest()
        hash_int = int(hash_digest[:8], 16)
        
        weights = self.current_routing_rule.weights
        green_weight = weights.get('green', 0)
        
        # Route based on hash modulo
        if (hash_int % 100) < green_weight:
            return "green"
        else:
            return "blue"
    
    def _route_canary(self, request_data: Dict[str, Any]) -> str:
        """Route canary traffic to green environment"""
        canary_percentage = self.router_config['routing_decisions']['canary_user_percentage']
        
        # Use client ID or user agent for consistent canary routing
        client_id = request_data.get('client_id') or request_data.get('user_agent', 'anonymous')
        hash_digest = hashlib.md5(client_id.encode()).hexdigest()
        hash_int = int(hash_digest[:8], 16)
        
        if (hash_int % 100) < canary_percentage:
            return "green"  # Canary users get green
        else:
            return "blue"   # Regular users get blue
    
    def _route_a_b_test(self, request_data: Dict[str, Any]) -> str:
        """Route for A/B testing"""
        hash_field = self.router_config['routing_decisions']['a_b_test_hash_field']
        hash_value = request_data.get(hash_field, request_data.get('user_agent', 'default'))
        
        # Generate consistent hash for A/B testing
        hash_digest = hashlib.md5(str(hash_value).encode()).hexdigest()
        hash_int = int(hash_digest[:8], 16)
        
        weights = self.current_routing_rule.weights
        green_weight = weights.get('green', 50)  # Default 50/50 for A/B test
        
        if (hash_int % 100) < green_weight:
            return "green"
        else:
            return "blue"
    
    def make_prediction(self, text: str, request_data: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using routed model"""
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        
        try:
            # Determine routing
            if request_data is None:
                request_data = {}
            
            environment = self.route_request(request_data)
            
            # Get appropriate model and vectorizer
            if environment == "green" and self.green_model and self.green_vectorizer:
                model = self.green_model
                vectorizer = self.green_vectorizer
            else:
                # Fallback to blue
                environment = "blue"
                model = self.blue_model
                vectorizer = self.blue_vectorizer
            
            if not model or not vectorizer:
                raise ValueError(f"No model available for {environment} environment")
            
            # Make prediction
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = float(max(probabilities))
            
            # Convert prediction to readable format
            label = "Fake" if prediction == 1 else "Real"
            
            processing_time = time.time() - start_time
            
            # Record metrics
            self.record_request_metrics(
                request_id=request_id,
                environment=environment,
                response_time=processing_time,
                status_code=200,
                confidence=confidence,
                prediction=label,
                client_id=request_data.get('client_id'),
                user_agent=request_data.get('user_agent')
            )
            
            result = {
                'prediction': label,
                'confidence': confidence,
                'processing_time': processing_time,
                'environment': environment,
                'request_id': request_id,
                'model_version': 'unknown',  # Could be enhanced with version info
                'timestamp': datetime.now().isoformat()
            }
            
            return environment, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            self.record_request_metrics(
                request_id=request_id,
                environment=environment if 'environment' in locals() else 'unknown',
                response_time=processing_time,
                status_code=500,
                confidence=None,
                prediction=None,
                client_id=request_data.get('client_id'),
                user_agent=request_data.get('user_agent')
            )
            
            logger.error(f"Prediction failed: {e}")
            raise e
    
    def record_request_metrics(self, request_id: str, environment: str, 
                             response_time: float, status_code: int,
                             confidence: Optional[float] = None,
                             prediction: Optional[str] = None,
                             client_id: Optional[str] = None,
                             user_agent: Optional[str] = None):
        """Record metrics for a request"""
        try:
            metrics = RequestMetrics(
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                environment=environment,
                response_time=response_time,
                status_code=status_code,
                confidence=confidence,
                prediction=prediction,
                client_id=client_id,
                user_agent=user_agent
            )
            
            self.request_metrics.append(metrics)
            
            # Keep buffer size manageable
            buffer_size = self.router_config['performance_tracking']['metrics_buffer_size']
            if len(self.request_metrics) > buffer_size:
                self.request_metrics = self.request_metrics[-buffer_size:]
            
            # Log to file periodically
            if len(self.request_metrics) % 100 == 0:
                self.save_request_metrics()
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
    
    def get_environment_performance(self, environment: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for an environment"""
        try:
            # Check cache first
            cache_key = f"{environment}_{window_minutes}"
            cache_timeout = self.router_config['performance_tracking']['cache_performance_seconds']
            
            if (cache_key in self.performance_cache and 
                time.time() - self.performance_cache[cache_key]['cached_at'] < cache_timeout):
                return self.performance_cache[cache_key]['data']
            
            # Calculate performance from recent metrics
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            
            relevant_metrics = [
                m for m in self.request_metrics
                if (m.environment == environment and 
                    datetime.fromisoformat(m.timestamp) > cutoff_time)
            ]
            
            if not relevant_metrics:
                return {
                    'environment': environment,
                    'window_minutes': window_minutes,
                    'request_count': 0,
                    'avg_response_time': 0,
                    'error_rate': 0,
                    'avg_confidence': 0,
                    'requests_per_minute': 0
                }
            
            # Calculate metrics
            response_times = [m.response_time for m in relevant_metrics]
            error_count = len([m for m in relevant_metrics if m.status_code >= 400])
            confidences = [m.confidence for m in relevant_metrics if m.confidence is not None]
            
            performance = {
                'environment': environment,
                'window_minutes': window_minutes,
                'request_count': len(relevant_metrics),
                'avg_response_time': sum(response_times) / len(response_times),
                'error_rate': error_count / len(relevant_metrics),
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'requests_per_minute': len(relevant_metrics) / window_minutes,
                'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                'successful_requests': len(relevant_metrics) - error_count
            }
            
            # Cache result
            self.performance_cache[cache_key] = {
                'data': performance,
                'cached_at': time.time()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get environment performance: {e}")
            return {'error': str(e)}
    
    def compare_environment_performance(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Compare performance between blue and green environments"""
        try:
            blue_perf = self.get_environment_performance('blue', window_minutes)
            green_perf = self.get_environment_performance('green', window_minutes)
            
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'window_minutes': window_minutes,
                'blue_performance': blue_perf,
                'green_performance': green_perf,
                'comparison': {}
            }
            
            if blue_perf.get('request_count', 0) > 0 and green_perf.get('request_count', 0) > 0:
                # Calculate relative differences
                comparison['comparison'] = {
                    'response_time_diff': green_perf['avg_response_time'] - blue_perf['avg_response_time'],
                    'error_rate_diff': green_perf['error_rate'] - blue_perf['error_rate'],
                    'confidence_diff': green_perf['avg_confidence'] - blue_perf['avg_confidence'],
                    'traffic_distribution': {
                        'blue_percentage': (blue_perf['request_count'] / (blue_perf['request_count'] + green_perf['request_count'])) * 100,
                        'green_percentage': (green_perf['request_count'] / (blue_perf['request_count'] + green_perf['request_count'])) * 100
                    }
                }
                
                # Add recommendations
                recommendations = []
                if green_perf['error_rate'] > blue_perf['error_rate'] * 1.5:
                    recommendations.append("Green environment has significantly higher error rate")
                if green_perf['avg_response_time'] > blue_perf['avg_response_time'] * 1.5:
                    recommendations.append("Green environment has significantly slower response times")
                if green_perf['avg_confidence'] < blue_perf['avg_confidence'] * 0.9:
                    recommendations.append("Green environment has lower prediction confidence")
                
                comparison['recommendations'] = recommendations
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare environment performance: {e}")
            return {'error': str(e)}
    
    def load_models(self):
        """Load models for both environments"""
        try:
            # Load blue environment
            blue_model_path = self.blue_model_dir / "model.pkl"
            blue_vectorizer_path = self.blue_model_dir / "vectorizer.pkl"
            
            if blue_model_path.exists() and blue_vectorizer_path.exists():
                self.blue_model = joblib.load(blue_model_path)
                self.blue_vectorizer = joblib.load(blue_vectorizer_path)
                logger.info("Loaded blue environment models")
            
            # Load green environment
            green_model_path = self.green_model_dir / "model.pkl"
            green_vectorizer_path = self.green_model_dir / "vectorizer.pkl"
            
            if green_model_path.exists() and green_vectorizer_path.exists():
                self.green_model = joblib.load(green_model_path)
                self.green_vectorizer = joblib.load(green_vectorizer_path)
                logger.info("Loaded green environment models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def get_routing_status(self) -> Dict[str, Any]:
        """Get current routing status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'current_routing_rule': asdict(self.current_routing_rule) if self.current_routing_rule else None,
                'environment_status': {
                    'blue': {
                        'model_loaded': self.blue_model is not None,
                        'vectorizer_loaded': self.blue_vectorizer is not None
                    },
                    'green': {
                        'model_loaded': self.green_model is not None,
                        'vectorizer_loaded': self.green_vectorizer is not None
                    }
                },
                'recent_performance': {
                    'blue': self.get_environment_performance('blue', 15),
                    'green': self.get_environment_performance('green', 15)
                },
                'traffic_distribution': self._get_recent_traffic_distribution()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get routing status: {e}")
            return {'error': str(e)}
    
    def _get_recent_traffic_distribution(self) -> Dict[str, Any]:
        """Get recent traffic distribution"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=15)
            recent_metrics = [
                m for m in self.request_metrics
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'blue': 0, 'green': 0, 'total': 0}
            
            blue_count = len([m for m in recent_metrics if m.environment == 'blue'])
            green_count = len([m for m in recent_metrics if m.environment == 'green'])
            total_count = len(recent_metrics)
            
            return {
                'blue': blue_count,
                'green': green_count,
                'total': total_count,
                'blue_percentage': (blue_count / total_count) * 100 if total_count > 0 else 0,
                'green_percentage': (green_count / total_count) * 100 if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get traffic distribution: {e}")
            return {'error': str(e)}
    
    def save_routing_state(self):
        """Save current routing state"""
        try:
            state = {
                'current_routing_rule': asdict(self.current_routing_rule) if self.current_routing_rule else None,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.routing_state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save routing state: {e}")
    
    def load_routing_state(self):
        """Load routing state from file"""
        try:
            if self.routing_state_path.exists():
                with open(self.routing_state_path, 'r') as f:
                    state = json.load(f)
                
                if state.get('current_routing_rule'):
                    self.current_routing_rule = RoutingRule(**state['current_routing_rule'])
                
                logger.info("Loaded routing state from file")
            else:
                # Set default routing rule
                self.set_routing_weights(100, 0)  # Default to 100% blue
            
        except Exception as e:
            logger.warning(f"Failed to load routing state: {e}")
            # Set default routing rule
            self.set_routing_weights(100, 0)
    
    def save_request_metrics(self):
        """Save request metrics to file"""
        try:
            # Save last 1000 metrics
            metrics_to_save = self.request_metrics[-1000:]
            metrics_data = [asdict(m) for m in metrics_to_save]
            
            with open(self.request_log_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save request metrics: {e}")
    
    def log_routing_event(self, event: str, message: str, details: Dict = None):
        """Log routing events"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'message': message,
                'details': details or {}
            }
            
            # This could be enhanced to save to a separate routing events log
            logger.info(f"Routing event: {event} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to log routing event: {e}")
    
    def cleanup_old_metrics(self, days: int = 7):
        """Clean up old metrics data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent metrics
            self.request_metrics = [
                m for m in self.request_metrics
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            # Clear performance cache
            self.performance_cache.clear()
            
            logger.info(f"Cleaned up metrics older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")