import json
import time
import smtplib
import logging
import numpy as np
from pathlib import Path
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: str
    type: str  # 'info', 'warning', 'critical'
    category: str  # 'system', 'api', 'model', 'prediction'
    title: str
    message: str
    source: str
    severity_score: float
    metadata: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[str] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    category: str
    condition: Dict[str, Any]
    threshold: float
    severity: str
    cooldown_minutes: int
    enabled: bool = True

class AlertSystem:
    """Comprehensive alerting and notification system"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.monitor_dir = self.base_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.alerts_log_path = self.monitor_dir / "alerts.json"
        self.alert_rules_path = self.monitor_dir / "alert_rules.json"
        self.alert_config_path = self.monitor_dir / "alert_config.json"
        
        # In-memory storage
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=10000)
        self.alert_rules = {}  # rule_id -> AlertRule
        self.alert_cooldowns = defaultdict(float)  # rule_id -> last_triggered_time
        
        # Notification channels
        self.notification_handlers = {}
        
        # Alert statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': defaultdict(int),
            'alerts_by_category': defaultdict(int),
            'resolution_times': []
        }
        
        # Load configuration and rules
        self.load_alert_configuration()
        self.load_alert_rules()
        self.load_alert_history()
    
    def add_notification_handler(self, name: str, handler: Callable):
        """Add a custom notification handler"""
        self.notification_handlers[name] = handler
        logger.info(f"Added notification handler: {name}")
    
    def create_alert(self, 
                    alert_type: str,
                    category: str,
                    title: str,
                    message: str,
                    source: str,
                    metadata: Dict[str, Any] = None,
                    severity_score: float = 0.5) -> str:
        """Create a new alert"""
        
        alert_id = self._generate_alert_id(category, title)
        
        # Check if similar alert already exists
        if self._is_duplicate_alert(alert_id, category, title):
            logger.debug(f"Duplicate alert suppressed: {title}")
            return alert_id
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now().isoformat(),
            type=alert_type,
            category=category,
            title=title,
            message=message,
            source=source,
            severity_score=severity_score,
            metadata=metadata or {},
            acknowledged=False,
            resolved=False
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['alerts_by_type'][alert_type] += 1
        self.alert_stats['alerts_by_category'][category] += 1
        
        # Save to log
        self._append_to_log(self.alerts_log_path, asdict(alert))
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"Created {alert_type} alert: {title}")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledger: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.metadata['acknowledged_by'] = acknowledger
            alert.metadata['acknowledged_at'] = datetime.now().isoformat()
            
            self._append_to_log(self.alerts_log_path, {
                'action': 'acknowledge',
                'alert_id': alert_id,
                'acknowledger': acknowledger,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledger}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolver: str = "system", resolution_note: str = "") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now().isoformat()
            alert.metadata['resolved_by'] = resolver
            alert.metadata['resolution_note'] = resolution_note
            
            # Calculate resolution time
            alert_time = datetime.fromisoformat(alert.timestamp)
            resolution_time = datetime.now()
            resolution_duration = (resolution_time - alert_time).total_seconds() / 60  # minutes
            
            self.alert_stats['resolution_times'].append(resolution_duration)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self._append_to_log(self.alerts_log_path, {
                'action': 'resolve',
                'alert_id': alert_id,
                'resolver': resolver,
                'resolution_note': resolution_note,
                'resolution_duration_minutes': resolution_duration,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Alert resolved: {alert_id} by {resolver}")
            return True
        
        return False
    
    def check_metric_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_id, rule.cooldown_minutes):
                continue
            
            # Evaluate rule condition
            if self._evaluate_rule_condition(rule, metrics):
                self._trigger_rule_alert(rule, metrics)
    
    def check_anomaly_detection(self, 
                               current_metrics: Dict[str, Any],
                               historical_metrics: List[Dict[str, Any]]):
        """Check for anomalies using statistical methods"""
        
        if len(historical_metrics) < 10:  # Need sufficient history
            return
        
        # Define metrics to monitor for anomalies
        anomaly_metrics = {
            'response_time': 'api.avg_response_time',
            'error_rate': 'api.error_rate',
            'cpu_usage': 'system.cpu_percent',
            'memory_usage': 'system.memory_percent',
            'confidence': 'model.avg_confidence'
        }
        
        for metric_name, metric_path in anomaly_metrics.items():
            try:
                # Extract historical values
                historical_values = []
                for hist_metric in historical_metrics:
                    value = self._get_nested_value(hist_metric, metric_path)
                    if value is not None:
                        historical_values.append(value)
                
                if len(historical_values) < 5:
                    continue
                
                # Get current value
                current_value = self._get_nested_value(current_metrics, metric_path)
                if current_value is None:
                    continue
                
                # Statistical anomaly detection
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                
                # Z-score based detection
                if std_val > 0:
                    z_score = abs(current_value - mean_val) / std_val
                    
                    if z_score > 3:  # 3 sigma threshold
                        self.create_alert(
                            alert_type='warning',
                            category='anomaly',
                            title=f'Anomaly Detected: {metric_name}',
                            message=f'{metric_name} value {current_value:.2f} is {z_score:.1f} standard deviations from normal',
                            source='anomaly_detection',
                            metadata={
                                'metric_name': metric_name,
                                'current_value': current_value,
                                'historical_mean': mean_val,
                                'historical_std': std_val,
                                'z_score': z_score
                            },
                            severity_score=min(z_score / 5, 1.0)
                        )
                
            except Exception as e:
                logger.error(f"Error in anomaly detection for {metric_name}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alerts_by_category(self, category: str, hours: int = 24) -> List[Alert]:
        """Get alerts by category within time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if (alert.category == category and 
                datetime.fromisoformat(alert.timestamp) > cutoff_time)
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and metrics"""
        active_count = len(self.active_alerts)
        
        # Recent alerts (last 24 hours)
        recent_alerts = self.get_recent_alerts(hours=24)
        
        # Resolution time statistics
        resolution_times = self.alert_stats['resolution_times']
        resolution_stats = {}
        if resolution_times:
            resolution_stats = {
                'avg_resolution_time_minutes': float(np.mean(resolution_times)),
                'median_resolution_time_minutes': float(np.median(resolution_times)),
                'max_resolution_time_minutes': float(np.max(resolution_times)),
                'min_resolution_time_minutes': float(np.min(resolution_times))
            }
        
        return {
            'active_alerts': active_count,
            'total_alerts_24h': len(recent_alerts),
            'alerts_by_type': dict(self.alert_stats['alerts_by_type']),
            'alerts_by_category': dict(self.alert_stats['alerts_by_category']),
            'resolution_statistics': resolution_stats,
            'alert_rate_per_hour': len(recent_alerts) / 24.0,
            'critical_alerts_active': len([a for a in self.active_alerts.values() if a.type == 'critical']),
            'unacknowledged_alerts': len([a for a in self.active_alerts.values() if not a.acknowledged])
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts within time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        ]
    
    def create_default_alert_rules(self):
        """Create default alert rules"""
        default_rules = [
            {
                'id': 'high_response_time',
                'name': 'High Response Time',
                'category': 'api',
                'condition': {'metric': 'avg_response_time', 'operator': '>', 'value': 5.0},
                'threshold': 5.0,
                'severity': 'warning',
                'cooldown_minutes': 5
            },
            {
                'id': 'critical_response_time',
                'name': 'Critical Response Time',
                'category': 'api',
                'condition': {'metric': 'avg_response_time', 'operator': '>', 'value': 10.0},
                'threshold': 10.0,
                'severity': 'critical',
                'cooldown_minutes': 2
            },
            {
                'id': 'high_error_rate',
                'name': 'High Error Rate',
                'category': 'api',
                'condition': {'metric': 'error_rate', 'operator': '>', 'value': 0.05},
                'threshold': 0.05,
                'severity': 'warning',
                'cooldown_minutes': 5
            },
            {
                'id': 'critical_error_rate',
                'name': 'Critical Error Rate',
                'category': 'api',
                'condition': {'metric': 'error_rate', 'operator': '>', 'value': 0.1},
                'threshold': 0.1,
                'severity': 'critical',
                'cooldown_minutes': 2
            },
            {
                'id': 'high_cpu_usage',
                'name': 'High CPU Usage',
                'category': 'system',
                'condition': {'metric': 'cpu_percent', 'operator': '>', 'value': 80.0},
                'threshold': 80.0,
                'severity': 'warning',
                'cooldown_minutes': 10
            },
            {
                'id': 'critical_cpu_usage',
                'name': 'Critical CPU Usage',
                'category': 'system',
                'condition': {'metric': 'cpu_percent', 'operator': '>', 'value': 95.0},
                'threshold': 95.0,
                'severity': 'critical',
                'cooldown_minutes': 5
            },
            {
                'id': 'high_memory_usage',
                'name': 'High Memory Usage',
                'category': 'system',
                'condition': {'metric': 'memory_percent', 'operator': '>', 'value': 85.0},
                'threshold': 85.0,
                'severity': 'warning',
                'cooldown_minutes': 10
            },
            {
                'id': 'low_model_confidence',
                'name': 'Low Model Confidence',
                'category': 'model',
                'condition': {'metric': 'avg_confidence', 'operator': '<', 'value': 0.6},
                'threshold': 0.6,
                'severity': 'warning',
                'cooldown_minutes': 15
            }
        ]
        
        for rule_data in default_rules:
            rule = AlertRule(**rule_data)
            self.alert_rules[rule.id] = rule
        
        self.save_alert_rules()
        logger.info(f"Created {len(default_rules)} default alert rules")
    
    def _generate_alert_id(self, category: str, title: str) -> str:
        """Generate unique alert ID"""
        import hashlib
        content = f"{category}_{title}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _is_duplicate_alert(self, alert_id: str, category: str, title: str, window_minutes: int = 10) -> bool:
        """Check if similar alert exists within time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        for alert in self.alert_history:
            if (alert.category == category and 
                alert.title == title and 
                datetime.fromisoformat(alert.timestamp) > cutoff_time and
                not alert.resolved):
                return True
        
        return False
    
    def _is_in_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.alert_cooldowns:
            return False
        
        last_triggered = self.alert_cooldowns[rule_id]
        cooldown_period = cooldown_minutes * 60  # Convert to seconds
        
        return (time.time() - last_triggered) < cooldown_period
    
    def _evaluate_rule_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate if rule condition is met"""
        try:
            condition = rule.condition
            metric_value = self._get_nested_value(metrics, condition['metric'])
            
            if metric_value is None:
                return False
            
            operator = condition['operator']
            threshold_value = condition['value']
            
            if operator == '>':
                return metric_value > threshold_value
            elif operator == '<':
                return metric_value < threshold_value
            elif operator == '>=':
                return metric_value >= threshold_value
            elif operator == '<=':
                return metric_value <= threshold_value
            elif operator == '==':
                return metric_value == threshold_value
            elif operator == '!=':
                return metric_value != threshold_value
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition for {rule.id}: {e}")
            return False
    
    def _trigger_rule_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger alert based on rule"""
        metric_value = self._get_nested_value(metrics, rule.condition['metric'])
        
        alert_id = self.create_alert(
            alert_type=rule.severity,
            category=rule.category,
            title=rule.name,
            message=f"{rule.name}: {rule.condition['metric']} = {metric_value} (threshold: {rule.threshold})",
            source=f"rule_{rule.id}",
            metadata={
                'rule_id': rule.id,
                'metric_name': rule.condition['metric'],
                'metric_value': metric_value,
                'threshold': rule.threshold,
                'operator': rule.condition['operator']
            }
        )
        
        # Set cooldown
        self.alert_cooldowns[rule.id] = time.time()
        
        logger.info(f"Rule alert triggered: {rule.name} (ID: {alert_id})")
    
    def _get_nested_value(self, data: Dict, path: str):
        """Get nested value from dictionary using dot notation"""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception:
            return None
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        for handler_name, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler {handler_name}: {e}")
    
    def _append_to_log(self, log_path: Path, data: Dict):
        """Append data to log file"""
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log {log_path}: {e}")
    
    def load_alert_configuration(self):
        """Load alert system configuration"""
        try:
            if self.alert_config_path.exists():
                with open(self.alert_config_path, 'r') as f:
                    config = json.load(f)
                
                # Update notification settings, thresholds, etc.
                logger.info("Loaded alert configuration")
            else:
                # Create default configuration
                default_config = {
                    'notification_channels': ['console'],
                    'alert_retention_days': 30,
                    'auto_resolve_after_hours': 24,
                    'duplicate_suppression_minutes': 10
                }
                
                with open(self.alert_config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                logger.info("Created default alert configuration")
                
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}")
    
    def load_alert_rules(self):
        """Load alert rules from file"""
        try:
            if self.alert_rules_path.exists():
                with open(self.alert_rules_path, 'r') as f:
                    rules_data = json.load(f)
                
                for rule_id, rule_data in rules_data.items():
                    rule = AlertRule(**rule_data)
                    self.alert_rules[rule_id] = rule
                
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
            else:
                # Create default rules
                self.create_default_alert_rules()
                
        except Exception as e:
            logger.error(f"Failed to load alert rules: {e}")
            # Create default rules as fallback
            self.create_default_alert_rules()
    
    def save_alert_rules(self):
        """Save alert rules to file"""
        try:
            rules_data = {}
            for rule_id, rule in self.alert_rules.items():
                rules_data[rule_id] = asdict(rule)
            
            with open(self.alert_rules_path, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Saved {len(self.alert_rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Failed to save alert rules: {e}")
    
    def load_alert_history(self):
        """Load recent alert history"""
        try:
            if self.alerts_log_path.exists():
                cutoff_time = datetime.now() - timedelta(days=7)  # Last 7 days
                
                with open(self.alerts_log_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            
                            # Skip action logs
                            if 'action' in data:
                                continue
                            
                            alert = Alert(**data)
                            
                            # Only load recent alerts
                            if datetime.fromisoformat(alert.timestamp) > cutoff_time:
                                self.alert_history.append(alert)
                                
                                # Add to active alerts if not resolved
                                if not alert.resolved:
                                    self.active_alerts[alert.id] = alert
                        
                        except Exception:
                            continue
                
                logger.info(f"Loaded {len(self.alert_history)} recent alerts, "
                           f"{len(self.active_alerts)} active")
                
        except Exception as e:
            logger.error(f"Failed to load alert history: {e}")

# Default notification handlers
def console_notification_handler(alert: Alert):
    """Simple console notification handler"""
    icon = "🔴" if alert.type == "critical" else "🟡" if alert.type == "warning" else "🔵"
    print(f"{icon} [{alert.type.upper()}] {alert.title}: {alert.message}")

def email_notification_handler(alert: Alert, 
                              smtp_server: str,
                              smtp_port: int,
                              username: str,
                              password: str,
                              recipients: List[str]):
    """Email notification handler"""
    try:
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[{alert.type.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Type: {alert.type}
- Category: {alert.category}
- Timestamp: {alert.timestamp}
- Source: {alert.source}
- Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        text = msg.as_string()
        server.sendmail(username, recipients, text)
        server.quit()
        
        logger.info(f"Email notification sent for alert: {alert.id}")
        
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")