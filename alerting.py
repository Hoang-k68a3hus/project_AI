"""
Alerting System for CF Recommendation Service.

This module provides alerting functionality for:
- Service health alerts (latency, error rate)
- Data drift alerts
- Model performance degradation
- BERT embedding staleness

Usage:
    >>> from recsys.cf.alerting import AlertManager
    >>> alert_mgr = AlertManager()
    >>> alert_mgr.send_alert("High Latency", "P95 latency exceeded 200ms", "warning")
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import yaml
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_ALERT_CONFIG_PATH = "config/alerts_config.yaml"
ALERT_LOG_PATH = "logs/service/alerts.log"


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """
    Manager for sending alerts via email, Slack, and logging.
    
    Example:
        >>> mgr = AlertManager()
        >>> mgr.send_alert(
        ...     subject="High Latency Alert",
        ...     message="P95 latency is 250ms, threshold is 200ms",
        ...     severity="warning"
        ... )
    """
    
    def __init__(self, config_path: str = DEFAULT_ALERT_CONFIG_PATH):
        """
        Initialize AlertManager.
        
        Args:
            config_path: Path to alerts config YAML
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_alert_logger()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration."""
        path = Path(self.config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Alert config not found at {self.config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'alerts': [],
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender': 'alerts@example.com',
                'recipients': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': None
            },
            'logging': {
                'enabled': True
            }
        }
    
    def _setup_alert_logger(self):
        """Setup dedicated alert logger."""
        Path(ALERT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        self.alert_logger = logging.getLogger('cf.alerts')
        self.alert_logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(ALERT_LOG_PATH, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))
        self.alert_logger.addHandler(fh)
    
    def send_alert(
        self,
        subject: str,
        message: str,
        severity: str = 'info',
        metadata: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Send alert via configured channels.
        
        Args:
            subject: Alert subject/title
            message: Alert message body
            severity: 'info', 'warning', 'critical'
            metadata: Additional metadata
        
        Returns:
            Dict with success status for each channel
        """
        results = {}
        
        # Always log
        if self.config.get('logging', {}).get('enabled', True):
            self._log_alert(subject, message, severity, metadata)
            results['log'] = True
        
        # Email for warning and critical
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False) and severity in ['warning', 'critical']:
            try:
                self._send_email(subject, message, severity, email_config)
                results['email'] = True
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
                results['email'] = False
        
        # Slack for warning and critical
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False) and severity in ['warning', 'critical']:
            try:
                self._send_slack(subject, message, severity, slack_config)
                results['slack'] = True
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
                results['slack'] = False
        
        return results
    
    def _log_alert(
        self,
        subject: str,
        message: str,
        severity: str,
        metadata: Optional[Dict] = None
    ):
        """Log alert to file."""
        log_entry = {
            'subject': subject,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.ERROR
        }.get(severity, logging.INFO)
        
        self.alert_logger.log(level, json.dumps(log_entry, ensure_ascii=False))
    
    def _send_email(
        self,
        subject: str,
        message: str,
        severity: str,
        email_config: Dict
    ):
        """Send email alert."""
        sender = email_config.get('sender')
        recipients = email_config.get('recipients', [])
        
        if not recipients:
            logger.warning("No email recipients configured")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = f"[{severity.upper()}] {subject}"
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        # Body
        body = f"""
CF Recommendation Service Alert
================================

Severity: {severity.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

---
This is an automated alert from the CF Recommendation Service.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Send
        smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = email_config.get('smtp_port', 587)
        password = os.getenv('EMAIL_PASSWORD')
        
        if not password:
            logger.warning("EMAIL_PASSWORD environment variable not set")
            return
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        
        logger.info(f"Email alert sent to {recipients}")
    
    def _send_slack(
        self,
        subject: str,
        message: str,
        severity: str,
        slack_config: Dict
    ):
        """Send Slack alert."""
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        # Color based on severity
        color = {
            'info': '#36a64f',
            'warning': '#ff9800',
            'critical': '#f44336'
        }.get(severity, '#36a64f')
        
        # Slack payload
        payload = {
            'attachments': [{
                'color': color,
                'title': f"[{severity.upper()}] {subject}",
                'text': message,
                'footer': 'CF Recommendation Service',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        logger.info("Slack alert sent")
    
    def check_alert_conditions(
        self,
        metrics: Dict[str, float],
        model_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Check alert conditions and send alerts if thresholds exceeded.
        
        Args:
            metrics: Current metrics dict
            model_id: Current model ID
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for alert_config in self.config.get('alerts', []):
            metric_name = alert_config.get('metric')
            threshold = alert_config.get('threshold')
            severity = alert_config.get('severity', 'warning')
            
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            
            # Check threshold
            exceeded = False
            if isinstance(threshold, bool):
                exceeded = (value == threshold)
            elif isinstance(threshold, (int, float)):
                exceeded = (value > threshold)
            
            if exceeded:
                alert_name = alert_config.get('name', metric_name)
                message = f"{metric_name} = {value:.3f} exceeds threshold {threshold}"
                
                self.send_alert(
                    subject=alert_name,
                    message=message,
                    severity=severity,
                    metadata={'model_id': model_id, 'metric': metric_name, 'value': value}
                )
                
                triggered.append({
                    'name': alert_name,
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'severity': severity
                })
        
        return triggered


# ============================================================================
# Alert Conditions
# ============================================================================

class AlertCondition:
    """Base class for alert conditions."""
    
    def __init__(self, name: str, threshold: float, severity: str = 'warning'):
        self.name = name
        self.threshold = threshold
        self.severity = severity
    
    def check(self, value: float) -> bool:
        """Check if condition is triggered."""
        raise NotImplementedError


class HighValueAlert(AlertCondition):
    """Alert when value exceeds threshold."""
    
    def check(self, value: float) -> bool:
        return value > self.threshold


class LowValueAlert(AlertCondition):
    """Alert when value drops below threshold."""
    
    def check(self, value: float) -> bool:
        return value < self.threshold


# ============================================================================
# Predefined Alerts
# ============================================================================

DEFAULT_ALERTS = [
    {
        'name': 'high_latency',
        'metric': 'avg_latency_ms',
        'threshold': 200,
        'severity': 'warning',
        'description': 'Average latency exceeds 200ms'
    },
    {
        'name': 'critical_latency',
        'metric': 'p95_latency_ms',
        'threshold': 500,
        'severity': 'critical',
        'description': 'P95 latency exceeds 500ms'
    },
    {
        'name': 'high_error_rate',
        'metric': 'error_rate',
        'threshold': 0.05,
        'severity': 'critical',
        'description': 'Error rate exceeds 5%'
    },
    {
        'name': 'high_fallback_rate',
        'metric': 'fallback_rate',
        'threshold': 0.95,
        'severity': 'warning',
        'description': 'Fallback rate exceeds 95%'
    },
    {
        'name': 'low_requests',
        'metric': 'requests_per_minute',
        'threshold': 0,
        'severity': 'warning',
        'description': 'No requests in the last minute'
    }
]


# ============================================================================
# Convenience Functions
# ============================================================================

_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def send_alert(subject: str, message: str, severity: str = 'info'):
    """Send alert using default manager."""
    return get_alert_manager().send_alert(subject, message, severity)


def alert_high_latency(latency_ms: float, threshold: float = 200):
    """Send high latency alert."""
    if latency_ms > threshold:
        send_alert(
            subject="High Latency Alert",
            message=f"Average latency is {latency_ms:.1f}ms, threshold is {threshold}ms",
            severity="warning" if latency_ms < threshold * 2 else "critical"
        )


def alert_high_error_rate(error_rate: float, threshold: float = 0.05):
    """Send high error rate alert."""
    if error_rate > threshold:
        send_alert(
            subject="High Error Rate Alert",
            message=f"Error rate is {error_rate:.1%}, threshold is {threshold:.1%}",
            severity="critical"
        )


def alert_data_drift(drift_result: Dict):
    """Send data drift alert."""
    if drift_result.get('drift_detected', False):
        send_alert(
            subject="Data Drift Detected",
            message=f"Rating distribution has shifted (p-value={drift_result.get('p_value', 0):.4f}). "
                    f"Consider retraining the model.",
            severity="warning"
        )


def alert_model_performance(
    current_recall: float,
    baseline_recall: float,
    threshold_drop: float = 0.1
):
    """Send model performance degradation alert."""
    if baseline_recall > 0:
        drop = (baseline_recall - current_recall) / baseline_recall
        if drop > threshold_drop:
            send_alert(
                subject="Model Performance Degradation",
                message=f"Recall@10 dropped from {baseline_recall:.3f} to {current_recall:.3f} "
                        f"({drop:.1%} decrease). Consider retraining.",
                severity="warning"
            )
