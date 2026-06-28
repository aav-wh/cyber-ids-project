# ids.detection — alert generation, rule engine, threshold management, simulator
from ids.detection.alerts import Alert, AlertSeverity, generate_alert
from ids.detection.rules import RuleEngine

__all__ = ["Alert", "AlertSeverity", "generate_alert", "RuleEngine"]
