"""
app/security/audit_logger.py
─────────────────────────────
Structured, immutable audit logging for all experiment system actions.

Rules:
  - NEVER log PII (national_id, phone, full_name, exact amounts)
  - Always use actor_token (not username or user_id) in log entries
  - Log entries are written to DB and to structured log file
  - Log file uses JSON Lines format for easy parsing
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from app.models.db_models import ExperimentAuditLog

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

_log = structlog.get_logger("audit")


class AuditLogger:
    """
    Write audit events to both:
      1. The database (for board review and compliance queries)
      2. Structured log file (for SIEM integration in production)
    """

    # Actions that must ALWAYS be logged
    CRITICAL_ACTIONS = {
        "EXPERIMENT_APPROVED",
        "EXPERIMENT_SUSPENDED",
        "EXPERIMENT_RESULT_EXPORTED",
        "MEMBER_PII_DECRYPTED",
        "BULK_DATA_EXPORT",
        "RBAC_CONFIG_CHANGED",
        "GUARDRAIL_VIOLATION",
        "ASSIGNMENT_OVERRIDDEN",
    }

    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self._previous_hash: Optional[str] = None

    def _compute_chain_hash(self, entry: dict) -> str:
        """SHA-256 of entry content + previous hash for tamper-evidence."""
        content = json.dumps(entry, sort_keys=True, default=str)
        combined = (content + (self._previous_hash or "genesis")).encode()
        return hashlib.sha256(combined).hexdigest()

    def log(
        self,
        action: str,
        actor_token: str,
        actor_role: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        experiment_id: Optional[int] = None,
        details: Optional[dict] = None,
        ip_hash: Optional[str] = None,
    ) -> None:
        """
        Record an audit event. Call this for every significant action.

        Args:
            action: One of the action constants (e.g. "EXPERIMENT_APPROVED")
            actor_token: Pseudonymous actor identifier (never raw username)
            actor_role: Role of the actor (e.g. "data_steward")
            resource_type: What was acted on (e.g. "experiment", "result")
            resource_id: ID of the affected resource
            experiment_id: FK to experiments table if applicable
            details: Additional context — MUST NOT contain PII
            ip_hash: SHA-256 hash of client IP address
        """
        # Sanitise details — strip any keys that could be PII
        safe_details = self._sanitise_details(details or {})

        entry = {
            "action": action,
            "actor_token": actor_token,
            "actor_role": actor_role,
            "resource_type": resource_type,
            "resource_id": str(resource_id) if resource_id else None,
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": safe_details,
        }

        chain_hash = self._compute_chain_hash(entry)
        self._previous_hash = chain_hash

        # Write to structured log
        _log.info(
            action,
            actor_token=actor_token,
            actor_role=actor_role,
            resource_type=resource_type,
            resource_id=resource_id,
            experiment_id=experiment_id,
            details=safe_details,
            chain_hash=chain_hash,
        )

        # Write to database if session available
        if self._db:
            db_entry = ExperimentAuditLog(
                experiment_id=experiment_id,
                actor_token=actor_token,
                actor_role=actor_role,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                details=json.dumps(safe_details),
                ip_hash=ip_hash,
                previous_log_hash=self._previous_hash,
            )
            self._db.add(db_entry)
            self._db.commit()

        # Alert on critical actions
        if action in self.CRITICAL_ACTIONS:
            self._alert_critical(action, actor_token, actor_role, entry)

    def _sanitise_details(self, details: dict) -> dict:
        """Remove known PII keys from details payload."""
        PII_KEYS = {
            "national_id", "phone", "phone_number", "full_name",
            "name", "email", "id_number", "password", "token_value",
        }
        return {k: v for k, v in details.items()
                if k.lower() not in PII_KEYS}

    def _alert_critical(
        self, action: str, actor_token: str, role: str, entry: dict
    ) -> None:
        """In production: send to SIEM / alert channel. Here: log prominently."""
        _log.warning(
            "CRITICAL_AUDIT_EVENT",
            action=action,
            actor_token=actor_token,
            role=role,
            message="This action requires compliance officer review",
        )
