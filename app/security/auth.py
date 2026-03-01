"""
app/security/auth.py
──────────────────────
Role-Based Access Control (RBAC) for the prototype.

Roles and permissions matrix:
  experiment_designer  → create/edit experiments; cannot view results
  data_analyst         → view aggregated KPI results for assigned experiments
  data_steward         → approve experiments; view monitoring dashboard; suspend
  compliance_officer   → read-only access to audit logs and configurations
  credit_risk_officer  → review and approve loan-related experiment designs
  admin                → all permissions (local dev only; disabled in production)
"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Optional, List

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config.settings import get_settings

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Role(str, Enum):
    EXPERIMENT_DESIGNER = "experiment_designer"
    DATA_ANALYST = "data_analyst"
    DATA_STEWARD = "data_steward"
    COMPLIANCE_OFFICER = "compliance_officer"
    CREDIT_RISK_OFFICER = "credit_risk_officer"
    ADMIN = "admin"


# Permissions granted to each role
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.EXPERIMENT_DESIGNER: {
        "experiments:create",
        "experiments:read_own",
        "experiments:update_draft",
    },
    Role.DATA_ANALYST: {
        "experiments:read_config",
        "results:read_aggregated",
        "kpis:compute",
    },
    Role.DATA_STEWARD: {
        "experiments:read_all",
        "experiments:approve",
        "experiments:suspend",
        "monitoring:read",
        "results:read_aggregated",
        "audit:read_summary",
    },
    Role.COMPLIANCE_OFFICER: {
        "experiments:read_all",
        "audit:read_full",
        "results:read_aggregated",
        "members:read_eligibility_stats",
    },
    Role.CREDIT_RISK_OFFICER: {
        "experiments:read_all",
        "experiments:credit_review",
        "results:read_aggregated",
    },
    Role.ADMIN: {
        "experiments:create", "experiments:read_all", "experiments:update_draft",
        "experiments:approve", "experiments:suspend",
        "results:read_aggregated", "audit:read_full",
        "monitoring:read", "kpis:compute",
        "members:read_eligibility_stats",
        "admin:manage_users",
    },
}


class TokenData(BaseModel):
    sub: str           # user identifier (not PII — internal user ID)
    role: Role
    exp: datetime
    actor_token: str   # HMAC token of user identity for audit logs


class UserContext(BaseModel):
    user_id: str
    role: Role
    actor_token: str
    permissions: set[str]

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def require_permission(self, permission: str) -> None:
        if not self.has_permission(permission):
            raise PermissionError(
                f"Role '{self.role}' does not have permission '{permission}'"
            )


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(
    user_id: str,
    role: Role,
    actor_token: str,
) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_expire_minutes
    )
    payload = {
        "sub": user_id,
        "role": role.value,
        "actor_token": actor_token,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key,
                      algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[TokenData]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return TokenData(
            sub=payload["sub"],
            role=Role(payload["role"]),
            exp=payload["exp"],
            actor_token=payload["actor_token"],
        )
    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        return None


def get_user_context(token: str) -> Optional[UserContext]:
    token_data = decode_token(token)
    if not token_data:
        return None
    permissions = ROLE_PERMISSIONS.get(token_data.role, set())
    return UserContext(
        user_id=token_data.sub,
        role=token_data.role,
        actor_token=token_data.actor_token,
        permissions=permissions,
    )


def require_permission(permission: str):
    """Decorator for FastAPI route functions requiring specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user: UserContext = kwargs.get("current_user")
            if not user:
                raise PermissionError("No authenticated user in context")
            user.require_permission(permission)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
