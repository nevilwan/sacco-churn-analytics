"""
app/security/tokenization.py
─────────────────────────────
Handles all PII encryption and pseudonymous token generation.

Two mechanisms:
  1. Fernet symmetric encryption  — reversible, for fields we need to decrypt
     (national_id, phone, full_name) — used only within the identity service
  2. HMAC-SHA256 tokenization     — one-way pseudonymous token
     (member_token) — used everywhere in analytics layer

NEVER import this module in dashboard or analytics code.
Analytics code only ever receives member_token, never decrypts PII.
"""
import hashlib
import hmac
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from config.settings import get_settings

logger = logging.getLogger(__name__)


class TokenizationService:
    """
    Centralised PII tokenization and encryption.
    Instantiate once via get_tokenization_service().
    """

    def __init__(self, tokenization_secret_key: str, hmac_secret_key: str):
        self._fernet = Fernet(tokenization_secret_key.encode()
                              if isinstance(tokenization_secret_key, str)
                              else tokenization_secret_key)
        self._hmac_key = (hmac_secret_key.encode()
                          if isinstance(hmac_secret_key, str)
                          else hmac_secret_key)

    # ── Fernet encryption (reversible) ───────────────────────────

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string. Returns base64-encoded ciphertext."""
        if not plaintext:
            raise ValueError("Cannot encrypt empty string")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a Fernet-encrypted string.
        ONLY call this from identity service endpoints with appropriate RBAC.
        Log every decryption call.
        """
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as e:
            logger.error("Decryption failed — possible key mismatch or tampered data")
            raise ValueError("Decryption failed") from e

    # ── HMAC tokenization (one-way) ──────────────────────────────

    def generate_member_token(
        self,
        national_id: str,
        phone_number: str,
        account_number: str,
    ) -> str:
        """
        Generate a deterministic pseudonymous member token.

        token = HMAC-SHA256(hmac_key, national_id || phone_number || account_number)

        This token:
          - Is stable: same inputs always produce the same token
          - Is one-way: cannot be reversed without the hmac_key
          - Is unique per member: different national IDs produce different tokens
          - Is safe for analytics: analysts can use this as a join key
        """
        composite = f"{national_id.strip()}|{phone_number.strip()}|{account_number.strip()}"
        token = hmac.new(
            self._hmac_key,
            composite.encode(),
            hashlib.sha256,
        ).hexdigest()
        return token

    def generate_experiment_assignment_token(
        self,
        member_token: str,
        experiment_key: str,
    ) -> int:
        """
        Deterministic assignment value (0–99) for a member in a specific experiment.

        value = HMAC-SHA256(hmac_key, member_token || experiment_key) mod 100

        Values 0–49  → Control
        Values 50–99 → Treatment

        This guarantees:
          - Same member always gets same group for same experiment
          - Assignments across experiments are statistically independent
          - No one can predict assignment without the hmac_key
        """
        composite = f"{member_token}|{experiment_key}"
        raw = hmac.new(
            self._hmac_key,
            composite.encode(),
            hashlib.sha256,
        ).digest()
        # Use first 4 bytes as integer for mod operation
        value = int.from_bytes(raw[:4], byteorder="big") % 100
        return value

    def hash_ip(self, ip_address: str) -> str:
        """One-way hash of IP address for audit logs."""
        return hashlib.sha256(
            (ip_address + "ip_salt_v1").encode()
        ).hexdigest()

    def hash_mpesa_transaction_id(self, mpesa_id: str) -> str:
        """One-way hash of M-Pesa transaction ID for dedup without storing PII."""
        return hashlib.sha256(mpesa_id.encode()).hexdigest()

    def bucket_amount(self, amount: float) -> str:
        """
        Convert precise amount to privacy-preserving range bucket.
        Prevents re-identification in small cohorts where exact amounts
        may uniquely identify a member.
        """
        if amount < 0:
            return "negative"
        elif amount == 0:
            return "0"
        elif amount <= 1000:
            return "1-1000"
        elif amount <= 5000:
            return "1001-5000"
        elif amount <= 10000:
            return "5001-10000"
        elif amount <= 50000:
            return "10001-50000"
        elif amount <= 100000:
            return "50001-100000"
        else:
            return "100000+"


_tokenization_service: Optional[TokenizationService] = None


def get_tokenization_service() -> TokenizationService:
    """Singleton accessor — call this instead of instantiating directly."""
    global _tokenization_service
    if _tokenization_service is None:
        settings = get_settings()
        _tokenization_service = TokenizationService(
            tokenization_secret_key=settings.tokenization_secret_key,
            hmac_secret_key=settings.hmac_secret_key,
        )
    return _tokenization_service
