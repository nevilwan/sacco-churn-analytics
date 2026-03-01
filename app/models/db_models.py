"""
app/models/db_models.py
────────────────────────
SQLAlchemy ORM models for all SACCO entities.

Design principles:
  - PII fields (national_id, phone) are NEVER stored in plaintext
  - All sensitive fields store only the encrypted/tokenized form
  - member_token is the pseudonymous ID used throughout analytics
  - Raw member identity lives only in the `members` table and is
    never joined into experiment or analytics tables
"""
from datetime import datetime, date
from decimal import Decimal
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Numeric, Date, DateTime, Boolean,
    ForeignKey, Text, Index, CheckConstraint, UniqueConstraint,
    Enum as SAEnum
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────

class MemberStatus(str, PyEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    SUSPENDED = "suspended"
    EXITED = "exited"

class LoanStatus(str, PyEnum):
    PENDING = "pending"
    DISBURSED = "disbursed"
    REPAYING = "repaying"
    COMPLETED = "completed"
    DEFAULTED = "defaulted"
    RESTRUCTURED = "restructured"

class TransactionType(str, PyEnum):
    SAVINGS_DEPOSIT = "savings_deposit"
    SAVINGS_WITHDRAWAL = "savings_withdrawal"
    LOAN_REPAYMENT = "loan_repayment"
    LOAN_DISBURSEMENT = "loan_disbursement"
    SHARE_CAPITAL = "share_capital"
    DIVIDEND = "dividend"
    MPESA_DEPOSIT = "mpesa_deposit"
    MPESA_WITHDRAWAL = "mpesa_withdrawal"

class ExperimentStatus(str, PyEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    SUSPENDED = "suspended"

class AssignmentGroup(str, PyEnum):
    CONTROL = "control"
    TREATMENT = "treatment"

class GeographicZone(str, PyEnum):
    URBAN_NAIROBI = "urban_nairobi"
    URBAN_OTHER = "urban_other"
    PERI_URBAN = "peri_urban"
    RURAL = "rural"


# ── Core SACCO Tables ────────────────────────────────────────────

class Member(Base):
    """
    Core member record.
    
    SECURITY NOTE:
      - national_id_encrypted: Fernet-encrypted National ID (reversible, for KYC)
      - phone_encrypted: Fernet-encrypted phone number (reversible, for comms)
      - member_token: HMAC-SHA256(secret, national_id + phone + account_number)
        Used as the join key in ALL experiment and analytics tables.
        Analysts NEVER see national_id or phone — they use member_token only.
    """
    __tablename__ = "members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_token = Column(String(64), unique=True, nullable=False, index=True)
    member_number = Column(String(20), unique=True, nullable=False)  # e.g. SCC-0001234

    # PII — stored encrypted only
    national_id_encrypted = Column(Text, nullable=False)
    phone_encrypted = Column(Text, nullable=False)
    full_name_encrypted = Column(Text, nullable=False)

    # Non-sensitive demographic fields for stratification
    gender = Column(String(10), nullable=True)       # M / F / Other — used for fairness checks
    geographic_zone = Column(SAEnum(GeographicZone), nullable=False)
    join_date = Column(Date, nullable=False)
    employment_sector = Column(String(50), nullable=True)  # e.g. "public_sector", "agriculture"
    loan_product_type = Column(String(50), nullable=True)  # "salary_backed", "business", "emergency"

    # Status
    status = Column(SAEnum(MemberStatus), default=MemberStatus.ACTIVE, nullable=False)
    is_eligible_for_experiments = Column(Boolean, default=True, nullable=False)
    last_activity_date = Column(Date, nullable=True)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    savings_accounts = relationship("SavingsAccount", back_populates="member")
    loans = relationship("Loan", back_populates="member")
    transactions = relationship("Transaction", back_populates="member")
    experiment_assignments = relationship("ExperimentAssignment", back_populates="member")

    __table_args__ = (
        Index("ix_members_status_zone", "status", "geographic_zone"),
        Index("ix_members_join_date", "join_date"),
        Index("ix_members_loan_product", "loan_product_type"),
    )


class SavingsAccount(Base):
    __tablename__ = "savings_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    account_number = Column(String(20), unique=True, nullable=False)
    account_type = Column(String(30), nullable=False)  # "ordinary", "fixed_deposit", "holiday"
    balance = Column(Numeric(15, 2), default=0.00, nullable=False)
    share_capital_units = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    opened_date = Column(Date, nullable=False)
    last_transaction_date = Column(Date, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    member = relationship("Member", back_populates="savings_accounts")
    transactions = relationship("Transaction", back_populates="savings_account")

    __table_args__ = (
        Index("ix_savings_member_id", "member_id"),
        CheckConstraint("balance >= 0", name="ck_savings_balance_non_negative"),
    )


class Loan(Base):
    __tablename__ = "loans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    loan_number = Column(String(20), unique=True, nullable=False)
    product_type = Column(String(50), nullable=False)
    principal_amount = Column(Numeric(15, 2), nullable=False)
    interest_rate = Column(Numeric(5, 4), nullable=False)  # e.g. 0.1200 = 12%
    term_months = Column(Integer, nullable=False)
    monthly_installment = Column(Numeric(15, 2), nullable=False)
    outstanding_balance = Column(Numeric(15, 2), nullable=False)
    total_repaid = Column(Numeric(15, 2), default=0.00, nullable=False)
    status = Column(SAEnum(LoanStatus), default=LoanStatus.PENDING, nullable=False)
    disbursement_date = Column(Date, nullable=True)
    expected_completion_date = Column(Date, nullable=True)
    actual_completion_date = Column(Date, nullable=True)
    days_past_due = Column(Integer, default=0, nullable=False)
    loan_cycle = Column(Integer, default=1, nullable=False)  # which loan this is for member

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    member = relationship("Member", back_populates="loans")
    transactions = relationship("Transaction", back_populates="loan")

    __table_args__ = (
        Index("ix_loans_member_id", "member_id"),
        Index("ix_loans_status", "status"),
        Index("ix_loans_disbursement_date", "disbursement_date"),
        CheckConstraint("principal_amount > 0", name="ck_loan_principal_positive"),
        CheckConstraint("days_past_due >= 0", name="ck_loan_dpd_non_negative"),
    )


class Transaction(Base):
    """
    Immutable transaction log. Records are never updated or deleted.
    Amount is stored in buckets (ranges) for analytics to prevent
    re-identification in small cohorts.
    """
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    savings_account_id = Column(Integer, ForeignKey("savings_accounts.id"), nullable=True)
    loan_id = Column(Integer, ForeignKey("loans.id"), nullable=True)

    transaction_type = Column(SAEnum(TransactionType), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    # Amount range bucket for privacy-preserving analytics
    amount_bucket = Column(String(20), nullable=False)   # e.g. "0-2000", "2001-10000", "10000+"
    channel = Column(String(20), nullable=False)  # "branch", "ussd", "mpesa", "mobile_app"
    reference_number = Column(String(50), unique=True, nullable=False)

    # M-Pesa specific (when channel = mpesa)
    mpesa_transaction_id_hash = Column(String(64), nullable=True)  # SHA-256 hash only

    transaction_date = Column(DateTime(timezone=True), nullable=False)
    value_date = Column(Date, nullable=False)
    is_on_time = Column(Boolean, nullable=True)  # For repayments only
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    member = relationship("Member", back_populates="transactions")
    savings_account = relationship("SavingsAccount", back_populates="transactions")
    loan = relationship("Loan", back_populates="transactions")

    __table_args__ = (
        Index("ix_txn_member_date", "member_id", "transaction_date"),
        Index("ix_txn_type_date", "transaction_type", "value_date"),
        Index("ix_txn_loan_id", "loan_id"),
    )


# ── Experiment Tables ─────────────────────────────────────────────

class Experiment(Base):
    """
    Experiment configuration — tamper-evident via created_at immutability.
    After status moves to RUNNING, hypothesis and config fields must not change.
    """
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_key = Column(String(50), unique=True, nullable=False)  # e.g. "savings_reminder_v1"
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    hypothesis_null = Column(Text, nullable=False)
    hypothesis_alternative = Column(Text, nullable=False)

    # Statistical design parameters (locked at APPROVED status)
    primary_kpi = Column(String(100), nullable=False)
    secondary_kpis = Column(Text, nullable=True)   # JSON list
    guardrail_kpis = Column(Text, nullable=False)  # JSON list — must always be set
    significance_level = Column(Numeric(4, 3), nullable=False)  # e.g. 0.050
    target_power = Column(Numeric(4, 3), nullable=False)        # e.g. 0.800
    minimum_detectable_effect = Column(Numeric(6, 4), nullable=False)
    target_sample_size_per_arm = Column(Integer, nullable=False)
    baseline_rate = Column(Numeric(6, 4), nullable=False)

    # Eligibility filter (JSON)
    eligibility_criteria = Column(Text, nullable=False)

    # Timeline
    status = Column(SAEnum(ExperimentStatus), default=ExperimentStatus.DRAFT, nullable=False)
    planned_start_date = Column(Date, nullable=False)
    planned_end_date = Column(Date, nullable=False)
    actual_start_date = Column(Date, nullable=True)
    actual_end_date = Column(Date, nullable=True)

    # Governance
    designed_by = Column(String(100), nullable=False)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    credit_risk_reviewed_by = Column(String(100), nullable=True)  # required for loan experiments
    suspension_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    assignments = relationship("ExperimentAssignment", back_populates="experiment")
    results = relationship("ExperimentResult", back_populates="experiment")
    audit_logs = relationship("ExperimentAuditLog", back_populates="experiment")

    __table_args__ = (
        Index("ix_experiment_status", "status"),
        CheckConstraint("planned_end_date > planned_start_date",
                        name="ck_experiment_dates_valid"),
        CheckConstraint("significance_level > 0 AND significance_level < 1",
                        name="ck_experiment_alpha_valid"),
    )


class ExperimentAssignment(Base):
    """
    Member-to-group assignment. Written once at enrollment, never modified.
    Uses member_token (not member_id) to maintain analytics layer separation.
    """
    __tablename__ = "experiment_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    member_token = Column(String(64), nullable=False)
    assignment_group = Column(SAEnum(AssignmentGroup), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())

    # Stratification values captured at enrollment time (for balance checks)
    strata_geographic_zone = Column(String(50), nullable=True)
    strata_loan_product_type = Column(String(50), nullable=True)
    strata_tenure_band = Column(String(20), nullable=True)   # "new", "mid", "tenured"

    experiment = relationship("Experiment", back_populates="assignments")
    member = relationship("Member", back_populates="experiment_assignments")

    __table_args__ = (
        UniqueConstraint("experiment_id", "member_id",
                         name="uq_assignment_experiment_member"),
        Index("ix_assignment_exp_group", "experiment_id", "assignment_group"),
        Index("ix_assignment_token", "member_token"),
    )


class ExperimentResult(Base):
    """
    Computed KPI results per experiment arm.
    Stores aggregated metrics only — no individual member data.
    Written by the analysis module after experiment completes.
    """
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    analysis_window_days = Column(Integer, nullable=False)  # 30, 60, or 90

    # Sample counts
    control_n = Column(Integer, nullable=False)
    treatment_n = Column(Integer, nullable=False)

    # Primary KPI result
    control_rate = Column(Numeric(8, 6), nullable=False)
    treatment_rate = Column(Numeric(8, 6), nullable=False)
    absolute_effect = Column(Numeric(8, 6), nullable=False)
    relative_effect = Column(Numeric(8, 6), nullable=False)
    p_value = Column(Numeric(10, 8), nullable=False)
    ci_lower = Column(Numeric(8, 6), nullable=False)
    ci_upper = Column(Numeric(8, 6), nullable=False)
    statistic = Column(Numeric(10, 6), nullable=False)
    test_used = Column(String(50), nullable=False)
    null_rejected = Column(Boolean, nullable=False)

    # Guardrail results (JSON)
    guardrail_results = Column(Text, nullable=True)
    guardrail_passed = Column(Boolean, nullable=False, default=True)

    # Interpretation
    conclusion = Column(Text, nullable=True)
    analyst_token = Column(String(64), nullable=False)

    experiment = relationship("Experiment", back_populates="results")

    __table_args__ = (
        Index("ix_results_experiment_id", "experiment_id"),
    )


class ExperimentAuditLog(Base):
    """
    Immutable audit log. No updates, no deletes — ever.
    Chain integrity: each record hashes the previous record's content.
    """
    __tablename__ = "experiment_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    actor_token = Column(String(64), nullable=False)      # Pseudonymous actor identity
    actor_role = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)                  # JSON payload (no PII)
    ip_hash = Column(String(64), nullable=True)            # SHA-256 of IP address
    previous_log_hash = Column(String(64), nullable=True)  # Chain integrity
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    experiment = relationship("Experiment", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_actor", "actor_token"),
        Index("ix_audit_created_at", "created_at"),
        Index("ix_audit_action", "action"),
    )
