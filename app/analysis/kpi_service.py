"""
app/analysis/kpi_service.py
────────────────────────────
Computes retention and engagement KPIs from transaction data
for experiment cohorts.

IMPORTANT: This module NEVER touches PII fields.
It operates entirely on member_token and behavioural event data.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.db_models import (
    AssignmentGroup, Experiment, ExperimentAssignment,
    Transaction, TransactionType
)

logger = logging.getLogger(__name__)


class KPIService:

    def __init__(self, db: Session, min_cohort_size: int = 30):
        self._db = db
        self._min_cohort_size = min_cohort_size

    def compute_retention_rate(
        self,
        experiment_id: int,
        window_days: int = 90,
        group: Optional[AssignmentGroup] = None,
    ) -> dict:
        """
        30/60/90-day Active Member Rate.

        Definition: % of enrolled members who performed ≥1 qualifying transaction
        (savings_deposit, loan_repayment, mpesa_deposit) within the window.
        Window starts 7 days after enrollment to exclude novelty effect.
        """
        assignments = self._get_assignments(experiment_id, group)
        if not assignments:
            return self._empty_kpi("retention_rate", window_days)

        results = {}
        for assignment in assignments:
            # Count qualifying transactions in the window
            start_date = (assignment.enrolled_at.date() +
                          timedelta(days=7))   # skip novelty period
            end_date = start_date + timedelta(days=window_days)

            txn_count = (
                self._db.query(Transaction)
                .filter(
                    Transaction.member_id == assignment.member_id,
                    Transaction.transaction_type.in_([
                        TransactionType.SAVINGS_DEPOSIT,
                        TransactionType.LOAN_REPAYMENT,
                        TransactionType.MPESA_DEPOSIT,
                    ]),
                    Transaction.value_date >= start_date,
                    Transaction.value_date <= end_date,
                )
                .count()
            )
            results[assignment.member_token] = {
                "group": assignment.assignment_group,
                "retained": int(txn_count >= 1),
            }

        df = pd.DataFrame(results).T
        if len(df) < self._min_cohort_size:
            return {"error": f"Cohort too small (< {self._min_cohort_size}) to display"}

        summary = df.groupby("group")["retained"].agg(["sum", "count"])
        return {
            "kpi": f"retention_{window_days}d",
            "window_days": window_days,
            "groups": {
                str(g): {
                    "n": int(row["count"]),
                    "successes": int(row["sum"]),
                    "rate": round(row["sum"] / row["count"], 4),
                }
                for g, row in summary.iterrows()
            }
        }

    def compute_on_time_repayment_rate(
        self,
        experiment_id: int,
        group: Optional[AssignmentGroup] = None,
    ) -> dict:
        """
        On-time repayment rate — mandatory guardrail KPI.
        Must be checked every 7 days during a running experiment.
        """
        assignments = self._get_assignments(experiment_id, group)
        if not assignments:
            return self._empty_kpi("on_time_repayment", 0)

        results = {}
        for assignment in assignments:
            repayments = (
                self._db.query(Transaction)
                .filter(
                    Transaction.member_id == assignment.member_id,
                    Transaction.transaction_type == TransactionType.LOAN_REPAYMENT,
                    Transaction.is_on_time.isnot(None),
                )
                .all()
            )
            if repayments:
                on_time = sum(1 for r in repayments if r.is_on_time)
                results[assignment.member_token] = {
                    "group": assignment.assignment_group,
                    "on_time": on_time,
                    "total": len(repayments),
                    "rate": on_time / len(repayments),
                }

        if len(results) < self._min_cohort_size:
            return {"error": f"Cohort too small (< {self._min_cohort_size}) to display"}

        df = pd.DataFrame(results).T
        summary = df.groupby("group").apply(
            lambda g: pd.Series({
                "n_members": len(g),
                "total_repayments": int(g["total"].sum()),
                "on_time_repayments": int(g["on_time"].sum()),
                "rate": g["on_time"].sum() / g["total"].sum() if g["total"].sum() > 0 else 0,
            })
        )

        return {
            "kpi": "on_time_repayment_rate",
            "groups": {
                str(g): {
                    "n_members": int(row["n_members"]),
                    "total_repayments": int(row["total_repayments"]),
                    "on_time_repayments": int(row["on_time_repayments"]),
                    "rate": round(row["rate"], 4),
                }
                for g, row in summary.iterrows()
            }
        }

    def compute_savings_consistency_rate(
        self,
        experiment_id: int,
        observation_months: int = 4,
        min_active_months: int = 3,
    ) -> dict:
        """
        Savings Consistency Rate.
        % of members making a savings deposit in ≥ min_active_months of
        the past observation_months.
        """
        assignments = self._get_assignments(experiment_id)
        results = {}

        for assignment in assignments:
            start = assignment.enrolled_at.date()
            deposits_by_month = {}

            for m in range(observation_months):
                month_start = start + timedelta(days=m * 30)
                month_end = month_start + timedelta(days=30)
                count = (
                    self._db.query(Transaction)
                    .filter(
                        Transaction.member_id == assignment.member_id,
                        Transaction.transaction_type == TransactionType.SAVINGS_DEPOSIT,
                        Transaction.value_date >= month_start,
                        Transaction.value_date < month_end,
                    )
                    .count()
                )
                deposits_by_month[m] = int(count > 0)

            active_months = sum(deposits_by_month.values())
            results[assignment.member_token] = {
                "group": assignment.assignment_group,
                "consistent": int(active_months >= min_active_months),
                "active_months": active_months,
            }

        if len(results) < self._min_cohort_size:
            return {"error": f"Cohort too small to display"}

        df = pd.DataFrame(results).T
        summary = df.groupby("group")["consistent"].agg(["sum", "count"])

        return {
            "kpi": "savings_consistency_rate",
            "observation_months": observation_months,
            "min_active_months_required": min_active_months,
            "groups": {
                str(g): {
                    "n": int(row["count"]),
                    "consistent_count": int(row["sum"]),
                    "rate": round(row["sum"] / row["count"], 4),
                }
                for g, row in summary.iterrows()
            }
        }

    def compute_loan_repeat_rate(
        self,
        experiment_id: int,
        lookback_days: int = 90,
    ) -> dict:
        """
        Loan Repeat Rate: % of members who completed a loan and applied for
        a new one within lookback_days.
        """
        from app.models.db_models import Loan, LoanStatus

        assignments = self._get_assignments(experiment_id)
        results = {}

        for assignment in assignments:
            # Find loans completed during or before the experiment window
            completed_loans = (
                self._db.query(Loan)
                .filter(
                    Loan.member_id == assignment.member_id,
                    Loan.status == LoanStatus.COMPLETED,
                )
                .all()
            )

            repeated = 0
            if completed_loans:
                # Check if a new loan was initiated within lookback_days
                latest_completion = max(
                    l.actual_completion_date for l in completed_loans
                    if l.actual_completion_date
                )
                if latest_completion:
                    new_loan = (
                        self._db.query(Loan)
                        .filter(
                            Loan.member_id == assignment.member_id,
                            Loan.disbursement_date > latest_completion,
                            Loan.disbursement_date <=
                                latest_completion + timedelta(days=lookback_days),
                        )
                        .first()
                    )
                    repeated = 1 if new_loan else 0

                results[assignment.member_token] = {
                    "group": assignment.assignment_group,
                    "repeated": repeated,
                    "has_completed_loan": 1,
                }

        if len(results) < self._min_cohort_size:
            return {"error": f"Cohort too small to display"}

        df = pd.DataFrame(results).T
        summary = df.groupby("group")["repeated"].agg(["sum", "count"])

        return {
            "kpi": "loan_repeat_rate",
            "lookback_days": lookback_days,
            "groups": {
                str(g): {
                    "n": int(row["count"]),
                    "repeated_count": int(row["sum"]),
                    "rate": round(row["sum"] / row["count"], 4),
                }
                for g, row in summary.iterrows()
            }
        }

    # ── Private helpers ───────────────────────────────────────────

    def _get_assignments(
        self,
        experiment_id: int,
        group: Optional[AssignmentGroup] = None,
    ) -> list:
        query = (
            self._db.query(ExperimentAssignment)
            .filter(ExperimentAssignment.experiment_id == experiment_id)
        )
        if group:
            query = query.filter(
                ExperimentAssignment.assignment_group == group
            )
        return query.all()

    def _empty_kpi(self, name: str, window: int) -> dict:
        return {"kpi": name, "window_days": window, "groups": {}, "error": "No data"}
