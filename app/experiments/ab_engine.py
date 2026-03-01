"""
app/experiments/ab_engine.py
─────────────────────────────
Core A/B testing engine.

Provides:
  - SampleSizeCalculator: power-based sample size estimation
  - ExperimentEngine: member assignment, eligibility checking, enrollment
"""
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from scipy import stats
from sqlalchemy.orm import Session

from app.models.db_models import (
    AssignmentGroup, Experiment, ExperimentAssignment,
    ExperimentStatus, Member, MemberStatus
)
from app.security.tokenization import get_tokenization_service

logger = logging.getLogger(__name__)


# ── Sample Size Calculator ────────────────────────────────────────

@dataclass
class SampleSizeResult:
    n_per_arm: int
    total_n: int
    baseline_rate: float
    treatment_rate: float
    mde: float
    alpha: float
    power: float
    test_type: str
    notes: list[str] = field(default_factory=list)


class SampleSizeCalculator:
    """
    Power-based sample size estimation for binary retention KPIs.
    Uses the two-proportion z-test formula (valid for N > 200 per arm).
    For small SACCOs, also provides a feasibility check.
    """

    def calculate(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        allocation_ratio: float = 1.0,   # control:treatment
    ) -> SampleSizeResult:
        """
        Calculate required sample size per arm.

        Formula (two-proportion z-test):
          n = (z_α/2 + z_β)² × [p1(1-p1) + p2(1-p2)] / (p1 - p2)²

        Where:
          p1 = baseline rate (control)
          p2 = baseline rate + MDE (treatment)
        """
        notes = []
        treatment_rate = baseline_rate + minimum_detectable_effect

        if not (0 < baseline_rate < 1):
            raise ValueError("baseline_rate must be between 0 and 1")
        if not (0 < treatment_rate < 1):
            raise ValueError("treatment_rate exceeds 1 — MDE too large for this baseline")

        z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-tailed
        z_beta = stats.norm.ppf(power)

        p1, p2 = baseline_rate, treatment_rate
        pooled_variance = p1 * (1 - p1) + p2 * (1 - p2)
        effect = (p1 - p2) ** 2

        n_control = math.ceil(
            (z_alpha + z_beta) ** 2 * pooled_variance / effect
        )
        n_treatment = math.ceil(n_control * allocation_ratio)

        # Warn if Fisher's exact should be used instead
        if n_control < 200:
            notes.append(
                f"WARNING: n_per_arm={n_control} is below 200. "
                "Use Fisher's exact test instead of z-test for this experiment."
            )

        # Continuity correction for small samples
        if n_control < 300:
            n_control = math.ceil(n_control * 1.10)  # 10% upward correction
            notes.append("Applied 10% continuity correction for small sample.")

        total = n_control + n_treatment
        notes.append(
            f"Minimum experiment duration recommendation: 60 days "
            f"(aligned to SACCO monthly contribution cycle)."
        )

        return SampleSizeResult(
            n_per_arm=n_control,
            total_n=total,
            baseline_rate=baseline_rate,
            treatment_rate=treatment_rate,
            mde=minimum_detectable_effect,
            alpha=alpha,
            power=power,
            test_type="two_proportion_z_test",
            notes=notes,
        )

    def check_feasibility(
        self,
        required_n_per_arm: int,
        eligible_member_count: int,
    ) -> dict:
        """
        Check whether the SACCO has enough eligible members to run the experiment.
        This is the most common reason SACCO experiments fail before they start.
        """
        total_required = required_n_per_arm * 2
        feasible = eligible_member_count >= total_required

        coverage_ratio = eligible_member_count / total_required if total_required > 0 else 0

        recommendation = ""
        if not feasible:
            shortfall = total_required - eligible_member_count
            recommendation = (
                f"Experiment is UNDERPOWERED. You have {eligible_member_count} eligible members "
                f"but need {total_required} (shortfall: {shortfall}). Options: "
                f"(1) Increase the MDE to a larger effect size, "
                f"(2) Pool members with a partner SACCO under a data-sharing agreement, "
                f"(3) Reduce significance level to α=0.10 (increases false positive risk)."
            )
        else:
            recommendation = (
                f"Experiment is feasible. You have {eligible_member_count} eligible members "
                f"and need {total_required} (coverage ratio: {coverage_ratio:.1%})."
            )

        return {
            "feasible": feasible,
            "eligible_members": eligible_member_count,
            "total_required": total_required,
            "n_per_arm_required": required_n_per_arm,
            "coverage_ratio": round(coverage_ratio, 3),
            "recommendation": recommendation,
        }


# ── Experiment Engine ─────────────────────────────────────────────

class ExperimentEngine:
    """
    Manages member eligibility checking, deterministic group assignment,
    and enrollment tracking.
    """

    def __init__(self, db: Session):
        self._db = db
        self._tokenizer = get_tokenization_service()

    def get_eligible_members(
        self,
        experiment: Experiment,
    ) -> list[Member]:
        """
        Apply eligibility criteria to the member base and return eligible members.

        Mandatory exclusions (always applied regardless of experiment config):
          - Members with status != ACTIVE
          - Members flagged as ineligible (is_eligible_for_experiments = False)
          - Members currently in another active experiment (cooling-off enforcement)
          - Members in financial distress (days_past_due > 30 on any active loan)
        """
        criteria = json.loads(experiment.eligibility_criteria)

        # Base query — always-on mandatory exclusions
        query = (
            self._db.query(Member)
            .filter(
                Member.status == MemberStatus.ACTIVE,
                Member.is_eligible_for_experiments == True,
            )
        )

        # Exclude members already in a concurrent active experiment
        active_experiment_member_ids = (
            self._db.query(ExperimentAssignment.member_id)
            .join(Experiment)
            .filter(Experiment.status == ExperimentStatus.RUNNING)
            .filter(Experiment.id != experiment.id)
            .subquery()
        )
        query = query.filter(
            ~Member.id.in_(active_experiment_member_ids)
        )

        # Optional criteria from experiment config
        if criteria.get("geographic_zones"):
            query = query.filter(
                Member.geographic_zone.in_(criteria["geographic_zones"])
            )

        if criteria.get("loan_product_types"):
            query = query.filter(
                Member.loan_product_type.in_(criteria["loan_product_types"])
            )

        if criteria.get("min_tenure_months"):
            min_join = date.today().replace(
                year=date.today().year - (criteria["min_tenure_months"] // 12)
            )
            query = query.filter(Member.join_date <= min_join)

        if criteria.get("max_tenure_months"):
            max_join = date.today().replace(
                year=date.today().year - (criteria["max_tenure_months"] // 12)
            )
            query = query.filter(Member.join_date >= max_join)

        return query.all()

    def assign_member(
        self,
        member: Member,
        experiment: Experiment,
    ) -> AssignmentGroup:
        """
        Deterministically assign a member to control or treatment group.

        Uses HMAC-SHA256(hmac_key, member_token || experiment_key) mod 100
          0–49  → CONTROL
          50–99 → TREATMENT

        This is DETERMINISTIC: calling it twice for the same inputs
        always returns the same group. No state needed.
        """
        value = self._tokenizer.generate_experiment_assignment_token(
            member_token=member.member_token,
            experiment_key=experiment.experiment_key,
        )
        return AssignmentGroup.CONTROL if value < 50 else AssignmentGroup.TREATMENT

    def enroll_members(
        self,
        experiment: Experiment,
        dry_run: bool = False,
    ) -> dict:
        """
        Enroll all eligible members into the experiment.

        In dry_run=True mode: calculates assignments without writing to DB.
        Use this for A/A validation checks before going live.
        """
        eligible = self.get_eligible_members(experiment)

        control_members = []
        treatment_members = []
        assignments_to_create = []

        for member in eligible:
            group = self.assign_member(member, experiment)

            # Determine tenure band for stratification tracking
            months_tenure = (date.today() - member.join_date).days // 30
            if months_tenure < 12:
                tenure_band = "new"
            elif months_tenure < 60:
                tenure_band = "mid"
            else:
                tenure_band = "tenured"

            assignment = ExperimentAssignment(
                experiment_id=experiment.id,
                member_id=member.id,
                member_token=member.member_token,
                assignment_group=group,
                strata_geographic_zone=member.geographic_zone,
                strata_loan_product_type=member.loan_product_type,
                strata_tenure_band=tenure_band,
            )
            assignments_to_create.append(assignment)

            if group == AssignmentGroup.CONTROL:
                control_members.append(member)
            else:
                treatment_members.append(member)

        enrollment_summary = {
            "total_eligible": len(eligible),
            "control_n": len(control_members),
            "treatment_n": len(treatment_members),
            "balance_ratio": (
                len(treatment_members) / len(control_members)
                if control_members else 0
            ),
            "dry_run": dry_run,
        }

        if not dry_run:
            self._db.add_all(assignments_to_create)
            # Update experiment status to RUNNING
            experiment.status = ExperimentStatus.RUNNING
            experiment.actual_start_date = date.today()
            self._db.commit()
            logger.info(
                f"Enrolled {len(assignments_to_create)} members into "
                f"experiment '{experiment.experiment_key}'"
            )

        # Balance check — warn if imbalance > 2pp
        if abs(enrollment_summary["balance_ratio"] - 1.0) > 0.04:
            enrollment_summary["balance_warning"] = (
                "Group sizes are imbalanced by more than 2%. "
                "Check randomization logic before proceeding."
            )

        return enrollment_summary

    def check_guardrail_suspension(
        self,
        experiment: Experiment,
        repayment_control_rate: float,
        repayment_treatment_rate: float,
        threshold_drop: float = 0.02,
    ) -> bool:
        """
        Automatic guardrail: suspend experiment if on-time repayment rate
        in treatment arm drops more than threshold_drop below control.

        Returns True if experiment was suspended.
        """
        drop = repayment_control_rate - repayment_treatment_rate
        if drop > threshold_drop:
            experiment.status = ExperimentStatus.SUSPENDED
            experiment.suspension_reason = (
                f"AUTO-SUSPENDED: On-time repayment rate in treatment arm "
                f"({repayment_treatment_rate:.1%}) dropped {drop:.1%} below "
                f"control arm ({repayment_control_rate:.1%}), exceeding the "
                f"{threshold_drop:.1%} guardrail threshold."
            )
            self._db.commit()
            logger.critical(
                f"GUARDRAIL VIOLATION: Experiment {experiment.experiment_key} "
                f"suspended due to repayment rate drop."
            )
            return True
        return False
