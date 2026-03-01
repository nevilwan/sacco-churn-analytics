"""
app/api/main.py
────────────────
FastAPI application entry point.
Defines routes for experiment management and KPI querying.
"""
import json
import logging
from datetime import date
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.analysis.kpi_service import KPIService
from app.analysis.statistical_tests import (
    simulate_experiment_data,
    test_binary_kpi,
)
from app.core.database import get_db, init_db
from app.experiments.ab_engine import ExperimentEngine, SampleSizeCalculator
from app.models.db_models import (
    AssignmentGroup, Experiment, ExperimentStatus, Member, MemberStatus
)
from app.security.audit_logger import AuditLogger
from app.security.auth import (
    Role, UserContext, create_access_token, get_user_context, verify_password
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="SACCO Retention Analysis System",
    description="Experimental A/B testing framework for Kenyan SACCOs and MFIs",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url=None,
)

# CORS — localhost only in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit only
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ── Startup ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("SACCO Retention Analysis API started")


# ── Auth ──────────────────────────────────────────────────────────

# In production: replace with DB-backed user store
DEMO_USERS = {
    "analyst": {"hashed_password": "$2b$12$dummy_hash", "role": Role.DATA_ANALYST,
                "actor_token": "analyst_token_001"},
    "steward": {"hashed_password": "$2b$12$dummy_hash", "role": Role.DATA_STEWARD,
                "actor_token": "steward_token_001"},
    "designer": {"hashed_password": "$2b$12$dummy_hash", "role": Role.EXPERIMENT_DESIGNER,
                 "actor_token": "designer_token_001"},
    "compliance": {"hashed_password": "$2b$12$dummy_hash", "role": Role.COMPLIANCE_OFFICER,
                   "actor_token": "compliance_token_001"},
}


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserContext:
    user = get_user_context(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = DEMO_USERS.get(form.username)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    # In production: verify_password(form.password, user["hashed_password"])
    token = create_access_token(
        user_id=form.username,
        role=user["role"],
        actor_token=user["actor_token"],
    )
    return {"access_token": token, "token_type": "bearer"}


# ── Request/Response models ───────────────────────────────────────

class ExperimentCreate(BaseModel):
    experiment_key: str = Field(..., pattern=r"^[a-z0-9_]+$", max_length=50)
    name: str = Field(..., max_length=200)
    description: str
    hypothesis_null: str
    hypothesis_alternative: str
    primary_kpi: str
    guardrail_kpis: list[str]
    significance_level: float = Field(0.05, ge=0.001, le=0.10)
    target_power: float = Field(0.80, ge=0.70, le=0.95)
    minimum_detectable_effect: float = Field(..., gt=0, lt=1)
    baseline_rate: float = Field(..., gt=0, lt=1)
    planned_start_date: date
    planned_end_date: date
    eligibility_criteria: dict = {}


class SampleSizeRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0, lt=1)
    mde: float = Field(..., gt=0, lt=1)
    alpha: float = Field(0.05, gt=0, lt=1)
    power: float = Field(0.80, gt=0.5, lt=1)
    eligible_member_count: Optional[int] = None


# ── Experiment routes ─────────────────────────────────────────────

@app.post("/experiments/", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("experiments:create")

    # Check max concurrent experiments (hard limit = 3)
    running_count = db.query(Experiment).filter(
        Experiment.status == ExperimentStatus.RUNNING
    ).count()
    if running_count >= 3:
        raise HTTPException(
            status_code=400,
            detail="Maximum concurrent experiments (3) reached. "
                   "Complete or suspend an existing experiment before creating a new one."
        )

    # Check duplicate key
    existing = db.query(Experiment).filter(
        Experiment.experiment_key == payload.experiment_key
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Experiment key already exists")

    # Validate dates
    if payload.planned_end_date <= payload.planned_start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    min_duration = (payload.planned_end_date - payload.planned_start_date).days
    if min_duration < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment duration ({min_duration} days) is below the minimum "
                   "60 days required for SACCO member cycle alignment."
        )

    # Calculate sample size
    calc = SampleSizeCalculator()
    size_result = calc.calculate(
        baseline_rate=payload.baseline_rate,
        minimum_detectable_effect=payload.minimum_detectable_effect,
        alpha=payload.significance_level,
        power=payload.target_power,
    )

    experiment = Experiment(
        experiment_key=payload.experiment_key,
        name=payload.name,
        description=payload.description,
        hypothesis_null=payload.hypothesis_null,
        hypothesis_alternative=payload.hypothesis_alternative,
        primary_kpi=payload.primary_kpi,
        guardrail_kpis=json.dumps(payload.guardrail_kpis),
        significance_level=payload.significance_level,
        target_power=payload.target_power,
        minimum_detectable_effect=payload.minimum_detectable_effect,
        target_sample_size_per_arm=size_result.n_per_arm,
        baseline_rate=payload.baseline_rate,
        eligibility_criteria=json.dumps(payload.eligibility_criteria),
        planned_start_date=payload.planned_start_date,
        planned_end_date=payload.planned_end_date,
        status=ExperimentStatus.DRAFT,
        designed_by=current_user.actor_token,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    AuditLogger(db).log(
        action="EXPERIMENT_CREATED",
        actor_token=current_user.actor_token,
        actor_role=current_user.role,
        resource_type="experiment",
        resource_id=experiment.id,
        experiment_id=experiment.id,
        details={"experiment_key": payload.experiment_key, "required_n_per_arm": size_result.n_per_arm},
    )

    return {
        "id": experiment.id,
        "experiment_key": experiment.experiment_key,
        "status": experiment.status,
        "required_n_per_arm": size_result.n_per_arm,
        "sample_size_notes": size_result.notes,
    }


@app.post("/experiments/{experiment_id}/approve")
async def approve_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("experiments:approve")

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status != ExperimentStatus.DRAFT:
        raise HTTPException(status_code=400, detail=f"Experiment is in {exp.status} status, not DRAFT")

    exp.status = ExperimentStatus.APPROVED
    exp.approved_by = current_user.actor_token
    from datetime import datetime, timezone
    exp.approved_at = datetime.now(timezone.utc)
    db.commit()

    AuditLogger(db).log(
        action="EXPERIMENT_APPROVED",
        actor_token=current_user.actor_token,
        actor_role=current_user.role,
        resource_type="experiment",
        resource_id=experiment_id,
        experiment_id=experiment_id,
    )

    return {"message": "Experiment approved", "status": exp.status}


@app.post("/experiments/{experiment_id}/enroll")
async def enroll_members(
    experiment_id: int,
    dry_run: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("experiments:approve")

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if not dry_run and exp.status != ExperimentStatus.APPROVED:
        raise HTTPException(status_code=400, detail="Experiment must be APPROVED before enrollment")

    engine = ExperimentEngine(db)
    summary = engine.enroll_members(exp, dry_run=dry_run)

    if not dry_run:
        AuditLogger(db).log(
            action="EXPERIMENT_ENROLLMENT_COMPLETED",
            actor_token=current_user.actor_token,
            actor_role=current_user.role,
            resource_type="experiment",
            resource_id=experiment_id,
            experiment_id=experiment_id,
            details={
                "total_enrolled": summary["total_eligible"],
                "control_n": summary["control_n"],
                "treatment_n": summary["treatment_n"],
                "dry_run": dry_run,
            },
        )

    return summary


@app.post("/experiments/{experiment_id}/suspend")
async def suspend_experiment(
    experiment_id: int,
    reason: str = Query(..., min_length=10),
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("experiments:suspend")

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp.status = ExperimentStatus.SUSPENDED
    exp.suspension_reason = reason
    db.commit()

    AuditLogger(db).log(
        action="EXPERIMENT_SUSPENDED",
        actor_token=current_user.actor_token,
        actor_role=current_user.role,
        resource_type="experiment",
        resource_id=experiment_id,
        experiment_id=experiment_id,
        details={"reason": reason},
    )

    return {"message": "Experiment suspended", "reason": reason}


@app.get("/experiments/")
async def list_experiments(
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("experiments:read_all")
    exps = db.query(Experiment).all()
    return [
        {
            "id": e.id,
            "key": e.experiment_key,
            "name": e.name,
            "status": e.status,
            "planned_start": e.planned_start_date,
            "planned_end": e.planned_end_date,
            "primary_kpi": e.primary_kpi,
        }
        for e in exps
    ]


# ── Analytics routes ──────────────────────────────────────────────

@app.post("/analysis/sample-size")
async def calculate_sample_size(
    req: SampleSizeRequest,
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("kpis:compute")
    calc = SampleSizeCalculator()
    result = calc.calculate(
        baseline_rate=req.baseline_rate,
        minimum_detectable_effect=req.mde,
        alpha=req.alpha,
        power=req.power,
    )
    response = {
        "n_per_arm": result.n_per_arm,
        "total_n": result.total_n,
        "baseline_rate": result.baseline_rate,
        "treatment_rate": result.treatment_rate,
        "alpha": result.alpha,
        "power": result.power,
        "notes": result.notes,
    }
    if req.eligible_member_count:
        response["feasibility"] = calc.check_feasibility(
            result.n_per_arm, req.eligible_member_count
        )
    return response


@app.get("/analysis/experiments/{experiment_id}/kpis")
async def get_experiment_kpis(
    experiment_id: int,
    window_days: int = Query(90, ge=30, le=180),
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("results:read_aggregated")

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    kpi_service = KPIService(db, min_cohort_size=settings.min_cohort_display_size)

    retention = kpi_service.compute_retention_rate(experiment_id, window_days)
    repayment = kpi_service.compute_on_time_repayment_rate(experiment_id)
    savings = kpi_service.compute_savings_consistency_rate(experiment_id)

    AuditLogger(db).log(
        action="KPI_RESULTS_QUERIED",
        actor_token=current_user.actor_token,
        actor_role=current_user.role,
        resource_type="experiment_kpis",
        resource_id=experiment_id,
        experiment_id=experiment_id,
        details={"window_days": window_days},
    )

    return {
        "experiment_id": experiment_id,
        "experiment_key": exp.experiment_key,
        "window_days": window_days,
        "retention_rate": retention,
        "on_time_repayment": repayment,
        "savings_consistency": savings,
    }


@app.get("/analysis/experiments/{experiment_id}/test")
async def run_statistical_test(
    experiment_id: int,
    window_days: int = Query(90, ge=30),
    db: Session = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    current_user.require_permission("results:read_aggregated")

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    kpi_service = KPIService(db, min_cohort_size=settings.min_cohort_display_size)
    retention = kpi_service.compute_retention_rate(experiment_id, window_days)

    if "error" in retention:
        raise HTTPException(status_code=422, detail=retention["error"])

    groups = retention.get("groups", {})
    control = groups.get("control", {})
    treatment = groups.get("treatment", {})

    if not control or not treatment:
        raise HTTPException(status_code=422, detail="Insufficient data for both groups")

    result = test_binary_kpi(
        control_successes=control["successes"],
        control_total=control["n"],
        treatment_successes=treatment["successes"],
        treatment_total=treatment["n"],
        alpha=float(exp.significance_level),
        kpi_name=f"retention_{window_days}d",
    )

    return {
        "experiment_key": exp.experiment_key,
        "kpi": result.kpi_name,
        "control": {"n": result.control_n, "rate": result.control_rate},
        "treatment": {"n": result.treatment_n, "rate": result.treatment_rate},
        "absolute_effect": result.absolute_effect,
        "relative_effect": result.relative_effect,
        "test": {
            "name": result.test_result.test_name,
            "statistic": result.test_result.statistic,
            "p_value": result.test_result.p_value,
            "ci_lower": result.test_result.ci_lower,
            "ci_upper": result.test_result.ci_upper,
            "effect_size_cohens_h": result.test_result.effect_size,
            "null_rejected": result.test_result.null_rejected,
            "alpha": result.test_result.alpha,
        },
        "conclusion": result.test_result.conclusion,
        "warnings": result.test_result.warnings,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SACCO Retention API"}
