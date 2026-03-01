"""
app/analysis/statistical_tests.py
───────────────────────────────────
Statistical analysis module for SACCO retention experiments.

Test selection logic:
  - Binary KPIs, N > 200 per arm  → Two-proportion z-test
  - Binary KPIs, N < 200 per arm  → Fisher's exact test  (DEFAULT for most SACCOs)
  - Continuous/skewed amounts      → Wilcoxon rank-sum (Mann-Whitney U)
  - Time-to-event / survival       → Kaplan-Meier + log-rank test
  - Multiple comparisons           → Bonferroni or Holm-Bonferroni correction

All functions return structured result dicts — no bare floats.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────

@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float              # Cohen's h for proportions, r for Wilcoxon
    null_rejected: bool
    alpha: float
    conclusion: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class KPIResult:
    kpi_name: str
    control_n: int
    treatment_n: int
    control_rate: float
    treatment_rate: float
    absolute_effect: float
    relative_effect: float
    test_result: TestResult


# ── Core test functions ───────────────────────────────────────────

def test_binary_kpi(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alpha: float = 0.05,
    kpi_name: str = "retention",
) -> KPIResult:
    """
    Test a binary KPI (e.g., retained vs not retained).

    Automatically selects:
      - Fisher's exact test if either arm N < 200
      - Two-proportion z-test otherwise

    Args:
        control_successes: Members who met the KPI definition in control group
        control_total: Total members in control group
        treatment_successes: Members who met the KPI definition in treatment group
        treatment_total: Total members in treatment group
        alpha: Significance level
        kpi_name: Label for the KPI being tested

    Returns:
        KPIResult with all test statistics and interpretation
    """
    warnings = []

    if control_total == 0 or treatment_total == 0:
        raise ValueError("Cannot test with empty groups")

    p_control = control_successes / control_total
    p_treatment = treatment_successes / treatment_total
    absolute_effect = p_treatment - p_control
    relative_effect = absolute_effect / p_control if p_control > 0 else 0

    # Select test based on sample size
    use_fisher = control_total < 200 or treatment_total < 200

    if use_fisher:
        # Fisher's exact test — no distributional assumptions
        contingency = np.array([
            [control_successes, control_total - control_successes],
            [treatment_successes, treatment_total - treatment_successes],
        ])
        oddsratio, p_value = fisher_exact(contingency, alternative="two-sided")
        statistic = oddsratio
        test_name = "Fisher's Exact Test"

        # Confidence interval using Wilson score method for each proportion
        ci_lower_c, ci_upper_c = _wilson_ci(control_successes, control_total, alpha)
        ci_lower_t, ci_upper_t = _wilson_ci(treatment_successes, treatment_total, alpha)

        # CI on the difference (Newcombe method approximation)
        ci_lower = (p_treatment - p_control) - 1.96 * np.sqrt(
            p_control * (1 - p_control) / control_total
            + p_treatment * (1 - p_treatment) / treatment_total
        )
        ci_upper = (p_treatment - p_control) + 1.96 * np.sqrt(
            p_control * (1 - p_control) / control_total
            + p_treatment * (1 - p_treatment) / treatment_total
        )

        warnings.append(
            f"Sample size (control={control_total}, treatment={treatment_total}) "
            "is small. Fisher's exact test used. Results should be interpreted cautiously."
        )
    else:
        # Two-proportion z-test with continuity correction
        count = np.array([treatment_successes, control_successes])
        nobs = np.array([treatment_total, control_total])

        from statsmodels.stats.proportion import proportions_ztest, proportion_confint
        statistic, p_value = proportions_ztest(count, nobs)
        test_name = "Two-Proportion Z-Test"

        # 95% CI on the difference
        se = np.sqrt(
            p_control * (1 - p_control) / control_total
            + p_treatment * (1 - p_treatment) / treatment_total
        )
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = absolute_effect - z_crit * se
        ci_upper = absolute_effect + z_crit * se

    # Cohen's h effect size for proportions
    h = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))

    # ensure Python bool (numpy.bool_ can break json.dumps)
    null_rejected = bool(p_value < alpha)

    conclusion = _build_conclusion(
        kpi_name, p_control, p_treatment, absolute_effect,
        p_value, alpha, ci_lower, ci_upper, null_rejected
    )

    return KPIResult(
        kpi_name=kpi_name,
        control_n=control_total,
        treatment_n=treatment_total,
        control_rate=round(p_control, 6),
        treatment_rate=round(p_treatment, 6),
        absolute_effect=round(absolute_effect, 6),
        relative_effect=round(relative_effect, 6),
        test_result=TestResult(
            test_name=test_name,
            statistic=round(float(statistic), 6),
            p_value=round(float(p_value), 8),
            ci_lower=round(float(ci_lower), 6),
            ci_upper=round(float(ci_upper), 6),
            effect_size=round(float(h), 6),
            null_rejected=null_rejected,
            alpha=alpha,
            conclusion=conclusion,
            warnings=warnings,
        ),
    )


def test_continuous_kpi(
    control_values: list[float],
    treatment_values: list[float],
    alpha: float = 0.05,
    kpi_name: str = "repayment_amount",
) -> KPIResult:
    """
    Test a continuous KPI (e.g., average repayment amount, savings balance change).

    Uses Wilcoxon rank-sum (Mann-Whitney U) — appropriate for:
      - Non-normal distributions (common in financial data)
      - Small sample sizes
      - Heavily right-skewed data (typical for SACCO amounts)
    """
    warnings = []
    c = np.array(control_values)
    t = np.array(treatment_values)

    if len(c) < 10 or len(t) < 10:
        warnings.append("Very small sample — results are highly uncertain.")

    statistic, p_value = mannwhitneyu(t, c, alternative="two-sided")

    # Effect size: rank-biserial correlation r = 1 - 2U / (n1 * n2)
    n1, n2 = len(t), len(c)
    r_effect = 1 - (2 * statistic) / (n1 * n2)

    # Bootstrap CI on the median difference
    ci_lower, ci_upper = _bootstrap_median_ci(c, t, alpha, n_bootstrap=2000)

    p_control = float(np.median(c))
    p_treatment = float(np.median(t))
    absolute_effect = p_treatment - p_control
    relative_effect = absolute_effect / p_control if p_control != 0 else 0

    # numpy.bool_ conversion
    null_rejected = bool(p_value < alpha)

    conclusion = (
        f"Wilcoxon rank-sum test for '{kpi_name}': "
        f"median control={p_control:.2f}, median treatment={p_treatment:.2f}. "
        f"p={p_value:.4f}. "
        + ("Null hypothesis rejected." if null_rejected
           else "Null hypothesis NOT rejected.")
    )

    return KPIResult(
        kpi_name=kpi_name,
        control_n=len(c),
        treatment_n=len(t),
        control_rate=p_control,
        treatment_rate=p_treatment,
        absolute_effect=round(absolute_effect, 4),
        relative_effect=round(relative_effect, 4),
        test_result=TestResult(
            test_name="Wilcoxon Rank-Sum (Mann-Whitney U)",
            statistic=round(float(statistic), 4),
            p_value=round(float(p_value), 8),
            ci_lower=round(float(ci_lower), 4),
            ci_upper=round(float(ci_upper), 4),
            effect_size=round(float(r_effect), 6),
            null_rejected=null_rejected,
            alpha=alpha,
            conclusion=conclusion,
            warnings=warnings,
        ),
    )


def apply_multiple_comparison_correction(
    p_values: list[float],
    alpha: float = 0.05,
    method: str = "holm",
) -> list[dict]:
    """
    Apply Holm-Bonferroni step-down correction for multiple KPIs.

    Args:
        p_values: List of raw p-values from individual tests
        alpha: Family-wise error rate
        method: "holm" (recommended) or "bonferroni" (more conservative)

    Returns:
        List of dicts with raw_p, corrected_p, and rejected flags
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n

    if method == "bonferroni":
        for orig_idx, p in indexed:
            corrected = min(p * n, 1.0)
            results[orig_idx] = {
                "original_index": orig_idx,
                "raw_p": p,
                "corrected_p": round(corrected, 8),
                "rejected": corrected < alpha,
            }
    else:  # Holm
        reject_all_above = False
        for rank, (orig_idx, p) in enumerate(indexed):
            if reject_all_above:
                corrected = 1.0
                rejected = False
            else:
                corrected = min(p * (n - rank), 1.0)
                rejected = corrected < alpha
                if not rejected:
                    reject_all_above = True
            results[orig_idx] = {
                "original_index": orig_idx,
                "raw_p": p,
                "corrected_p": round(corrected, 8),
                "rejected": rejected,
                "method": "Holm-Bonferroni",
            }

    return results


# ── Simulation / test data helper ────────────────────────────────

def simulate_experiment_data(
    n_control: int = 150,
    n_treatment: int = 150,
    control_retention_rate: float = 0.62,
    treatment_effect: float = 0.06,   # True effect we're testing for
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic experiment data for testing and demonstration.

    Returns two DataFrames: control_df, treatment_df
    Each row is one member's outcomes over the observation window.
    """
    rng = np.random.default_rng(seed)

    def make_cohort(n, retention_rate, group_label):
        retained = rng.binomial(1, retention_rate, n)
        months_active = rng.integers(0, 4, n)  # 0–3 months active
        avg_monthly_savings = rng.lognormal(mean=8.5, sigma=1.2, size=n)  # KSh amounts
        on_time_repayment = rng.binomial(1, retention_rate + 0.05, n)
        loan_cycle = rng.choice([1, 2, 3, 4], n, p=[0.45, 0.30, 0.15, 0.10])

        return pd.DataFrame({
            "member_token": [f"token_{group_label}_{i:05d}" for i in range(n)],
            "group": group_label,
            "retained_90d": retained,
            "months_active": months_active,
            "avg_monthly_savings_ksh": avg_monthly_savings.round(2),
            "on_time_repayment": on_time_repayment,
            "loan_cycle": loan_cycle,
        })

    control_df = make_cohort(n_control, control_retention_rate, "control")
    treatment_df = make_cohort(
        n_treatment, control_retention_rate + treatment_effect, "treatment"
    )
    return control_df, treatment_df


# ── Private helpers ───────────────────────────────────────────────

def _wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — more accurate than normal approximation for small N."""
    if total == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


def _bootstrap_median_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    """Bootstrap CI on median difference (treatment - control)."""
    rng = np.random.default_rng(123)
    diffs = []
    for _ in range(n_bootstrap):
        c_sample = rng.choice(control, len(control), replace=True)
        t_sample = rng.choice(treatment, len(treatment), replace=True)
        diffs.append(np.median(t_sample) - np.median(c_sample))
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _build_conclusion(
    kpi_name, p_control, p_treatment, absolute_effect,
    p_value, alpha, ci_lower, ci_upper, null_rejected
) -> str:
    direction = "increased" if absolute_effect > 0 else "decreased"
    decision = "REJECT null hypothesis" if null_rejected else "FAIL TO REJECT null hypothesis"
    return (
        f"{kpi_name}: Control={p_control:.1%}, Treatment={p_treatment:.1%}. "
        f"Treatment {direction} KPI by {abs(absolute_effect):.1%} "
        f"(95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]). "
        f"p={p_value:.4f} (α={alpha}). "
        f"{decision}. "
        + ("Feature shows statistically significant improvement."
           if null_rejected and absolute_effect > 0
           else "No statistically significant improvement detected.")
    )
