"""
tests/test_core.py
───────────────────
Core test suite for the SACCO Retention Analysis System.
Tests tokenization, A/B engine logic, and statistical functions.
"""
import os
import pytest
import numpy as np

# Set test environment before any imports
os.environ.setdefault("TOKENIZATION_SECRET_KEY",
                      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")  # 32-byte Fernet key (test only)
os.environ.setdefault("HMAC_SECRET_KEY", "a" * 64)
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret")
os.environ.setdefault("DATABASE_URL",
                      "postgresql://sacco_user:sacco_pass@localhost:5432/sacco_experiments_test")
os.environ.setdefault("DATABASE_TEST_URL",
                      "postgresql://sacco_user:sacco_pass@localhost:5432/sacco_experiments_test")
os.environ.setdefault("APP_ENV", "development")


# ── Tokenization tests ────────────────────────────────────────────

class TestTokenizationService:

    @pytest.fixture
    def tokenizer(self):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        import hmac as _hmac
        import secrets
        hmac_key = secrets.token_hex(32)
        from app.security.tokenization import TokenizationService
        return TokenizationService(key, hmac_key)

    def test_encrypt_decrypt_roundtrip(self, tokenizer):
        plaintext = "12345678"
        encrypted = tokenizer.encrypt(plaintext)
        assert encrypted != plaintext
        assert tokenizer.decrypt(encrypted) == plaintext

    def test_member_token_is_deterministic(self, tokenizer):
        """Same inputs must always produce same token."""
        token1 = tokenizer.generate_member_token("12345678", "0722123456", "ACC-001")
        token2 = tokenizer.generate_member_token("12345678", "0722123456", "ACC-001")
        assert token1 == token2

    def test_different_members_get_different_tokens(self, tokenizer):
        token1 = tokenizer.generate_member_token("12345678", "0722123456", "ACC-001")
        token2 = tokenizer.generate_member_token("87654321", "0733654321", "ACC-002")
        assert token1 != token2

    def test_assignment_token_range(self, tokenizer):
        """Assignment value must always be 0–99."""
        for i in range(200):
            val = tokenizer.generate_experiment_assignment_token(
                f"token_{i}", "test_experiment"
            )
            assert 0 <= val <= 99

    def test_assignment_is_deterministic(self, tokenizer):
        """Same member + experiment always gets same group."""
        v1 = tokenizer.generate_experiment_assignment_token("abc123", "exp_savings_v1")
        v2 = tokenizer.generate_experiment_assignment_token("abc123", "exp_savings_v1")
        assert v1 == v2

    def test_assignment_independence_across_experiments(self, tokenizer):
        """Same member gets independent assignments across different experiments."""
        values = [
            tokenizer.generate_experiment_assignment_token("token_abc", f"exp_{i}")
            for i in range(20)
        ]
        # Should not all be the same — independence check
        assert len(set(values)) > 1

    def test_group_balance(self, tokenizer):
        """With 1000 members, balance should be approximately 50/50."""
        control = 0
        treatment = 0
        for i in range(1000):
            val = tokenizer.generate_experiment_assignment_token(
                f"member_{i:05d}", "balance_test"
            )
            if val < 50:
                control += 1
            else:
                treatment += 1
        # Allow 3pp deviation from 50/50
        assert abs(control - treatment) / 1000 < 0.03, (
            f"Group imbalance too large: {control} control, {treatment} treatment"
        )

    def test_amount_buckets(self, tokenizer):
        assert tokenizer.bucket_amount(0) == "0"
        assert tokenizer.bucket_amount(500) == "1-1000"
        assert tokenizer.bucket_amount(1000) == "1-1000"
        assert tokenizer.bucket_amount(1001) == "1001-5000"
        assert tokenizer.bucket_amount(50000) == "10001-50000"
        assert tokenizer.bucket_amount(200000) == "100000+"

    def test_encrypt_empty_raises(self, tokenizer):
        with pytest.raises(ValueError):
            tokenizer.encrypt("")


# ── Sample size calculator tests ──────────────────────────────────

class TestSampleSizeCalculator:

    @pytest.fixture
    def calc(self):
        from app.experiments.ab_engine import SampleSizeCalculator
        return SampleSizeCalculator()

    def test_basic_calculation(self, calc):
        result = calc.calculate(
            baseline_rate=0.62,
            minimum_detectable_effect=0.06,
            alpha=0.05,
            power=0.80,
        )
        assert result.n_per_arm > 0
        assert result.total_n == result.n_per_arm * 2
        assert result.treatment_rate == pytest.approx(0.68, abs=0.01)

    def test_larger_mde_requires_smaller_sample(self, calc):
        r1 = calc.calculate(0.62, 0.05, 0.05, 0.80)
        r2 = calc.calculate(0.62, 0.10, 0.05, 0.80)
        assert r2.n_per_arm < r1.n_per_arm

    def test_higher_power_requires_larger_sample(self, calc):
        r1 = calc.calculate(0.62, 0.06, 0.05, 0.80)
        r2 = calc.calculate(0.62, 0.06, 0.05, 0.90)
        assert r2.n_per_arm > r1.n_per_arm

    def test_stricter_alpha_requires_larger_sample(self, calc):
        r1 = calc.calculate(0.62, 0.06, 0.05, 0.80)
        r2 = calc.calculate(0.62, 0.06, 0.01, 0.80)
        assert r2.n_per_arm > r1.n_per_arm

    def test_small_n_triggers_warning(self, calc):
        result = calc.calculate(0.62, 0.20, 0.05, 0.80)  # large MDE → small N
        # Should have warning about Fisher's test
        if result.n_per_arm < 200:
            assert any("Fisher" in note for note in result.notes)

    def test_infeasible_mde_raises(self, calc):
        with pytest.raises(ValueError):
            calc.calculate(0.90, 0.15, 0.05, 0.80)  # treatment_rate > 1

    def test_feasibility_check_feasible(self, calc):
        f = calc.check_feasibility(required_n_per_arm=150, eligible_member_count=500)
        assert f["feasible"] is True
        assert f["coverage_ratio"] > 1.0

    def test_feasibility_check_infeasible(self, calc):
        f = calc.check_feasibility(required_n_per_arm=300, eligible_member_count=400)
        assert f["feasible"] is False
        assert "UNDERPOWERED" in f["recommendation"]


# ── Statistical test suite ────────────────────────────────────────

class TestStatisticalTests:

    def test_fisher_exact_detects_large_effect(self):
        from app.analysis.statistical_tests import test_binary_kpi
        result = test_binary_kpi(
            control_successes=50,
            control_total=100,
            treatment_successes=75,
            treatment_total=100,
            alpha=0.05,
        )
        # 25pp effect should be easily detected
        assert result.test_result.null_rejected is True
        assert result.absolute_effect == pytest.approx(0.25, abs=0.01)

    def test_z_test_no_effect(self):
        from app.analysis.statistical_tests import test_binary_kpi
        result = test_binary_kpi(
            control_successes=310,
            control_total=500,
            treatment_successes=315,
            treatment_total=500,
            alpha=0.05,
        )
        # 1pp effect with N=500 should not be significant
        assert result.test_result.null_rejected is False

    def test_small_sample_uses_fisher(self):
        from app.analysis.statistical_tests import test_binary_kpi
        result = test_binary_kpi(
            control_successes=30,
            control_total=50,
            treatment_successes=38,
            treatment_total=50,
            alpha=0.05,
        )
        assert "Fisher" in result.test_result.test_name

    def test_large_sample_uses_z_test(self):
        from app.analysis.statistical_tests import test_binary_kpi
        result = test_binary_kpi(
            control_successes=620,
            control_total=1000,
            treatment_successes=680,
            treatment_total=1000,
            alpha=0.05,
        )
        assert "Z-Test" in result.test_result.test_name

    def test_wilcoxon_for_continuous(self):
        from app.analysis.statistical_tests import test_continuous_kpi
        rng = np.random.default_rng(99)
        control = list(rng.lognormal(8.0, 1.2, 200))
        treatment = list(rng.lognormal(8.3, 1.2, 200))
        result = test_continuous_kpi(control, treatment, alpha=0.05)
        assert "Wilcoxon" in result.test_result.test_name
        assert result.test_result.p_value > 0
        assert result.test_result.p_value <= 1

    def test_multiple_comparison_correction(self):
        from app.analysis.statistical_tests import apply_multiple_comparison_correction
        p_values = [0.001, 0.04, 0.08, 0.50]
        results = apply_multiple_comparison_correction(p_values, alpha=0.05)

        # First p-value (0.001) should be rejected after correction
        assert results[0]["rejected"] is True
        # Last p-value (0.50) should definitely not be rejected
        assert results[3]["rejected"] is False

    def test_ci_contains_true_effect(self):
        """CI should contain true effect in most simulations."""
        from app.analysis.statistical_tests import test_binary_kpi
        contained = 0
        true_effect = 0.08

        for seed in range(100):
            rng = np.random.default_rng(seed)
            n = 300
            c_succ = int(rng.binomial(n, 0.62))
            t_succ = int(rng.binomial(n, 0.62 + true_effect))
            result = test_binary_kpi(c_succ, n, t_succ, n, alpha=0.05)
            if result.test_result.ci_lower <= true_effect <= result.test_result.ci_upper:
                contained += 1

        # Should contain true effect ~95% of the time
        assert contained / 100 >= 0.88, f"CI coverage too low: {contained}%"

    def test_simulated_data_generation(self):
        from app.analysis.statistical_tests import simulate_experiment_data
        ctrl, trt = simulate_experiment_data(n_control=200, n_treatment=200, seed=42)
        assert len(ctrl) == 200
        assert len(trt) == 200
        assert "retained_90d" in ctrl.columns
        assert set(ctrl["group"]) == {"control"}
        assert set(trt["group"]) == {"treatment"}
        # Both should have values in [0, 1]
        assert ctrl["retained_90d"].isin([0, 1]).all()


# ── Auth tests ─────────────────────────────────────────────────────

class TestAuth:

    def test_token_creation_and_decoding(self):
        from app.security.auth import create_access_token, decode_token, Role
        token = create_access_token("user_001", Role.DATA_ANALYST, "actor_token_001")
        decoded = decode_token(token)
        assert decoded is not None
        assert decoded.role == Role.DATA_ANALYST
        assert decoded.sub == "user_001"

    def test_invalid_token_returns_none(self):
        from app.security.auth import decode_token
        result = decode_token("not.a.valid.token")
        assert result is None

    def test_role_permissions(self):
        from app.security.auth import ROLE_PERMISSIONS, Role
        assert "experiments:create" in ROLE_PERMISSIONS[Role.EXPERIMENT_DESIGNER]
        assert "experiments:create" not in ROLE_PERMISSIONS[Role.DATA_ANALYST]
        assert "experiments:approve" in ROLE_PERMISSIONS[Role.DATA_STEWARD]
        assert "audit:read_full" in ROLE_PERMISSIONS[Role.COMPLIANCE_OFFICER]

    def test_user_context_permission_check(self):
        from app.security.auth import create_access_token, get_user_context, Role
        token = create_access_token("u1", Role.DATA_ANALYST, "actor_001")
        ctx = get_user_context(token)
        assert ctx.has_permission("results:read_aggregated")
        assert not ctx.has_permission("experiments:approve")

    def test_permission_error_on_missing_permission(self):
        from app.security.auth import create_access_token, get_user_context, Role
        token = create_access_token("u1", Role.DATA_ANALYST, "actor_001")
        ctx = get_user_context(token)
        with pytest.raises(PermissionError):
            ctx.require_permission("experiments:approve")
