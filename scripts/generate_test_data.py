"""
scripts/generate_test_data.py
──────────────────────────────
Generates realistic synthetic SACCO data for local testing.

Produces:
  - 2,000 members (scalable via --members flag)
  - Savings accounts
  - Loan records (active + completed)
  - Transaction history (6 months)
  - Simulated M-Pesa logs

Run:
  python scripts/generate_test_data.py --members 2000 --seed 42
"""
import argparse
import json
import logging
import random
import sys
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker
import numpy as np

from app.models.db_models import (
    AssignmentGroup, Base, GeographicZone, Loan, LoanStatus,
    Member, MemberStatus, SavingsAccount, Transaction, TransactionType
)
from app.security.tokenization import get_tokenization_service
from config.settings import get_settings
from app.core.database import get_engine, get_session_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker("en_KE")  # Kenya locale
rng = None  # Will be set in main()


# ── Kenyan-specific reference data ──────────────────────────────

KENYAN_EMPLOYMENT_SECTORS = [
    ("public_sector", 0.35),     # Government, teachers, nurses
    ("private_sector", 0.25),
    ("agriculture", 0.20),
    ("jua_kali", 0.10),          # Informal sector
    ("business", 0.10),
]

LOAN_PRODUCTS = [
    ("salary_backed", 0.40),
    ("business", 0.25),
    ("emergency", 0.20),
    ("development", 0.15),
]

GEOGRAPHIC_ZONES = [
    (GeographicZone.URBAN_NAIROBI, 0.25),
    (GeographicZone.URBAN_OTHER, 0.20),
    (GeographicZone.PERI_URBAN, 0.25),
    (GeographicZone.RURAL, 0.30),
]

CHANNELS = {
    GeographicZone.URBAN_NAIROBI: [("mpesa", 0.50), ("mobile_app", 0.25), ("ussd", 0.15), ("branch", 0.10)],
    GeographicZone.URBAN_OTHER:   [("mpesa", 0.45), ("ussd", 0.25), ("branch", 0.20), ("mobile_app", 0.10)],
    GeographicZone.PERI_URBAN:    [("mpesa", 0.40), ("ussd", 0.30), ("branch", 0.25), ("mobile_app", 0.05)],
    GeographicZone.RURAL:         [("mpesa", 0.35), ("ussd", 0.35), ("branch", 0.30)],
}


def weighted_choice(options):
    items, weights = zip(*options)
    return rng.choice(items, p=np.array(weights) / sum(weights))


def random_kenyan_phone():
    """Generate realistic Kenyan mobile number."""
    prefix = rng.choice(["0722", "0733", "0712", "0700", "0710", "0720"])
    suffix = "".join([str(rng.integers(0, 10)) for _ in range(6)])
    return f"{prefix}{suffix}"


def random_national_id():
    """Kenyan National ID: 7-8 digit number."""
    return str(rng.integers(10000000, 40000000))


def bucket_amount(amount: float) -> str:
    if amount <= 1000: return "1-1000"
    elif amount <= 5000: return "1001-5000"
    elif amount <= 10000: return "5001-10000"
    elif amount <= 50000: return "10001-50000"
    elif amount <= 100000: return "50001-100000"
    return "100000+"


def generate_members(n: int, tokenizer) -> list[dict]:
    logger.info(f"Generating {n} members...")
    members = []

    for i in range(n):
        national_id = random_national_id()
        phone = random_kenyan_phone()
        full_name = fake.name()
        member_number = f"SACCO-{i+1:06d}"
        account_number = f"ACC-{i+1:06d}"

        # Generate token
        member_token = tokenizer.generate_member_token(
            national_id, phone, account_number
        )

        # Encrypt PII
        national_id_enc = tokenizer.encrypt(national_id)
        phone_enc = tokenizer.encrypt(phone)
        name_enc = tokenizer.encrypt(full_name)

        # Join date: 1–10 years ago, weighted toward 2–5 years
        days_ago = int(rng.choice(
            range(90, 3650),
            p=np.array([
                1 if d < 365 else
                3 if d < 1825 else
                2 if d < 2920 else 1
                for d in range(90, 3650)
            ], dtype=float) / sum(
                1 if d < 365 else
                3 if d < 1825 else
                2 if d < 2920 else 1
                for d in range(90, 3650)
            )
        ))
        join_date = date.today() - timedelta(days=days_ago)

        zone = weighted_choice(GEOGRAPHIC_ZONES)
        sector = weighted_choice(KENYAN_EMPLOYMENT_SECTORS)
        loan_product = weighted_choice(LOAN_PRODUCTS)

        # Status — mostly active
        status_roll = rng.random()
        if status_roll < 0.82:
            status = MemberStatus.ACTIVE
        elif status_roll < 0.92:
            status = MemberStatus.DORMANT
        elif status_roll < 0.97:
            status = MemberStatus.SUSPENDED
        else:
            status = MemberStatus.EXITED

        last_activity = date.today() - timedelta(days=int(rng.integers(1, 180)))

        members.append({
            "member_token": member_token,
            "member_number": member_number,
            "national_id_encrypted": national_id_enc,
            "phone_encrypted": phone_enc,
            "full_name_encrypted": name_enc,
            "gender": rng.choice(["M", "F"], p=[0.52, 0.48]),
            "geographic_zone": zone,
            "join_date": join_date,
            "employment_sector": sector,
            "loan_product_type": loan_product,
            "status": status,
            "is_eligible_for_experiments": status == MemberStatus.ACTIVE,
            "last_activity_date": last_activity,
        })

    return members


def generate_savings_accounts(members_db: list) -> list[dict]:
    logger.info("Generating savings accounts...")
    accounts = []
    for member in members_db:
        account_types = ["ordinary"]
        if rng.random() < 0.30:
            account_types.append("holiday")
        if rng.random() < 0.15:
            account_types.append("fixed_deposit")

        for acc_type in account_types:
            balance = float(rng.lognormal(mean=9.5, sigma=1.5))  # KSh, lognormal
            accounts.append({
                "member_id": member.id,
                "account_number": f"SAV-{member.id:06d}-{acc_type[:3].upper()}",
                "account_type": acc_type,
                "balance": round(balance, 2),
                "share_capital_units": int(rng.integers(1, 200)),
                "is_active": rng.random() > 0.05,
                "opened_date": member.join_date + timedelta(days=int(rng.integers(0, 30))),
                "last_transaction_date": date.today() - timedelta(days=int(rng.integers(1, 90))),
            })
    return accounts


def generate_loans(members_db: list) -> list[dict]:
    logger.info("Generating loans...")
    loans = []
    loan_counter = 1

    for member in members_db:
        # Not all members have loans
        if rng.random() > 0.65:
            continue

        # Number of loan cycles: 1–4 based on tenure
        tenure_days = (date.today() - member.join_date).days
        max_cycles = min(4, max(1, tenure_days // 365))
        n_loans = int(rng.integers(1, max_cycles + 1))

        for cycle in range(1, n_loans + 1):
            principal = float(rng.choice([
                20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000
            ], p=[0.15, 0.20, 0.25, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01]))

            term_months = int(rng.choice([6, 12, 18, 24, 36], p=[0.15, 0.35, 0.25, 0.15, 0.10]))
            rate = float(rng.choice([0.12, 0.13, 0.14, 0.15], p=[0.25, 0.35, 0.25, 0.15]))
            monthly_rate = rate / 12
            monthly_installment = (
                principal * monthly_rate * (1 + monthly_rate) ** term_months
                / ((1 + monthly_rate) ** term_months - 1)
            )

            # Disbursement date based on cycle
            disburse_days_ago = int(
                (tenure_days / n_loans) * (n_loans - cycle + 1)
                + rng.integers(-30, 30)
            )
            disbursement_date = date.today() - timedelta(days=max(0, disburse_days_ago))
            expected_completion = disbursement_date + timedelta(days=term_months * 30)
            actual_completion = None

            # Loan status
            days_elapsed = (date.today() - disbursement_date).days
            months_elapsed = days_elapsed / 30

            if cycle < n_loans:
                status = LoanStatus.COMPLETED
                actual_completion = disbursement_date + timedelta(days=term_months * 30)
                outstanding = 0.0
                total_repaid = principal * 1.05  # approximate total with interest
            elif months_elapsed >= term_months:
                roll = rng.random()
                if roll < 0.88:
                    status = LoanStatus.COMPLETED
                    actual_completion = expected_completion
                    outstanding = 0.0
                    total_repaid = principal * 1.05
                elif roll < 0.95:
                    status = LoanStatus.DEFAULTED
                    outstanding = float(rng.uniform(0.1, 0.5)) * principal
                    total_repaid = principal - outstanding
                else:
                    status = LoanStatus.RESTRUCTURED
                    outstanding = float(rng.uniform(0.2, 0.6)) * principal
                    total_repaid = principal - outstanding
            else:
                status = LoanStatus.REPAYING
                fraction_paid = min(months_elapsed / term_months, 0.95)
                total_repaid = principal * fraction_paid
                outstanding = principal - total_repaid

            days_past_due = 0
            if status == LoanStatus.REPAYING:
                days_past_due = max(0, int(rng.integers(-10, 45)))

            loans.append({
                "member_id": member.id,
                "loan_number": f"LN-{loan_counter:07d}",
                "product_type": member.loan_product_type or "salary_backed",
                "principal_amount": round(principal, 2),
                "interest_rate": round(rate, 4),
                "term_months": term_months,
                "monthly_installment": round(monthly_installment, 2),
                "outstanding_balance": round(max(0, outstanding), 2),
                "total_repaid": round(max(0, total_repaid), 2),
                "status": status,
                "disbursement_date": disbursement_date,
                "expected_completion_date": expected_completion,
                "actual_completion_date": actual_completion,
                "days_past_due": days_past_due,
                "loan_cycle": cycle,
            })
            loan_counter += 1

    return loans


def generate_transactions(
    members_db: list,
    savings_db: list,
    loans_db: list,
    months_back: int = 6,
) -> list[dict]:
    logger.info(f"Generating transactions ({months_back} months)...")
    transactions = []
    savings_by_member = {s.member_id: s for s in savings_db if s.is_active}
    loans_by_member = {}
    for loan in loans_db:
        if loan.member_id not in loans_by_member:
            loans_by_member[loan.member_id] = []
        loans_by_member[loan.member_id].append(loan)

    txn_counter = 1

    for member in members_db:
        if member.status not in [MemberStatus.ACTIVE, MemberStatus.DORMANT]:
            continue

        # Channel preference by zone
        zone_channels = CHANNELS.get(member.geographic_zone, CHANNELS[GeographicZone.RURAL])
        savings_acct = savings_by_member.get(member.id)
        member_loans = loans_by_member.get(member.id, [])

        # Activity level: dormant members have fewer transactions
        is_dormant = member.status == MemberStatus.DORMANT
        activity_multiplier = 0.15 if is_dormant else 1.0

        # Generate savings deposits (monthly-ish)
        for month_offset in range(months_back):
            if rng.random() > (0.75 * activity_multiplier):
                continue
            txn_date = date.today() - timedelta(
                days=month_offset * 30 + int(rng.integers(0, 28))
            )
            amount = float(rng.lognormal(mean=8.0, sigma=0.8))
            channel = weighted_choice(zone_channels)

            transactions.append({
                "member_id": member.id,
                "savings_account_id": savings_acct.id if savings_acct else None,
                "loan_id": None,
                "transaction_type": TransactionType.SAVINGS_DEPOSIT,
                "amount": round(amount, 2),
                "amount_bucket": bucket_amount(amount),
                "channel": channel,
                "reference_number": f"TXN-{txn_counter:09d}",
                "mpesa_transaction_id_hash": None,
                "transaction_date": datetime.combine(txn_date, datetime.min.time()),
                "value_date": txn_date,
                "is_on_time": None,
                "notes": None,
            })
            txn_counter += 1

        # Generate loan repayments
        for loan in member_loans:
            if loan.status not in [LoanStatus.REPAYING, LoanStatus.COMPLETED]:
                continue

            repayment_months = min(
                loan.term_months,
                (date.today() - loan.disbursement_date).days // 30
            )

            for r in range(repayment_months):
                if rng.random() > (0.80 * activity_multiplier):
                    continue

                due_date = loan.disbursement_date + timedelta(days=(r + 1) * 30)
                # Simulate on-time vs late payment
                on_time_prob = 0.82 if member.geographic_zone in [
                    GeographicZone.URBAN_NAIROBI, GeographicZone.URBAN_OTHER
                ] else 0.72

                actual_days_offset = (
                    int(rng.integers(-3, 3)) if rng.random() < on_time_prob
                    else int(rng.integers(1, 30))
                )
                actual_date = due_date + timedelta(days=actual_days_offset)
                if actual_date > date.today():
                    continue

                is_on_time = actual_days_offset <= 0
                channel = weighted_choice(zone_channels)
                amount = float(loan.monthly_installment)

                transactions.append({
                    "member_id": member.id,
                    "savings_account_id": None,
                    "loan_id": loan.id,
                    "transaction_type": TransactionType.LOAN_REPAYMENT,
                    "amount": round(amount, 2),
                    "amount_bucket": bucket_amount(amount),
                    "channel": channel,
                    "reference_number": f"TXN-{txn_counter:09d}",
                    "mpesa_transaction_id_hash": None,
                    "transaction_date": datetime.combine(actual_date, datetime.min.time()),
                    "value_date": actual_date,
                    "is_on_time": is_on_time,
                    "notes": None,
                })
                txn_counter += 1

    logger.info(f"Generated {len(transactions)} transactions")
    return transactions


def load_to_database(n_members: int, seed: int):
    """Load all generated data into the database."""
    global rng
    rng = np.random.default_rng(seed)
    random.seed(seed)
    fake.seed_locale("en_KE", seed)

    settings = get_settings()
    engine = get_engine(settings.database_url)
    SessionFactory = get_session_factory(engine)

    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    tokenizer = get_tokenization_service()

    with SessionFactory() as session:
        # 1. Members
        member_dicts = generate_members(n_members, tokenizer)
        member_objects = [Member(**m) for m in member_dicts]
        session.add_all(member_objects)
        session.flush()  # Get IDs
        logger.info(f"Inserted {len(member_objects)} members")

        # 2. Savings accounts
        savings_dicts = generate_savings_accounts(member_objects)
        savings_objects = [SavingsAccount(**s) for s in savings_dicts]
        session.add_all(savings_objects)
        session.flush()
        logger.info(f"Inserted {len(savings_objects)} savings accounts")

        # 3. Loans
        loan_dicts = generate_loans(member_objects)
        loan_objects = [Loan(**l) for l in loan_dicts]
        session.add_all(loan_objects)
        session.flush()
        logger.info(f"Inserted {len(loan_objects)} loans")

        # 4. Transactions
        txn_dicts = generate_transactions(member_objects, savings_objects, loan_objects)
        # Insert in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(txn_dicts), batch_size):
            batch = [Transaction(**t) for t in txn_dicts[i:i + batch_size]]
            session.add_all(batch)
        logger.info(f"Inserted {len(txn_dicts)} transactions")

        session.commit()
        logger.info("✅ Test data generation complete.")

    # Print summary
    with SessionFactory() as session:
        from sqlalchemy import func
        from app.models.db_models import Member, Loan, Transaction, SavingsAccount
        print("\n📊 Data Summary:")
        print(f"  Members:           {session.query(func.count(Member.id)).scalar():,}")
        print(f"  Savings Accounts:  {session.query(func.count(SavingsAccount.id)).scalar():,}")
        print(f"  Loans:             {session.query(func.count(Loan.id)).scalar():,}")
        print(f"  Transactions:      {session.query(func.count(Transaction.id)).scalar():,}")
        active = session.query(func.count(Member.id)).filter(
            Member.status == MemberStatus.ACTIVE
        ).scalar()
        print(f"  Active Members:    {active:,}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SACCO test data")
    parser.add_argument("--members", type=int, default=2000, help="Number of members to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    load_to_database(args.members, args.seed)
