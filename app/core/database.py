"""
app/core/database.py
─────────────────────
Database connection management using SQLAlchemy.
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings


def get_engine(database_url: str = None) -> Engine:
    settings = get_settings()
    url = database_url or settings.database_url
    return create_engine(
        url,
        pool_pre_ping=True,       # Detect stale connections
        pool_size=5,
        max_overflow=10,
        echo=settings.is_development,  # Log SQL in dev only
    )


def get_session_factory(engine: Engine = None) -> sessionmaker:
    eng = engine or get_engine()
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=eng,
        class_=Session,
    )


# Convenience dependency for FastAPI
_engine = None
_SessionFactory = None


def init_db():
    global _engine, _SessionFactory
    _engine = get_engine()
    _SessionFactory = get_session_factory(_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    if _SessionFactory is None:
        init_db()
    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()
