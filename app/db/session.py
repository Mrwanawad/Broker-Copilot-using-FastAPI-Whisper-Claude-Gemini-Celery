import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base, Lead


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if url.startswith("sqlite"):
        Path("data").mkdir(exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False})
    if url.startswith("postgres://"):  # Render hands out the legacy scheme
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def _factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def SessionLocal() -> Session:  # noqa: N802 - reads like the usual factory name
    return _factory()()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(seed: bool = True) -> None:
    Base.metadata.create_all(get_engine())
    if not seed:
        return
    path = Path(get_settings().leads_path)
    if not path.exists():
        return
    with SessionLocal() as db:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if db.get(Lead, row["lead_id"]):
                continue
            core = {k: row.pop(k, None) for k in ("name", "phone", "owner")}
            lead_id = row.pop("lead_id")
            db.add(Lead(lead_id=lead_id, data=row, **core))
        db.commit()
