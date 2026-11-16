"""CRM adapters. `local` keeps leads in the app database (demo); `http` pushes approved updates to an
external CRM REST API (HubSpot/Zoho/in-house) with a PATCH per lead."""

import logging
from typing import Protocol

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Lead

log = logging.getLogger(__name__)


class CRMClient(Protocol):
    def get_lead(self, lead_id: str) -> dict | None: ...

    def update_lead(self, lead_id: str, fields: dict) -> dict: ...


class LocalCRM:
    def __init__(self, db: Session):
        self.db = db

    def get_lead(self, lead_id: str) -> dict | None:
        lead = self.db.get(Lead, lead_id)
        return lead.as_dict() if lead else None

    def update_lead(self, lead_id: str, fields: dict) -> dict:
        lead = self.db.get(Lead, lead_id)
        if lead is None:
            raise KeyError(lead_id)
        data = dict(lead.data or {})
        data.update({k: v for k, v in fields.items() if k not in ("lead_id",)})
        lead.data = data
        self.db.commit()
        return lead.as_dict()


class HttpCRM:
    """Reads go to the local mirror; writes are PATCHed to the external CRM, then mirrored."""

    def __init__(self, db: Session):
        s = get_settings()
        self.local = LocalCRM(db)
        self.http = httpx.Client(base_url=s.crm_api_url, timeout=15,
                                 headers={"Authorization": f"Bearer {s.crm_api_token}"})

    def get_lead(self, lead_id: str) -> dict | None:
        return self.local.get_lead(lead_id)

    def update_lead(self, lead_id: str, fields: dict) -> dict:
        res = self.http.patch(f"/leads/{lead_id}", json=fields)
        res.raise_for_status()
        return self.local.update_lead(lead_id, fields)


def get_crm(db: Session) -> CRMClient:
    return HttpCRM(db) if get_settings().crm_backend == "http" else LocalCRM(db)
