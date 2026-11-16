import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import api, hooks
from app.db.session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Broker Copilot",
    description="Calls and WhatsApp -> transcript, structured summary, CRM diff, matched units, follow-up draft.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api, prefix="/api")
app.include_router(hooks)


@app.get("/health", tags=["monitoring"])
def health() -> dict:
    return {"status": "ok"}
