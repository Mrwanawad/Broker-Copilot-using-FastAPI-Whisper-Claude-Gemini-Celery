"""Celery app. With BC_REDIS_URL empty, tasks run eagerly in the API process (local demo, tests)."""

from celery import Celery

from app.config import get_settings

_redis = get_settings().redis_url
celery_app = Celery("broker_copilot", broker=_redis or "memory://", backend=_redis or "cache+memory://")
celery_app.conf.update(
    task_always_eager=not _redis,
    task_acks_late=True,  # a crashed worker re-delivers the call instead of losing it
    worker_prefetch_multiplier=1,  # ASR jobs are long; do not hoard them
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    task_default_queue="calls",
)


@celery_app.task(name="process_call", bind=True, max_retries=2, default_retry_delay=30)
def process_call_task(self, call_id: int) -> str:
    from app.db.session import SessionLocal
    from app.pipeline import process_call

    with SessionLocal() as db:
        call = process_call(db, call_id)
        return call.status


def enqueue(call_id: int) -> None:
    process_call_task.delay(call_id)
