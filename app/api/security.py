import base64
import hashlib
import hmac

from fastapi import Header, HTTPException, Request

from app.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Returns the actor name bound to the key (written to approvals and corrections)."""
    actor = get_settings().api_key_map.get(x_api_key or "")
    if not actor:
        raise HTTPException(401, "invalid or missing X-API-Key")
    return actor


def twilio_signature_ok(url: str, params: dict[str, str], signature: str, auth_token: str) -> bool:
    """Twilio request validation: HMAC-SHA1 over the URL plus sorted POST params, base64-encoded."""
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


async def verify_twilio(request: Request) -> dict[str, str]:
    form = {k: str(v) for k, v in (await request.form()).items()}
    token = get_settings().twilio_auth_token
    if token and not twilio_signature_ok(str(request.url), form, request.headers.get("X-Twilio-Signature", ""), token):
        raise HTTPException(403, "bad Twilio signature")
    return form
