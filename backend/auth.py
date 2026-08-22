"""
API Key authentication dependency.

Usage:
    Pass `X-API-Key: <key>` header on protected endpoints.

To generate a key for development, run:
    python scripts/generate_api_key.py

The BYPASS_AUTH=true env var disables auth entirely for local dev.
"""
import os
from dotenv import load_dotenv
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

load_dotenv()

from backend.database import get_db
from backend.models.api_key import APIKey

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency that validates the X-API-Key header.
    Skipped entirely when BYPASS_AUTH=true or not configured in dev.
    """
    bypass = os.getenv("BYPASS_AUTH", "true").lower() == "true"
    if bypass:
        return "dev-bypass"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # Import here to avoid circular import at module load time
    from backend.database import SessionLocal
    with SessionLocal() as session:
        db_key = session.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True,  # noqa: E712
        ).first()

    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key",
        )

    return db_key.owner
