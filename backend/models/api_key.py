from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime, timezone
from backend.database import Base


class APIKey(Base):
    """
    Stores API keys for authenticating clients.
    Keys are stored as-is (for MVP); in production hash them with bcrypt.
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    owner = Column(String, nullable=False)           # name/email of key owner
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
