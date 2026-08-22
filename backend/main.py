import os
import logging
from dotenv import load_dotenv

# Load .env variables before any other imports
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from backend.routers import profiles
from backend.routers import review
from backend.database import engine, Base
import backend.models.profile   # noqa: F401
import backend.models.api_key   # noqa: F401

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trueprofile")

# ──────────────────────────────────────────────────────────────────
# Sentry (error monitoring) — only active when DSN is set
# ──────────────────────────────────────────────────────────────────
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=0.2,   # capture 20% of transactions for performance
        send_default_pii=False,
    )
    logger.info("Sentry error monitoring enabled.")
else:
    logger.info("SENTRY_DSN not set — error monitoring disabled (fine for local dev).")

# ──────────────────────────────────────────────────────────────────
# Rate Limiter — per-IP, in-memory (swap to Redis in production)
# ──────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ──────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueProfile AI",
    version="1.0.0",
    description="AI-powered fake social media profile detection API.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach the rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# Startup — apply Alembic migrations
# ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def run_migrations():
    """Apply all pending Alembic migrations at startup."""
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
    alembic_cfg.set_main_option(
        "script_location",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic")
    )
    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied successfully.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")

# Ensure uploads directory exists for multipart image uploads
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ──────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────
app.include_router(profiles.router)
app.include_router(review.router)

# ──────────────────────────────────────────────────────────────────
# Root endpoint (health check)
# ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
@limiter.limit("30/minute")
def read_root(request: Request):
    return {
        "service": "TrueProfile AI",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }
