from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import profiles
from backend.database import engine, Base
import backend.models.profile  # noqa: F401 - ensures model is registered with Base

app = FastAPI(title="TrueProfile AI", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production: list specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def run_migrations():
    """
    Apply all pending Alembic migrations at startup.
    This is the production-safe way to keep the schema up-to-date
    (as opposed to create_all which would silently skip schema changes).
    """
    from alembic.config import Config
    from alembic import command
    import os

    alembic_cfg = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
    alembic_cfg.set_main_option(
        "script_location",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic")
    )
    command.upgrade(alembic_cfg, "head")

app.include_router(profiles.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to TrueProfile AI backend"}
