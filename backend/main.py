from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import profiles
from backend.database import engine, Base
import backend.models.profile  # noqa: F401 - ensures model is registered with Base

app = FastAPI(title="TrueProfile AI", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
Base.metadata.create_all(bind=engine)

app.include_router(profiles.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to TrueProfile AI backend"}
