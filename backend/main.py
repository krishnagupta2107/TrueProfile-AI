from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import profiles

app = FastAPI(title="TrueProfile AI", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profiles.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to TrueProfile AI backend"}
