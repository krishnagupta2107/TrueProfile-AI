"""
Generates and saves an API key to the database.

Usage:
    python scripts/generate_api_key.py --owner "your-name-or-email"

The generated key is printed to stdout — save it, it won't be shown again.
"""
import os
import sys
import secrets
import argparse

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database import SessionLocal, Base, engine
from backend.models.api_key import APIKey  # noqa: F401
import backend.models.profile              # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Generate a TrueProfile AI API key")
    parser.add_argument("--owner", required=True, help="Owner name or email for this key")
    args = parser.parse_args()

    # Ensure tables exist (safe in dev before alembic runs)
    Base.metadata.create_all(bind=engine)

    raw_key = f"tp-{secrets.token_urlsafe(32)}"

    with SessionLocal() as session:
        key_obj = APIKey(key=raw_key, owner=args.owner)
        session.add(key_obj)
        session.commit()

    print(f"\n✅ API key created for '{args.owner}':")
    print(f"\n   {raw_key}\n")
    print("Include this in your requests as the 'X-API-Key' header.")
    print("This key will not be shown again.\n")


if __name__ == "__main__":
    main()
