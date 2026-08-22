"""
Pytest configuration - use a temp file-based SQLite test database.
"""
import os
import pytest

TEST_DB_PATH = "test_trueprofile.db"
os.environ["DATABASE_URL"] = f"sqlite:///./{TEST_DB_PATH}"


def pytest_sessionfinish(session, exitstatus):
    """Clean up test database file after all tests finish."""
    try:
        from backend.database import engine
        engine.dispose()
    except Exception:
        pass
        
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except PermissionError:
            pass
