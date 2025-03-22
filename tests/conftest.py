import pytest
import os

@pytest.fixture(autouse=True)
def env_setup():
    """Set up test environment variables"""
    os.environ["GOOGLE_BOOKS_API_KEY"] = "test_key"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "test_credentials"
    yield
    # Clean up
    os.environ.pop("GOOGLE_BOOKS_API_KEY", None)
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)