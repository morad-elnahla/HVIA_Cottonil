"""Tests are offline by default, including when a local .env has real credentials."""
import socket
import pytest
import requests
import logging


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network disabled in tests; supply a mocked transport")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def isolated_llm_log(monkeypatch):
    # Keep mocked transport events out of the production diagnostic history.
    from src import llm
    monkeypatch.setattr(llm, "_logger", lambda: logging.getLogger("cottonil.test.llm"))
