"""Smoke test HTTP: la ruta /ask responde 200 y devuelve JSON con campo `answer`."""
from fastapi.testclient import TestClient

from main import app


def test_ask_endpoint_ok() -> None:

    client = TestClient(app)
    response = client.get("/ask", params={"question": "Explica el higado"})
    assert response.status_code == 200
    assert "answer" in response.json()
