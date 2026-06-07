import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.getcwd())
from app import app
import json 

SINGLE_JSON = os.path.join("test_api","single_request.json")
BATCH_JSON = os.path.join("test_api","batch_requests.json")

@pytest.fixture(scope="module")
def client_with_model():
    """Initialize the app with the startup event triggered."""
    with TestClient(app) as client:
        return client


@pytest.fixture(scope="module")
def open_files():
    with open(SINGLE_JSON) as sin_file:
        single = json.load(sin_file)

    with open(BATCH_JSON) as bat_file:
        batch = json.load(bat_file)
    
    return single, batch


def test_health(client_with_model):
    response = client_with_model.get("/")

    assert response.status_code == 200


def test_single_prediction(client_with_model, open_files):
    single, _ = open_files

    response = client_with_model.post(
        "/predict/single",
        json=single
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction_probability" in data


def test_batch_prediction(client_with_model, open_files):
    _, batch = open_files

    response = client_with_model.post(
        "/predict/batch/json",
        json=batch
    )

    assert response.status_code == 200

    data = response.json()

    for e in data:
        assert "prediction_probability" in e