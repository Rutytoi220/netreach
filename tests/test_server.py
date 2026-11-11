"""
Unit and Integration Tests for NetReach REST API Server Daemon (`netreach serve`).
"""

import pytest
from starlette.testclient import TestClient

from netreach.server import create_app, FASTAPI_AVAILABLE
from netreach.engine import NetReachEngine


@pytest.fixture(scope="module")
def client():
    """Create test client with lightweight engine instance."""
    assert FASTAPI_AVAILABLE, "FastAPI must be installed for REST API tests."
    engine = NetReachEngine()
    app = create_app(engine=engine)
    return TestClient(app)


def test_api_health_endpoint(client):
    """Verify GET /api/v1/health returns engine status and version."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.7.0"
    assert data["parameter_count"] == 38041604
    assert data["model"] == "ChallengerLatentNetV4"
    assert "device" in data


def test_api_audit_endpoint(client):
    """Verify POST /api/v1/audit executes reachability audit."""
    payload = {
        "graph": "examples/hybrid_cloud_iam.json",
        "source": "WORKSTATION-01.CORP.LOCAL",
        "target": "arn:aws:iam::aws:policy/AdministratorAccess",
        "depths": [4, 8],
    }
    response = client.post("/api/v1/audit", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert data["target"] == "arn:aws:iam::aws:policy/AdministratorAccess"
    assert data["risk_level"] in ("CRITICAL", "HIGH", "MODERATE")
    assert "8" in data["probabilities"]
    assert len(data["neural_path"]) > 0
    assert data["inference_latency_ms"] > 0


def test_api_harden_endpoint(client):
    """Verify POST /api/v1/harden performs neural edge ablation and chokepoint ranking."""
    payload = {
        "graph": "examples/hybrid_cloud_iam.json",
        "source": "WORKSTATION-01.CORP.LOCAL",
        "target": "arn:aws:iam::aws:policy/AdministratorAccess",
        "top_k": 2,
        "depth": 6,
    }
    response = client.post("/api/v1/harden", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert data["baseline_probability"] > 0.70
    assert len(data["chokepoints"]) == 2
    assert data["chokepoints"][0]["rank"] == 1
    assert data["chokepoints"][0]["delta_p"] > 0.50
    assert "mitigation_cmd" in data["chokepoints"][0]


def test_api_audit_invalid_node(client):
    """Verify 400 Bad Request error when querying invalid/non-existent graph nodes."""
    payload = {
        "graph": "examples/hybrid_cloud_iam.json",
        "source": "INVALID_PRINCIPAL_NAME",
        "target": "arn:aws:iam::aws:policy/AdministratorAccess",
    }
    response = client.post("/api/v1/audit", json=payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()
