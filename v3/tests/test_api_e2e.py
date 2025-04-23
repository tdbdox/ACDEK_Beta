"""
End-to-end API tests for ACDEK API server.
"""
import pytest
import requests
import json
import os
import subprocess
import time
import signal
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL = "http://localhost:8000"

@pytest.fixture(scope="module")
def api_server():
    """Start the API server for testing and tear it down after tests."""
    # Get the path to the API server module
    api_path = Path(__file__).parent.parent / "web" / "api" / "main.py"
    
    # Start the server as a subprocess
    proc = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=api_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the server time to start
    time.sleep(2)
    
    yield
    
    # Terminate the server
    proc.send_signal(signal.SIGTERM)
    proc.wait()

def test_status_endpoint(api_server):
    """Test the /status endpoint."""
    response = requests.get(f"{API_URL}/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "monitor_available" in data

def test_run_pipeline(api_server):
    """Test the /run endpoint with a simple request."""
    data = {"data": {"text": "Test input for pipeline"}}
    response = requests.post(f"{API_URL}/run", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "result" in result

def test_validate_constraints(api_server):
    """Test the /validate endpoint."""
    data = {"data": {"text": "Test input for constraint validation"}}
    response = requests.post(f"{API_URL}/validate", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "passed" in result

def test_confidence_scoring(api_server):
    """Test the /confidence endpoint."""
    # Create valid input with a vector
    import numpy as np
    vector = list(np.random.rand(10))  # Convert to list for JSON serialization
    data = {"data": {"vector": vector, "model_name": "test-model"}}
    
    response = requests.post(f"{API_URL}/confidence", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "aggregate_score" in result
    assert "factors" in result