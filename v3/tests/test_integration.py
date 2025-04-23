"""
Integration tests for the ACDEK v3 system.

This module tests the integration between components and the system's
end-to-end functionality.
"""
import os
import sys
import pytest
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk.acdek_client import Client
from enrichment.standard_pipelines import create_standard_pipeline, create_minimal_pipeline
from registry.enhanced_plugin_registry import enhanced_registry, FallbackStrategy

class TestSDKIntegration:
    """Test the SDK client integration with all components."""
    
    def test_client_initialization(self):
        """Test that the Client can be initialized."""
        client = Client()
        assert client is not None
        assert client.enrichment_pipeline is not None
        assert client.constraint_verifier is not None
        assert client.confidence_scorer is not None
    
    def test_client_with_minimal_pipeline(self):
        """Test that the Client works with minimal pipeline."""
        client = Client(config={"pipeline_type": "minimal"})
        assert client is not None
        assert client.enrichment_pipeline is not None
    
    def test_client_fallback_modes(self):
        """Test that the Client handles different fallback modes."""
        # Test with valid mode
        client = Client(config={"fallback_mode": "relaxed"})
        assert client is not None
        
        # Test with invalid mode (should fallback to relaxed)
        client = Client(config={"fallback_mode": "invalid_mode"})
        assert client is not None
    
    def test_run_enrichment(self):
        """Test the run_enrichment method."""
        client = Client()
        input_data = {"text": "Test input for enrichment"}
        result = client.run_enrichment(input_data)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "data" in result
    
    def test_validate_constraints(self):
        """Test the validate_constraints method."""
        client = Client()
        input_data = {"text": "Test input for constraints"}
        result = client.validate_constraints(input_data)
        
        assert isinstance(result, dict)
        assert "passed" in result
    
    def test_score_confidence(self):
        """Test the score_confidence method."""
        client = Client()
        import numpy as np
        
        # Create a proper input with vector
        vector = np.random.rand(10)
        input_data = {
            "vector": vector,
            "model_name": "test-model"
        }
        
        result = client.score_confidence(input_data)
        
        assert isinstance(result, dict)
        assert "aggregate_score" in result
        assert "factors" in result
    
    def test_full_pipeline(self):
        """Test the full_pipeline method."""
        client = Client()
        input_data = {"text": "Test input for full pipeline"}
        result = client.full_pipeline(input_data)
        
        assert isinstance(result, dict)
        assert "enrichment" in result
        assert "constraints" in result
        assert "confidence" in result
        assert "timing" in result
    
    def test_client_error_handling(self):
        """Test that the Client handles errors gracefully."""
        # Create a client with dummy pipeline to test error handling
        client = Client(config={"pipeline_type": "dummy"})
        
        # Missing required fields should not crash
        input_data = {}
        result = client.full_pipeline(input_data)
        
        assert isinstance(result, dict)
        assert "enrichment" in result
        assert "constraints" in result
        assert "confidence" in result
    
    def test_json_output_format(self):
        """Test that the Client can output JSON."""
        client = Client()
        input_data = {"text": "Test input for JSON output"}
        result = client.full_pipeline(input_data, output_format="json")
        
        assert isinstance(result, str)
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

class TestComponentIntegration:
    """Test the integration between different components."""
    
    def test_enhanced_registry_fallbacks(self):
        """Test that the enhanced registry handles fallbacks correctly."""
        # Test loading a non-existent plugin with fallback strategy
        result = enhanced_registry.load_plugin(
            plugin_type="ENRICHMENT",  # Should convert to enum automatically
            name="non_existent_plugin",
            fallback_strategy=FallbackStrategy(
                use_default=True,
                allow_missing=True
            )
        )
        
        # Should not raise an exception
        assert result is not None
        assert isinstance(result.success, bool)
    
    def test_standard_pipeline_creation(self):
        """Test that the standard pipeline can be created."""
        pipeline = create_standard_pipeline()
        assert pipeline is not None
        
        # Should have at least confidence scoring
        has_confidence = False
        for stage in pipeline.stages:
            if "confidence" in stage.name:
                has_confidence = True
                break
        
        assert has_confidence, "Standard pipeline should include confidence scoring"
    
    def test_minimal_pipeline_creation(self):
        """Test that the minimal pipeline can be created."""
        pipeline = create_minimal_pipeline()
        assert pipeline is not None
        assert len(pipeline.stages) > 0
    
    def test_pipeline_execution(self):
        """Test that the pipelines can execute."""
        pipeline = create_minimal_pipeline()
        result = pipeline.execute({"text": "Test input"})
        
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "data")


# The following tests require the web API server to be running
# They are marked with skip_if_no_server to avoid failing when the server is not available
def is_server_running():
    """Check if the API server is running."""
    import requests
    try:
        response = requests.get("http://localhost:8000/status", timeout=1)
        return response.status_code == 200
    except:
        return False

@pytest.mark.skipif(not is_server_running(), reason="API server not running")
class TestWebAPIIntegration:
    """Test the Web API integration."""
    
    def test_status_endpoint(self):
        """Test the status endpoint."""
        import requests
        response = requests.get("http://localhost:8000/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
    
    def test_run_endpoint(self):
        """Test the run endpoint."""
        import requests
        data = {"data": {"text": "Test input for API"}}
        response = requests.post("http://localhost:8000/run", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "result" in result
    
    def test_validate_endpoint(self):
        """Test the validate endpoint."""
        import requests
        data = {"data": {"text": "Test input for validation"}}
        response = requests.post("http://localhost:8000/validate", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "passed" in result
    
    def test_confidence_endpoint(self):
        """Test the confidence endpoint."""
        import requests
        import numpy as np
        vector = list(np.random.rand(10))
        data = {"data": {"vector": vector, "model_name": "test-model"}}
        response = requests.post("http://localhost:8000/confidence", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "aggregate_score" in result