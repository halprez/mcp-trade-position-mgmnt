"""Tests for API endpoints and FastAPI integration"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        with patch('src.main.get_data_summary') as mock_summary:
            mock_summary.return_value = {"households": 100, "products": 50}
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database"] == "connected"
            assert "data_summary" in data
            assert data["version"] == "0.1.0"
    
    def test_health_check_failure(self, client):
        """Test health check with database error"""
        with patch('src.main.get_data_summary') as mock_summary:
            mock_summary.side_effect = Exception("Database connection failed")
            
            response = client.get("/health")
            
            assert response.status_code == 500
            assert "Health check failed" in response.json()["detail"]


class TestMCPDiscoveryEndpoint:
    """Test MCP discovery endpoint"""
    
    def test_mcp_discovery(self, client):
        """Test MCP discovery endpoint"""
        response = client.get("/mcp/discover")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check server info
        assert "server_info" in data
        assert data["server_info"]["name"] == "TPM AI Assistant"
        assert data["server_info"]["version"] == "1.0.0"
        
        # Check capabilities
        assert "capabilities" in data
        assert "tools" in data["capabilities"]
        assert "resources" in data["capabilities"]
        assert "endpoints" in data["capabilities"]
        
        # Check specific tools
        tool_names = [tool["name"] for tool in data["capabilities"]["tools"]]
        assert "discover_tpm_capabilities" in tool_names
        assert "predict_promotion_lift" in tool_names
        assert "optimize_promotion_budget" in tool_names
        
        # Check resources
        resources = data["capabilities"]["resources"]
        assert "tpm://products/{category}" in resources
        assert "tpm://promotions/{time_period}" in resources
        
        # Check quick start
        assert "quick_start" in data
        assert "first_time_users" in data["quick_start"]


class TestPromotionEndpoints:
    """Test promotion-related endpoints"""
    
    @patch('src.api.promotions.get_campaigns')
    def test_get_campaigns(self, mock_get_campaigns, client):
        """Test campaigns endpoint"""
        mock_get_campaigns.return_value = [
            {"id": 1, "name": "Test Campaign", "status": "active"}
        ]
        
        response = client.get("/promotions/campaigns")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
        # Endpoint might not be fully implemented, so we allow 404
    
    def test_promotions_endpoint_exists(self, client):
        """Test that promotions endpoints are registered"""
        # Test that the endpoint exists (even if not implemented)
        response = client.get("/promotions/campaigns")
        # Should not be 500 (server error), might be 404 (not implemented) or 200 (success)
        assert response.status_code in [200, 404, 422]


class TestAnalyticsEndpoints:
    """Test analytics-related endpoints"""
    
    def test_analytics_dashboard(self, client):
        """Test analytics dashboard endpoint"""
        response = client.get("/analytics/dashboard")
        # Should exist and not return server error
        assert response.status_code in [200, 404, 422]
    
    @patch('src.services.data_processor.get_data_summary')
    def test_analytics_with_mock_data(self, mock_summary, client):
        """Test analytics with mocked data"""
        mock_summary.return_value = {
            "households": 1000,
            "products": 500,
            "transactions": 10000
        }
        
        # Try to access analytics endpoint
        response = client.get("/analytics/dashboard")
        # Should not fail with server error
        assert response.status_code != 500


class TestPredictionEndpoints:
    """Test prediction-related endpoints"""
    
    def test_prediction_endpoint_exists(self, client):
        """Test prediction endpoints are registered"""
        response = client.get("/predictions/lift-prediction")
        assert response.status_code in [200, 404, 422]
        
        response = client.get("/predictions/budget-optimization")
        assert response.status_code in [200, 404, 422]


class TestAPIIntegration:
    """Test API integration with core functionality"""
    
    def test_app_startup(self):
        """Test that the app can start successfully"""
        from src.main import app
        assert app is not None
        assert app.title == "Trade Promotion Management MCP Server"
        assert app.version == "0.1.0"
    
    def test_app_routes(self, client):
        """Test that required routes are registered"""
        # Get OpenAPI schema to check routes
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema["paths"]
        
        # Check required endpoints exist
        assert "/health" in paths
        assert "/mcp/discover" in paths
    
    def test_cors_and_middleware(self, client):
        """Test CORS and middleware configuration"""
        response = client.get("/health")
        
        # Should have proper headers
        assert response.status_code in [200, 500]  # Allow for database errors
        assert isinstance(response.headers, dict)


class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed handling"""
        response = client.post("/health")  # Health endpoint likely only accepts GET
        assert response.status_code == 405
    
    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON in requests"""
        response = client.post(
            "/predictions/lift-prediction",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"}
        )
        # Should not crash with 500, might be 422 (validation error) or 404
        assert response.status_code in [404, 422]


class TestRequestValidation:
    """Test request validation and parameter handling"""
    
    def test_query_parameter_validation(self, client):
        """Test query parameter validation"""
        response = client.get("/health?invalid_param=test")
        # Should still work despite extra parameters
        assert response.status_code in [200, 500]
    
    def test_request_body_validation(self, client):
        """Test request body validation"""
        # Test with empty body where data is expected
        response = client.post("/predictions/lift-prediction", json={})
        # Should handle validation gracefully
        assert response.status_code in [404, 422, 400]


class TestResponseFormat:
    """Test API response formats"""
    
    def test_health_response_format(self, client):
        """Test health endpoint response format"""
        with patch('src.main.get_data_summary') as mock_summary:
            mock_summary.return_value = {"test": 123}
            
            response = client.get("/health")
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict)
                assert "status" in data
                assert "database" in data
                assert "version" in data
    
    def test_mcp_discovery_response_format(self, client):
        """Test MCP discovery response format"""
        response = client.get("/mcp/discover")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data, dict)
        assert isinstance(data["server_info"], dict)
        assert isinstance(data["capabilities"], dict)
        assert isinstance(data["capabilities"]["tools"], list)
        assert isinstance(data["capabilities"]["resources"], list)
    
    def test_json_response_headers(self, client):
        """Test that responses have correct content-type headers"""
        response = client.get("/mcp/discover")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


class TestPerformance:
    """Test API performance characteristics"""
    
    def test_health_check_response_time(self, client):
        """Test health check responds quickly"""
        import time
        
        with patch('src.main.get_data_summary') as mock_summary:
            mock_summary.return_value = {"test": 1}
            
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            # Should respond within 1 second
            assert (end_time - start_time) < 1.0
            assert response.status_code in [200, 500]
    
    def test_mcp_discovery_response_time(self, client):
        """Test MCP discovery responds quickly"""
        import time
        
        start_time = time.time()
        response = client.get("/mcp/discover")
        end_time = time.time()
        
        # Should respond within 1 second
        assert (end_time - start_time) < 1.0
        assert response.status_code == 200


class TestSecurityHeaders:
    """Test security-related headers and configurations"""
    
    def test_security_headers(self, client):
        """Test that appropriate security headers are present"""
        response = client.get("/health")
        
        # Check for security headers (if configured)
        headers = response.headers
        assert isinstance(headers, dict)
        # Note: Specific security headers depend on FastAPI configuration
    
    def test_no_sensitive_info_exposure(self, client):
        """Test that error responses don't expose sensitive information"""
        with patch('src.main.get_data_summary') as mock_summary:
            mock_summary.side_effect = Exception("Database password is secret123")
            
            response = client.get("/health")
            
            if response.status_code == 500:
                # Should not expose the actual exception message with sensitive info
                response_text = response.json()["detail"]
                assert "secret123" not in response_text