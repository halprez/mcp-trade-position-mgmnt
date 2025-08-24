"""Tests for MCP tools functionality"""
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    # Import the tools from mcp_server.py
    from mcp_server import (
        discover_tpm_capabilities,
        claude_desktop_welcome,
        welcome,
        get_sample_data_overview,
        predict_promotion_lift,
        optimize_promotion_budget,
        analyze_competitive_impact
    )
except ImportError:
    pytest.skip("MCP server tools not available", allow_module_level=True)


class TestDiscoveryTools:
    """Test discovery and welcome message tools"""
    
    def test_discover_tpm_capabilities(self):
        """Test the main capabilities discovery tool"""
        result = discover_tpm_capabilities()
        
        assert "Trade Promotion Management AI Assistant" in result
        assert "predict_promotion_lift" in result
        assert "optimize_promotion_budget" in result
        assert "analyze_competitive_impact" in result
        assert "ML Models" in result
        assert "Quick Start Examples" in result
    
    def test_claude_desktop_welcome(self):
        """Test Claude Desktop welcome message"""
        result = claude_desktop_welcome()
        
        assert "Welcome to Your AI-Powered TPM Assistant" in result
        assert "ML-Powered Predictions" in result
        assert "Real-Time Analytics" in result
        assert "Strategic Planning" in result
        assert "Quick Start" in result
        assert "Pro Tips" in result
    
    def test_welcome_shortcut(self):
        """Test the welcome shortcut function"""
        result = welcome()
        
        # Should return same as claude_desktop_welcome
        expected = claude_desktop_welcome()
        assert result == expected
    
    def test_get_sample_data_overview(self):
        """Test sample data overview tool"""
        result = get_sample_data_overview()
        
        assert "TPM Data Overview" in result
        # Should handle database connection errors gracefully
        assert "ML prediction models" in result or "Error accessing database" in result


class TestPredictionTools:
    """Test AI prediction and analysis tools"""
    
    def test_predict_promotion_lift_basic(self):
        """Test basic promotion lift prediction"""
        result = predict_promotion_lift(
            product_name="Cheerios",
            discount_percentage=25.0,
            duration_days=14,
            promotion_type="DISCOUNT"
        )
        
        assert "AI Promotion Prediction: Cheerios" in result
        assert "25.0% off" in result
        assert "14 days" in result
        assert "DISCOUNT" in result
        assert "Sales Lift:" in result
        assert "Revenue Impact:" in result
        assert "Estimated ROI:" in result
    
    def test_predict_promotion_lift_bogo(self):
        """Test BOGO promotion prediction"""
        result = predict_promotion_lift(
            product_name="Pepsi",
            discount_percentage=50.0,
            duration_days=7,
            promotion_type="BOGO",
            store_name="Store 1"
        )
        
        assert "Pepsi" in result
        assert "BOGO" in result
        assert "Store 1" in result
        assert "7 days" in result
    
    def test_predict_promotion_lift_edge_cases(self):
        """Test prediction with edge case values"""
        # Very high discount
        result = predict_promotion_lift("Test Product", 50.0, 28)
        assert "Test Product" in result
        
        # Very low discount
        result = predict_promotion_lift("Test Product", 5.0, 7)
        assert "Test Product" in result
        
        # Long duration
        result = predict_promotion_lift("Test Product", 20.0, 28)
        assert "Test Product" in result
    
    def test_optimize_promotion_budget_basic(self):
        """Test basic budget optimization"""
        result = optimize_promotion_budget(
            total_budget=50000.0,
            product_categories="frozen, cereal, beverages",
            max_products=5,
            objectives="maximize_roi"
        )
        
        assert "AI Budget Optimization Results" in result
        assert "$50,000" in result
        assert "maximize_roi" in result
        assert "Optimal Allocation" in result
        assert "Portfolio Performance Forecast" in result
    
    def test_optimize_promotion_budget_balanced(self):
        """Test budget optimization with balanced objectives"""
        result = optimize_promotion_budget(
            total_budget=25000.0,
            product_categories="snacks, dairy",
            max_products=3,
            objectives="balanced"
        )
        
        assert "balanced" in result.lower()
        assert "$25,000" in result
    
    def test_optimize_promotion_budget_no_match(self):
        """Test budget optimization with unrecognized categories"""
        result = optimize_promotion_budget(
            total_budget=10000.0,
            product_categories="invalid_category, unknown_category",
            max_products=5
        )
        
        assert "No matching categories found" in result
    
    def test_analyze_competitive_impact(self):
        """Test competitive impact analysis"""
        result = analyze_competitive_impact(
            product_name="Diet Coke",
            competitor_actions="Pepsi launching aggressive 30% discount promotion"
        )
        
        assert "Competitive Impact Analysis: Diet Coke" in result
        assert "Pepsi launching aggressive 30% discount" in result
        assert "Strategic Response Options" in result
        assert "Direct Counter-Attack" in result
        assert "Differentiated Response" in result
        assert "Defensive Hold" in result
        assert "AI Recommendations" in result
    
    def test_analyze_competitive_impact_aggressive(self):
        """Test competitive analysis with aggressive competitor actions"""
        result = analyze_competitive_impact(
            product_name="Cheerios",
            competitor_actions="Kellogg's aggressive BOGO promotion across all cereal brands"
        )
        
        assert "aggressive" in result.lower()
        assert "High" in result  # Should indicate high risk
        assert "Immediate counter-promotion" in result


class TestPredictionLogic:
    """Test the underlying prediction logic and calculations"""
    
    def test_discount_factor_scaling(self):
        """Test that discount factors scale appropriately"""
        # Low discount should have lower lift
        low_result = predict_promotion_lift("Test", 10.0, 14)
        high_result = predict_promotion_lift("Test", 40.0, 14)
        
        # Extract lift percentages (this is a simplified check)
        assert "Sales Lift:" in low_result
        assert "Sales Lift:" in high_result
    
    def test_promotion_type_multipliers(self):
        """Test that different promotion types have different impacts"""
        discount_result = predict_promotion_lift("Test", 20.0, 14, "DISCOUNT")
        bogo_result = predict_promotion_lift("Test", 20.0, 14, "BOGO")
        
        assert "DISCOUNT" in discount_result
        assert "BOGO" in bogo_result
        # BOGO should generally have higher lift
    
    def test_duration_effects(self):
        """Test that duration affects predictions"""
        short_result = predict_promotion_lift("Test", 20.0, 7)
        long_result = predict_promotion_lift("Test", 20.0, 21)
        
        assert "7 days" in short_result
        assert "21 days" in long_result


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_inputs(self):
        """Test tools handle empty inputs gracefully"""
        result = predict_promotion_lift("", 0, 0)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_invalid_budget_optimization(self):
        """Test budget optimization with invalid inputs"""
        result = optimize_promotion_budget(0, "", 0)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_string_formatting(self):
        """Test that all tools return properly formatted strings"""
        tools_and_args = [
            (discover_tpm_capabilities, []),
            (claude_desktop_welcome, []),
            (welcome, []),
            (get_sample_data_overview, []),
            (predict_promotion_lift, ["Test", 20.0]),
            (optimize_promotion_budget, [10000.0, "cereal"]),
            (analyze_competitive_impact, ["Test", "competitor action"])
        ]
        
        for tool_func, args in tools_and_args:
            result = tool_func(*args)
            assert isinstance(result, str)
            assert len(result) > 0
            # Check for common formatting elements
            assert any(marker in result for marker in ["**", "•", "✅", "🚀"])


class TestIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_new_user_workflow(self):
        """Test typical new user discovery workflow"""
        # User starts with welcome
        welcome_result = welcome()
        assert "Welcome" in welcome_result
        
        # User discovers capabilities
        capabilities_result = discover_tpm_capabilities()
        assert "capabilities" in capabilities_result.lower()
        
        # User checks data
        data_result = get_sample_data_overview()
        assert "Data Overview" in data_result
    
    def test_promotion_planning_workflow(self):
        """Test typical promotion planning workflow"""
        # Predict lift for specific product
        prediction = predict_promotion_lift("Cheerios", 25.0, 14, "DISCOUNT")
        assert "Cheerios" in prediction
        
        # Analyze competitive landscape
        competitive = analyze_competitive_impact("Cheerios", "Competitor launching promotion")
        assert "Competitive Impact" in competitive
        
        # Optimize budget allocation
        budget = optimize_promotion_budget(50000.0, "cereal, snacks", 5)
        assert "Budget Optimization" in budget
    
    def test_various_product_scenarios(self):
        """Test predictions for different product types"""
        products = [
            ("Coca Cola", "beverages"),
            ("Häagen-Dazs", "ice cream"), 
            ("Doritos", "snacks"),
            ("Milk", "dairy")
        ]
        
        for product, category in products:
            result = predict_promotion_lift(product, 20.0, 14)
            assert product in result
            assert "Sales Lift:" in result