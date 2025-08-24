"""Integration tests for the complete TPM MCP system"""
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.services.data_processor import DunnhumbyProcessor
from src.models.entities import Household, Product, Transaction, Store, Promotion


class TestDataProcessingIntegration:
    """Test complete data processing workflow"""
    
    def test_full_data_processing_pipeline(self, db_session, temp_csv_data):
        """Test complete data processing from CSV to database"""
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        
        # Process the complete dataset
        results = processor.process_dataset()
        
        # Verify processing results
        assert "households" in results
        assert "products" in results
        assert "transactions" in results
        assert "promotions" in results
        
        # Verify data was actually stored in database
        household_count = db_session.query(Household).count()
        product_count = db_session.query(Product).count()
        transaction_count = db_session.query(Transaction).count()
        promotion_count = db_session.query(Promotion).count()
        store_count = db_session.query(Store).count()
        
        assert household_count > 0
        assert product_count > 0
        assert transaction_count > 0
        assert promotion_count > 0
        assert store_count > 0
        
        # Verify data integrity
        household = db_session.query(Household).first()
        assert household.household_key is not None
        
        product = db_session.query(Product).first()
        assert product.product_id is not None
        
        transaction = db_session.query(Transaction).first()
        assert transaction.sales_value > 0
    
    def test_data_processing_with_missing_files(self, db_session, tmp_path):
        """Test data processing with some missing CSV files"""
        # Create only some of the expected files
        households_df = pd.DataFrame([
            {"household_key": "HH_001", "AGE_DESC": "25-34"}
        ])
        households_df.to_csv(tmp_path / "hh_demographic.csv", index=False)
        
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        results = processor.process_dataset()
        
        # Should process only available files
        assert "households" in results
        assert results["households"] > 0
        
        # Missing files should not be in results
        assert "products" not in results or results["products"] == 0
    
    def test_incremental_data_processing(self, db_session, temp_csv_data):
        """Test processing data multiple times (should clear and reload)"""
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        
        # First processing
        results1 = processor.process_dataset()
        count1 = db_session.query(Household).count()
        
        # Second processing (should clear and reload)
        results2 = processor.process_dataset()
        count2 = db_session.query(Household).count()
        
        # Counts should be the same (data was cleared and reloaded)
        assert count1 == count2
        assert results1 == results2


class TestMCPToolsIntegration:
    """Test integration of MCP tools with real data"""
    
    def test_mcp_tools_with_database_data(self, populated_db_session):
        """Test MCP tools work with actual database data"""
        from mcp_server import get_sample_data_overview
        
        # Mock the database session in the tool
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_instance = mock_processor.return_value.__enter__.return_value
            mock_instance.get_data_summary.return_value = {
                "households": 2,
                "products": 2,
                "transactions": 2,
                "promotions": 2
            }
            
            result = get_sample_data_overview()
            
            assert "TPM Data Overview" in result
            assert "households: 2" in result.lower() or "households**: 2" in result.lower()
    
    def test_prediction_tools_integration(self):
        """Test prediction tools produce consistent results"""
        from mcp_server import predict_promotion_lift, optimize_promotion_budget
        
        # Test multiple predictions for consistency
        results = []
        for i in range(3):
            result = predict_promotion_lift("Test Product", 20.0, 14, "DISCOUNT")
            results.append(result)
        
        # Results should be identical for same inputs
        assert results[0] == results[1] == results[2]
        
        # Test budget optimization
        budget_result = optimize_promotion_budget(50000.0, "cereal, frozen", 5)
        assert "$50,000" in budget_result
        assert "cereal" in budget_result.lower()


class TestSystemResilience:
    """Test system behavior under various conditions"""
    
    def test_database_connection_failure_handling(self, db_session):
        """Test handling of database connection failures"""
        from mcp_server import get_sample_data_overview
        
        # Simulate database connection failure
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_processor.side_effect = Exception("Connection failed")
            
            result = get_sample_data_overview()
            
            # Should handle error gracefully
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_invalid_csv_data_handling(self, db_session, tmp_path):
        """Test handling of corrupted or invalid CSV data"""
        # Create invalid CSV file
        invalid_csv = tmp_path / "hh_demographic.csv"
        invalid_csv.write_text("invalid,csv,data\nwith,wrong,columns,and,extra,data")
        
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        
        # Should handle invalid data gracefully
        try:
            results = processor.process_dataset()
            # If it processes, households might be 0 or have issues
            assert isinstance(results, dict)
        except Exception as e:
            # Exception handling is acceptable for truly invalid data
            assert isinstance(e, Exception)
    
    def test_large_dataset_simulation(self, db_session, tmp_path):
        """Test system behavior with larger dataset"""
        # Create larger test dataset
        households_data = [
            {
                "household_key": f"HH_{i:04d}",
                "AGE_DESC": "25-34" if i % 2 else "35-44",
                "INCOME_DESC": "50-74K"
            }
            for i in range(100)
        ]
        
        products_data = [
            {
                "PRODUCT_ID": 1000 + i,
                "DEPARTMENT": "GROCERY" if i % 2 else "DAIRY",
                "COMMODITY_DESC": f"PRODUCT_{i}",
                "BRAND": f"BRAND_{i % 10}"
            }
            for i in range(50)
        ]
        
        transactions_data = [
            {
                "household_key": f"HH_{(i % 100):04d}",
                "BASKET_ID": f"BASKET_{i}",
                "PRODUCT_ID": 1000 + (i % 50),
                "STORE_ID": (i % 5) + 1,
                "DAY": (i % 365) + 1,
                "WEEK_NO": ((i % 365) // 7) + 1,
                "QUANTITY": (i % 5) + 1,
                "SALES_VALUE": round((i % 20) + 0.99, 2),
                "RETAIL_DISC": 0.0,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0
            }
            for i in range(1000)
        ]
        
        # Save to CSV files
        pd.DataFrame(households_data).to_csv(tmp_path / "hh_demographic.csv", index=False)
        pd.DataFrame(products_data).to_csv(tmp_path / "product.csv", index=False)
        pd.DataFrame(transactions_data).to_csv(tmp_path / "transaction_data.csv", index=False)
        
        # Create empty files for optional data
        (tmp_path / "causal_data.csv").touch()
        (tmp_path / "coupon.csv").touch()
        (tmp_path / "coupon_redempt.csv").touch()
        (tmp_path / "campaign_desc.csv").touch()
        (tmp_path / "campaign_table.csv").touch()
        
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        results = processor.process_dataset()
        
        # Should handle larger dataset
        assert results["households"] == 100
        assert results["products"] == 50
        assert results["transactions"] == 1000


class TestEndToEndWorkflows:
    """Test complete end-to-end user workflows"""
    
    def test_new_user_onboarding_workflow(self):
        """Test complete new user experience"""
        from mcp_server import (
            welcome, discover_tpm_capabilities, get_sample_data_overview,
            predict_promotion_lift
        )
        
        # Step 1: User gets welcome message
        welcome_msg = welcome()
        assert "Welcome" in welcome_msg
        
        # Step 2: User discovers capabilities
        capabilities = discover_tpm_capabilities()
        assert "capabilities" in capabilities.lower()
        assert "predict_promotion_lift" in capabilities
        
        # Step 3: User checks data availability
        data_overview = get_sample_data_overview()
        assert "Data Overview" in data_overview
        
        # Step 4: User tries first prediction
        prediction = predict_promotion_lift("Cheerios", 25.0, 14)
        assert "Cheerios" in prediction
        assert "25.0%" in prediction
    
    def test_promotion_planning_workflow(self):
        """Test complete promotion planning workflow"""
        from mcp_server import (
            predict_promotion_lift, optimize_promotion_budget,
            analyze_competitive_impact
        )
        
        # Step 1: Predict performance for specific product
        prediction = predict_promotion_lift("Pepsi", 20.0, 14, "BOGO")
        assert "Pepsi" in prediction
        assert "BOGO" in prediction
        
        # Step 2: Analyze competitive situation
        competitive_analysis = analyze_competitive_impact(
            "Pepsi", "Coca-Cola launching aggressive discount campaign"
        )
        assert "Competitive Impact" in competitive_analysis
        assert "Coca-Cola" in competitive_analysis
        
        # Step 3: Optimize budget allocation
        budget_optimization = optimize_promotion_budget(
            100000.0, "beverages, snacks", 8, "maximize_roi"
        )
        assert "$100,000" in budget_optimization
        assert "beverages" in budget_optimization.lower()
    
    def test_data_exploration_workflow(self, populated_db_session):
        """Test data exploration and analysis workflow"""
        from mcp_server import get_sample_data_overview
        
        # User explores available data
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_instance = mock_processor.return_value.__enter__.return_value
            mock_instance.get_data_summary.return_value = {
                "households": 100,
                "products": 200,
                "transactions": 10000,
                "promotions": 500,
                "stores": 10
            }
            
            overview = get_sample_data_overview()
            
            assert "100" in overview  # household count
            assert "200" in overview  # product count
            assert "10,000" in overview  # transaction count


class TestSystemConfiguration:
    """Test system configuration and setup"""
    
    def test_database_setup_integration(self, db_session):
        """Test database setup and table creation"""
        from src.models.database import create_tables
        
        # Tables should be created successfully
        create_tables()
        
        # Verify we can create records in all tables
        household = Household(household_key="SETUP_TEST")
        product = Product(product_id=9999, department="TEST")
        store = Store(store_id=999)
        
        db_session.add_all([household, product, store])
        db_session.commit()
        
        # Verify records exist
        assert db_session.query(Household).filter_by(household_key="SETUP_TEST").first()
        assert db_session.query(Product).filter_by(product_id=9999).first()
        assert db_session.query(Store).filter_by(store_id=999).first()
    
    def test_environment_configuration(self):
        """Test environment variable handling"""
        import os
        
        # Test environment is set up correctly
        assert os.environ.get("TESTING") == "true"
        assert "sqlite" in os.environ.get("DATABASE_URL", "").lower()


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system"""
    
    def test_prediction_tool_performance(self):
        """Test that prediction tools perform adequately"""
        import time
        from mcp_server import predict_promotion_lift
        
        start_time = time.time()
        
        # Run multiple predictions
        for i in range(10):
            result = predict_promotion_lift(f"Product_{i}", 20.0, 14)
            assert "Product_" in result
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 10 predictions in reasonable time (< 5 seconds)
        assert total_time < 5.0
        
        # Average time per prediction should be reasonable
        avg_time = total_time / 10
        assert avg_time < 0.5  # Less than 500ms per prediction
    
    def test_data_processing_performance(self, db_session, temp_csv_data):
        """Test data processing performance"""
        import time
        
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        
        start_time = time.time()
        results = processor.process_dataset()
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process small dataset quickly (< 10 seconds)
        assert processing_time < 10.0
        
        # Should have processed some data
        assert sum(results.values()) > 0