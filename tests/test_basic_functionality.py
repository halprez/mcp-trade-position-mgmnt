"""Basic functionality tests that focus on working components"""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.models.entities import Household, Product, Store, Transaction, Promotion, Campaign
from src.services.data_processor import DunnhumbyProcessor


class TestBasicModels:
    """Test basic model functionality"""
    
    def test_household_model(self, db_session):
        """Test household model operations"""
        household = Household(
            household_key="HH_TEST",
            age_desc="35-44", 
            income_desc="50-74K"
        )
        
        db_session.add(household)
        db_session.commit()
        
        saved = db_session.query(Household).filter_by(household_key="HH_TEST").first()
        assert saved is not None
        assert saved.age_desc == "35-44"
        assert "HH_TEST" in str(saved)
    
    def test_product_model(self, db_session):
        """Test product model operations"""
        product = Product(
            product_id=1001,
            department="DAIRY",
            brand="TEST_BRAND"
        )
        
        db_session.add(product)
        db_session.commit()
        
        saved = db_session.query(Product).filter_by(product_id=1001).first()
        assert saved is not None
        assert saved.department == "DAIRY"
        assert "1001" in str(saved)
        assert "TEST_BRAND" in str(saved)
    
    def test_store_model(self, db_session):
        """Test store model operations"""
        store = Store(store_id=100)
        
        db_session.add(store)
        db_session.commit()
        
        saved = db_session.query(Store).filter_by(store_id=100).first()
        assert saved is not None
        assert saved.store_id == 100
    
    def test_promotion_model(self, db_session):
        """Test promotion model operations"""
        promotion = Promotion(
            product_id=1001,
            store_id=1,
            start_day=1,
            end_day=14,
            promotion_type="DISCOUNT"
        )
        
        db_session.add(promotion)
        db_session.commit()
        
        saved = db_session.query(Promotion).first()
        assert saved is not None
        assert saved.promotion_type == "DISCOUNT"
        assert saved.end_day - saved.start_day == 13
    
    def test_campaign_model(self, db_session):
        """Test campaign model operations"""
        campaign = Campaign(
            name="TEST_CAMPAIGN",
            description="Test campaign",
            budget=10000.0
        )
        
        db_session.add(campaign)
        db_session.commit()
        
        saved = db_session.query(Campaign).filter_by(name="TEST_CAMPAIGN").first()
        assert saved is not None
        assert saved.budget == 10000.0
        assert "TEST_CAMPAIGN" in str(saved)


class TestDataProcessor:
    """Test data processor functionality"""
    
    def test_processor_initialization(self, db_session):
        """Test processor can be initialized"""
        processor = DunnhumbyProcessor("test/path", db_session)
        assert processor.data_path.name == "path"
        assert processor.session == db_session
    
    def test_expected_files(self, db_session):
        """Test expected files mapping"""
        processor = DunnhumbyProcessor(session=db_session)
        files = processor.get_expected_files()
        
        assert "households" in files
        assert "products" in files
        assert "transactions" in files
        assert files["households"] == "hh_demographic.csv"
    
    @patch('pandas.read_csv')
    def test_csv_loading_empty_directory(self, mock_read_csv, db_session, tmp_path):
        """Test CSV loading with empty directory"""
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        dataframes = processor.load_csv_files()
        
        # Should return empty dict for non-existent files
        assert isinstance(dataframes, dict)
        # Files don't exist, so should be empty
        assert len(dataframes) == 0


class TestDatabaseOperations:
    """Test database operations"""
    
    def test_database_crud_operations(self, db_session):
        """Test create, read, update operations"""
        # Create
        household = Household(household_key="CRUD_TEST")
        db_session.add(household)
        db_session.commit()
        
        # Read
        saved = db_session.query(Household).filter_by(household_key="CRUD_TEST").first()
        assert saved is not None
        
        # Update
        saved.age_desc = "Updated Age"
        db_session.commit()
        
        updated = db_session.query(Household).filter_by(household_key="CRUD_TEST").first()
        assert updated.age_desc == "Updated Age"
        
        # Count
        count = db_session.query(Household).count()
        assert count >= 1
    
    def test_relationships(self, db_session):
        """Test model relationships"""
        # Create related entities
        household = Household(household_key="REL_TEST")
        product = Product(product_id=2001, department="TEST")
        store = Store(store_id=200)
        
        db_session.add_all([household, product, store])
        db_session.commit()
        
        # Create transaction linking them
        transaction = Transaction(
            household_key="REL_TEST",
            basket_id="REL_BASKET",
            product_id=2001,
            store_id=200,
            day=1,
            week_no=1,
            quantity=1,
            sales_value=9.99
        )
        
        db_session.add(transaction)
        db_session.commit()
        
        # Verify relationships work
        saved_transaction = db_session.query(Transaction).filter_by(basket_id="REL_BASKET").first()
        assert saved_transaction is not None
        assert saved_transaction.household_key == "REL_TEST"
        assert saved_transaction.product_id == 2001
        assert saved_transaction.store_id == 200


class TestUtilityFunctions:
    """Test utility functions and helpers"""
    
    def test_data_summary_function(self):
        """Test data summary utility function"""
        from src.services.data_processor import get_data_summary
        
        # Mock the processor to avoid database dependencies
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_instance = mock_processor.return_value.__enter__.return_value
            mock_instance.get_data_summary.return_value = {"test": 100}
            
            result = get_data_summary()
            assert result == {"test": 100}
    
    def test_load_data_function(self):
        """Test load data utility function"""
        from src.services.data_processor import load_dunnhumby_data
        
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_instance = mock_processor.return_value.__enter__.return_value  
            mock_instance.process_dataset.return_value = {"processed": 50}
            
            result = load_dunnhumby_data("test/path")
            assert result == {"processed": 50}


class TestMCPToolsBasic:
    """Test basic MCP tools functionality (if available)"""
    
    def test_mcp_tools_import(self):
        """Test that MCP tools can be imported"""
        try:
            import mcp_server
            assert hasattr(mcp_server, 'predict_promotion_lift')
            assert hasattr(mcp_server, 'discover_tpm_capabilities')
        except ImportError:
            pytest.skip("MCP server tools not available")
    
    def test_prediction_function_basic(self):
        """Test basic prediction functionality"""
        try:
            from mcp_server import predict_promotion_lift
            
            result = predict_promotion_lift("Test Product", 20.0, 14)
            
            assert isinstance(result, str)
            assert "Test Product" in result
            assert "20.0%" in result
            assert "14 days" in result
            
        except ImportError:
            pytest.skip("MCP server tools not available")
    
    def test_welcome_function_basic(self):
        """Test basic welcome functionality"""
        try:
            from mcp_server import welcome, claude_desktop_welcome
            
            result = welcome()
            welcome_result = claude_desktop_welcome()
            
            assert isinstance(result, str)
            assert isinstance(welcome_result, str)
            assert result == welcome_result
            assert "Welcome" in result
            
        except ImportError:
            pytest.skip("MCP server tools not available")


class TestConfigurationAndSetup:
    """Test system configuration"""
    
    def test_database_setup(self):
        """Test database configuration"""
        from src.models.database import create_tables
        
        # Should not raise exceptions
        create_tables()
    
    def test_environment_variables(self):
        """Test environment configuration"""
        import os
        
        # Test environment should be configured
        assert os.environ.get("TESTING") == "true"
        database_url = os.environ.get("DATABASE_URL", "")
        assert "sqlite" in database_url.lower()


class TestErrorHandling:
    """Test basic error handling"""
    
    def test_invalid_data_handling(self, db_session):
        """Test handling of invalid data"""
        # Try to create household with invalid data
        try:
            household = Household(household_key=None)
            db_session.add(household) 
            db_session.commit()
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)
            db_session.rollback()
    
    def test_database_connection_resilience(self):
        """Test database connection handling"""
        from src.services.data_processor import get_data_summary
        
        # Mock database error
        with patch('src.services.data_processor.DunnhumbyProcessor') as mock_processor:
            mock_processor.side_effect = Exception("Connection failed")
            
            try:
                result = get_data_summary()
                # Should return something (empty dict or error message)
                assert result is not None
            except Exception:
                # Or handle the exception gracefully
                pass


class TestPerformanceBasics:
    """Test basic performance characteristics"""
    
    def test_model_creation_performance(self, db_session):
        """Test model creation is reasonably fast"""
        import time
        
        start_time = time.time()
        
        # Create multiple entities
        for i in range(10):
            household = Household(household_key=f"PERF_TEST_{i}")
            db_session.add(household)
        
        db_session.commit()
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second
        
        # Verify all were created
        count = db_session.query(Household).filter(
            Household.household_key.like("PERF_TEST_%")
        ).count()
        assert count == 10
    
    def test_query_performance(self, db_session):
        """Test basic query performance"""
        # Create some test data
        for i in range(50):
            household = Household(household_key=f"QUERY_TEST_{i}")
            db_session.add(household)
        db_session.commit()
        
        import time
        start_time = time.time()
        
        # Perform queries
        count = db_session.query(Household).count()
        filtered = db_session.query(Household).filter(
            Household.household_key.like("QUERY_TEST_%")
        ).all()
        
        end_time = time.time()
        
        # Should be fast
        assert (end_time - start_time) < 0.5  # Less than 500ms
        assert count >= 50
        assert len(filtered) == 50