"""Tests for data processing functionality"""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.services.data_processor import DunnhumbyProcessor, load_dunnhumby_data, get_data_summary
from src.models.entities import Household, Product, Store, Transaction, Promotion


class TestDunnhumbyProcessor:
    """Test DunnhumbyProcessor class functionality"""
    
    def test_init(self, db_session):
        """Test processor initialization"""
        processor = DunnhumbyProcessor("test/path", db_session)
        assert processor.data_path == Path("test/path")
        assert processor.session == db_session
        assert isinstance(processor._processing_stats, dict)
    
    def test_get_expected_files(self, db_session):
        """Test expected files mapping"""
        processor = DunnhumbyProcessor(session=db_session)
        expected_files = processor.get_expected_files()
        
        expected_keys = [
            "households", "products", "transactions", "promotions",
            "coupons", "coupon_redemptions", "campaign_descriptions", "campaign_table"
        ]
        
        for key in expected_keys:
            assert key in expected_files
            assert expected_files[key].endswith('.csv')
    
    def test_load_csv_files_empty_directory(self, db_session, tmp_path):
        """Test loading CSV files from empty directory"""
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        dataframes = processor.load_csv_files()
        
        assert isinstance(dataframes, dict)
        assert len(dataframes) == 0
    
    def test_load_csv_files_with_data(self, db_session, temp_csv_data):
        """Test loading CSV files with actual data"""
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        dataframes = processor.load_csv_files()
        
        assert "households" in dataframes
        assert "products" in dataframes  
        assert "transactions" in dataframes
        assert "promotions" in dataframes
        
        # Check data was loaded correctly
        assert len(dataframes["households"]) == 1
        assert dataframes["households"].iloc[0]["household_key"] == "HH_TEST_001"
    
    def test_load_csv_files_empty_files(self, db_session, tmp_path):
        """Test loading empty CSV files"""
        # Create empty files
        (tmp_path / "hh_demographic.csv").touch()
        (tmp_path / "campaign_desc.csv").touch()
        
        processor = DunnhumbyProcessor(str(tmp_path), db_session)
        dataframes = processor.load_csv_files()
        
        # Empty files should be skipped
        assert "households" not in dataframes
        assert "campaign_descriptions" not in dataframes
    
    def test_clear_existing_data(self, populated_db_session):
        """Test clearing existing database data"""
        # Verify data exists
        assert populated_db_session.query(Household).count() > 0
        assert populated_db_session.query(Product).count() > 0
        assert populated_db_session.query(Transaction).count() > 0
        
        processor = DunnhumbyProcessor(session=populated_db_session)
        processor.clear_existing_data()
        
        # Verify data was cleared
        assert populated_db_session.query(Transaction).count() == 0
        assert populated_db_session.query(Promotion).count() == 0
        assert populated_db_session.query(Store).count() == 0
        assert populated_db_session.query(Product).count() == 0
        assert populated_db_session.query(Household).count() == 0
    
    def test_process_households(self, db_session):
        """Test household data processing"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # Create test data
        hh_df = pd.DataFrame([
            {
                "household_key": "HH_TEST_001",
                "AGE_DESC": "35-44",
                "MARITAL_STATUS_CODE": "M",
                "INCOME_DESC": "50-74K",
                "HOMEOWNER_DESC": "Homeowner",
                "HH_COMP_DESC": "2 Adults Kids",
                "HOUSEHOLD_SIZE_DESC": "4",
                "KID_CATEGORY_DESC": "1-2"
            }
        ])
        
        count = processor.process_households(hh_df)
        
        assert count == 1
        household = db_session.query(Household).first()
        assert household.household_key == "HH_TEST_001"
        assert household.age_desc == "35-44"
        assert household.income_desc == "50-74K"
    
    def test_process_products(self, db_session):
        """Test product data processing"""
        processor = DunnhumbyProcessor(session=db_session)
        
        prod_df = pd.DataFrame([
            {
                "PRODUCT_ID": 1001,
                "DEPARTMENT": "DAIRY",
                "COMMODITY_DESC": "MILK",
                "SUB_COMMODITY_DESC": "REFRIGERATED REGULAR MILK",
                "MANUFACTURER": "PRIVATE LABEL",
                "BRAND": "STORE BRAND",
                "CURR_SIZE_OF_PRODUCT": "1 GALLON"
            }
        ])
        
        count = processor.process_products(prod_df)
        
        assert count == 1
        product = db_session.query(Product).first()
        assert product.product_id == 1001
        assert product.department == "DAIRY"
        assert product.brand == "STORE BRAND"
    
    def test_process_transactions(self, db_session):
        """Test transaction data processing"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # First create required household and product
        household = Household(household_key="HH_001")
        product = Product(product_id=1001, department="DAIRY", commodity_desc="MILK")
        db_session.add_all([household, product])
        db_session.commit()
        
        trans_df = pd.DataFrame([
            {
                "household_key": "HH_001",
                "BASKET_ID": "BASKET_001",
                "PRODUCT_ID": 1001,
                "STORE_ID": 1,
                "DAY": 1,
                "WEEK_NO": 1,
                "TRANS_TIME": 1200,
                "QUANTITY": 2,
                "SALES_VALUE": 6.99,
                "RETAIL_DISC": 0.50,
                "COUPON_DISC": 0.00,
                "COUPON_MATCH_DISC": 0.00
            }
        ])
        
        count = processor.process_transactions(trans_df)
        
        assert count == 1
        transaction = db_session.query(Transaction).first()
        assert transaction.household_key == "HH_001"
        assert transaction.product_id == 1001
        assert transaction.sales_value == 6.99
        
        # Check that store was created
        store = db_session.query(Store).first()
        assert store.store_id == 1
    
    def test_process_promotions(self, db_session):
        """Test promotion data processing"""
        processor = DunnhumbyProcessor(session=db_session)
        
        promo_df = pd.DataFrame([
            {
                "PRODUCT_ID": 1001,
                "STORE_ID": 1,
                "START_DAY": 1,
                "END_DAY": 14,
                "DISPLAY_LOC": "FRONT",
                "MAILER_LOC": "PAGE 1",
                "PROMOTION_TYPE": "DISCOUNT"
            }
        ])
        
        count = processor.process_promotions(promo_df)
        
        assert count == 1
        promotion = db_session.query(Promotion).first()
        assert promotion.product_id == 1001
        assert promotion.promotion_type == "DISCOUNT"
        assert promotion.start_day == 1
        assert promotion.end_day == 14
    
    def test_process_promotions_empty(self, db_session):
        """Test processing empty promotions dataframe"""
        processor = DunnhumbyProcessor(session=db_session)
        
        empty_df = pd.DataFrame()
        count = processor.process_promotions(empty_df)
        
        assert count == 0
    
    @patch('src.services.data_processor.DunnhumbyProcessor.load_csv_files')
    def test_process_dataset_no_files(self, mock_load_csv, db_session):
        """Test processing dataset with no files"""
        mock_load_csv.return_value = {}
        
        processor = DunnhumbyProcessor(session=db_session)
        results = processor.process_dataset()
        
        assert results == {}
    
    def test_process_dataset_full(self, db_session, temp_csv_data):
        """Test full dataset processing"""
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        results = processor.process_dataset()
        
        assert "households" in results
        assert "products" in results
        assert "transactions" in results
        assert "promotions" in results
        
        # Verify actual data was processed
        assert results["households"] > 0
        assert results["products"] > 0
        assert results["transactions"] > 0
        assert results["promotions"] > 0
    
    def test_get_data_summary(self, populated_db_session):
        """Test data summary generation"""
        processor = DunnhumbyProcessor(session=populated_db_session)
        summary = processor.get_data_summary()
        
        assert "households" in summary
        assert "products" in summary
        assert "stores" in summary
        assert "transactions" in summary
        assert "promotions" in summary
        
        assert summary["households"] > 0
        assert summary["products"] > 0
        assert summary["transactions"] > 0


class TestDataProcessingUtilities:
    """Test utility functions for data processing"""
    
    @patch('src.services.data_processor.DunnhumbyProcessor')
    def test_load_dunnhumby_data(self, mock_processor_class):
        """Test load_dunnhumby_data utility function"""
        mock_processor = MagicMock()
        mock_processor.process_dataset.return_value = {"test": 100}
        mock_processor_class.return_value.__enter__.return_value = mock_processor
        
        result = load_dunnhumby_data("test/path")
        
        assert result == {"test": 100}
        mock_processor_class.assert_called_once_with("test/path")
        mock_processor.process_dataset.assert_called_once()
    
    @patch('src.services.data_processor.DunnhumbyProcessor')
    def test_get_data_summary_utility(self, mock_processor_class):
        """Test get_data_summary utility function"""
        mock_processor = MagicMock()
        mock_processor.get_data_summary.return_value = {"households": 100}
        mock_processor_class.return_value.__enter__.return_value = mock_processor
        
        result = get_data_summary()
        
        assert result == {"households": 100}
        mock_processor.get_data_summary.assert_called_once()


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_batch_process_small_dataset(self, db_session):
        """Test batch processing with small dataset"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # Create test data
        data = pd.DataFrame([
            {"household_key": f"HH_{i:03d}", "AGE_DESC": "25-34"} 
            for i in range(5)
        ])
        
        def create_batch(batch):
            return [Household(household_key=row["household_key"], age_desc=row["AGE_DESC"]) 
                    for _, row in batch.iterrows()]
        
        count = processor.batch_process(data, create_batch, batch_size=2)
        
        assert count == 5
        assert db_session.query(Household).count() == 5
    
    def test_chunk_process(self, db_session):
        """Test chunk processing functionality"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # Create larger test dataset
        data = pd.DataFrame([
            {"value": i} for i in range(100)
        ])
        
        def process_chunk(chunk):
            # Simulate processing
            return len(chunk)
        
        count = processor.chunk_process(data, process_chunk, chunk_size=25)
        
        assert count == 100


class TestErrorHandling:
    """Test error handling in data processing"""
    
    def test_clear_data_error_handling(self, db_session):
        """Test error handling in clear_existing_data"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # Mock a database error
        with patch.object(db_session, 'query') as mock_query:
            mock_query.side_effect = Exception("Database error")
            
            # Should not raise exception
            processor.clear_existing_data()
    
    def test_create_stores_error_handling(self, db_session):
        """Test error handling in store creation"""
        processor = DunnhumbyProcessor(session=db_session)
        
        # Mock database error during store creation
        with patch.object(db_session, 'add_all') as mock_add:
            mock_add.side_effect = Exception("Database error")
            
            # Should not raise exception
            processor._create_stores([1, 2, 3])
    
    def test_get_data_summary_error_handling(self, db_session):
        """Test error handling in data summary"""
        processor = DunnhumbyProcessor(session=db_session)
        
        with patch.object(db_session, 'query') as mock_query:
            mock_query.side_effect = Exception("Database error")
            
            summary = processor.get_data_summary()
            assert isinstance(summary, dict)


class TestContextManager:
    """Test context manager functionality"""
    
    def test_context_manager_enter_exit(self, db_session):
        """Test context manager properly closes session"""
        with patch.object(db_session, 'close') as mock_close:
            with DunnhumbyProcessor(session=db_session) as processor:
                assert isinstance(processor, DunnhumbyProcessor)
            
            mock_close.assert_called_once()
    
    def test_context_manager_exception_handling(self, db_session):
        """Test context manager handles exceptions properly"""
        with patch.object(db_session, 'close') as mock_close:
            try:
                with DunnhumbyProcessor(session=db_session) as processor:
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            mock_close.assert_called_once()


class TestProcessingStats:
    """Test processing statistics tracking"""
    
    def test_update_stats(self, db_session):
        """Test statistics tracking"""
        processor = DunnhumbyProcessor(session=db_session)
        
        processor._update_stats("households", 100)
        processor._update_stats("products", 50)
        
        stats = processor.get_processing_summary()
        assert stats["households"] == 100
        assert stats["products"] == 50
    
    def test_processing_stats_integration(self, db_session, temp_csv_data):
        """Test stats are updated during processing"""
        processor = DunnhumbyProcessor(str(temp_csv_data), db_session)
        processor.process_dataset()
        
        stats = processor.get_processing_summary()
        assert len(stats) > 0
        assert all(count >= 0 for count in stats.values())