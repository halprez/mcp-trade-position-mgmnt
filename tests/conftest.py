"""Pytest configuration and fixtures for TPM MCP Server tests"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.database import Base
from src.models.entities import Campaign, Household, Product, Promotion, Store, Transaction


@pytest.fixture(scope="session")
def test_database():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield engine, SessionLocal
    
    engine.dispose()


@pytest.fixture
def db_session(test_database):
    """Create database session for individual tests"""
    engine, SessionLocal = test_database
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def sample_households():
    """Sample household data for testing"""
    return [
        {
            "household_key": "HH_001",
            "age_desc": "35-44",
            "marital_status_code": "M",
            "income_desc": "50-74K",
            "homeowner_desc": "Homeowner",
            "hh_comp_desc": "2 Adults Kids",
            "household_size_desc": "4",
            "kid_category_desc": "1-2"
        },
        {
            "household_key": "HH_002", 
            "age_desc": "25-34",
            "marital_status_code": "S",
            "income_desc": "25-49K",
            "homeowner_desc": "Renter",
            "hh_comp_desc": "Single Female",
            "household_size_desc": "1",
            "kid_category_desc": "None/Unknown"
        }
    ]


@pytest.fixture
def sample_products():
    """Sample product data for testing"""
    return [
        {
            "product_id": 1001,
            "department": "DAIRY",
            "commodity_desc": "MILK",
            "sub_commodity_desc": "REFRIGERATED REGULAR MILK",
            "manufacturer": "PRIVATE LABEL",
            "brand": "STORE BRAND",
            "curr_size_of_product": "1 GALLON"
        },
        {
            "product_id": 1002,
            "department": "GROCERY",
            "commodity_desc": "CEREAL",
            "sub_commodity_desc": "COLD CEREAL",
            "manufacturer": "GENERAL MILLS",
            "brand": "CHEERIOS",
            "curr_size_of_product": "12 OZ"
        }
    ]


@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing"""
    return [
        {
            "household_key": "HH_001",
            "basket_id": "BASKET_001",
            "product_id": 1001,
            "store_id": 1,
            "day": 1,
            "week_no": 1,
            "trans_time": 1200,
            "quantity": 2,
            "sales_value": 6.99,
            "retail_disc": 0.50,
            "coupon_disc": 0.00,
            "coupon_match_disc": 0.00
        },
        {
            "household_key": "HH_002",
            "basket_id": "BASKET_002", 
            "product_id": 1002,
            "store_id": 1,
            "day": 2,
            "week_no": 1,
            "trans_time": 1500,
            "quantity": 1,
            "sales_value": 4.49,
            "retail_disc": 0.00,
            "coupon_disc": 0.75,
            "coupon_match_disc": 0.00
        }
    ]


@pytest.fixture
def sample_promotions():
    """Sample promotion data for testing"""
    return [
        {
            "product_id": 1001,
            "store_id": 1,
            "start_day": 1,
            "end_day": 14,
            "display_loc": "FRONT",
            "mailer_loc": "PAGE 1",
            "promotion_type": "DISCOUNT"
        },
        {
            "product_id": 1002,
            "store_id": 1,
            "start_day": 5,
            "end_day": 19,
            "display_loc": "",
            "mailer_loc": "PAGE 2",
            "promotion_type": "BOGO"
        }
    ]


@pytest.fixture
def populated_db_session(db_session, sample_households, sample_products, sample_transactions, sample_promotions):
    """Database session with sample data loaded"""
    # Add households
    for hh_data in sample_households:
        household = Household(**hh_data)
        db_session.add(household)
    
    # Add products
    for prod_data in sample_products:
        product = Product(**prod_data)
        db_session.add(product)
    
    # Add stores
    store = Store(store_id=1)
    db_session.add(store)
    
    # Add transactions
    for trans_data in sample_transactions:
        transaction = Transaction(**trans_data)
        db_session.add(transaction)
    
    # Add promotions
    for promo_data in sample_promotions:
        promotion = Promotion(**promo_data)
        db_session.add(promotion)
    
    db_session.commit()
    
    return db_session


@pytest.fixture
def temp_csv_data():
    """Create temporary CSV files for testing data processing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV data
    households_df = pd.DataFrame([
        {"household_key": "HH_TEST_001", "AGE_DESC": "25-34", "MARITAL_STATUS_CODE": "M", 
         "INCOME_DESC": "50-74K", "HOMEOWNER_DESC": "Homeowner", "HH_COMP_DESC": "2 Adults No Kids",
         "HOUSEHOLD_SIZE_DESC": "2", "KID_CATEGORY_DESC": "None/Unknown"}
    ])
    
    products_df = pd.DataFrame([
        {"PRODUCT_ID": 2001, "DEPARTMENT": "DAIRY", "COMMODITY_DESC": "MILK",
         "SUB_COMMODITY_DESC": "REFRIGERATED REGULAR MILK", "MANUFACTURER": "PRIVATE LABEL",
         "BRAND": "STORE BRAND", "CURR_SIZE_OF_PRODUCT": "1 GALLON"}
    ])
    
    transactions_df = pd.DataFrame([
        {"household_key": "HH_TEST_001", "BASKET_ID": "TEST_BASKET_001", "PRODUCT_ID": 2001,
         "STORE_ID": 1, "DAY": 1, "WEEK_NO": 1, "TRANS_TIME": 1200, "QUANTITY": 1,
         "SALES_VALUE": 3.99, "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0}
    ])
    
    causal_df = pd.DataFrame([
        {"PRODUCT_ID": 2001, "STORE_ID": 1, "START_DAY": 1, "END_DAY": 7,
         "DISPLAY_LOC": "FRONT", "MAILER_LOC": "", "PROMOTION_TYPE": "DISCOUNT"}
    ])
    
    # Save to CSV files
    temp_path = Path(temp_dir)
    households_df.to_csv(temp_path / "hh_demographic.csv", index=False)
    products_df.to_csv(temp_path / "product.csv", index=False)
    transactions_df.to_csv(temp_path / "transaction_data.csv", index=False)
    causal_df.to_csv(temp_path / "causal_data.csv", index=False)
    
    # Create empty files for optional data
    (temp_path / "coupon.csv").touch()
    (temp_path / "coupon_redempt.csv").touch()
    (temp_path / "campaign_desc.csv").touch()
    (temp_path / "campaign_table.csv").touch()
    
    yield temp_path
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools for testing without actual MCP server"""
    class MockMCPTool:
        def __init__(self, name, func):
            self.name = name
            self.func = func
            
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    
    return MockMCPTool


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["TESTING"] = "true"
    yield
    # Cleanup after test
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]
    if "TESTING" in os.environ:
        del os.environ["TESTING"]