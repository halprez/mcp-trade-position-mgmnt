"""Tests for database models and entities"""
import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from src.models.entities import (
    Household, Product, Store, Transaction, Promotion, Campaign
)


class TestHouseholdModel:
    """Test Household entity model"""
    
    def test_household_creation(self, db_session):
        """Test basic household creation"""
        household = Household(
            household_key="HH_TEST_001",
            age_desc="35-44",
            marital_status_code="M",
            income_desc="50-74K",
            homeowner_desc="Homeowner",
            hh_comp_desc="2 Adults Kids",
            household_size_desc="4",
            kid_category_desc="1-2"
        )
        
        db_session.add(household)
        db_session.commit()
        
        saved_household = db_session.query(Household).filter_by(household_key="HH_TEST_001").first()
        assert saved_household is not None
        assert saved_household.age_desc == "35-44"
        assert saved_household.income_desc == "50-74K"
    
    def test_household_unique_key(self, db_session):
        """Test household key uniqueness constraint"""
        household1 = Household(household_key="HH_DUPLICATE")
        household2 = Household(household_key="HH_DUPLICATE")
        
        db_session.add(household1)
        db_session.commit()
        
        db_session.add(household2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_household_string_representation(self, db_session):
        """Test household string representation"""
        household = Household(household_key="HH_TEST_STR")
        assert "HH_TEST_STR" in str(household)
    
    def test_household_minimal_creation(self, db_session):
        """Test household creation with minimal required fields"""
        household = Household(household_key="HH_MINIMAL")
        db_session.add(household)
        db_session.commit()
        
        saved = db_session.query(Household).filter_by(household_key="HH_MINIMAL").first()
        assert saved is not None
        assert saved.household_key == "HH_MINIMAL"


class TestProductModel:
    """Test Product entity model"""
    
    def test_product_creation(self, db_session):
        """Test basic product creation"""
        product = Product(
            product_id=1001,
            department="DAIRY",
            commodity_desc="MILK",
            sub_commodity_desc="REFRIGERATED REGULAR MILK",
            manufacturer="PRIVATE LABEL",
            brand="STORE BRAND",
            curr_size_of_product="1 GALLON"
        )
        
        db_session.add(product)
        db_session.commit()
        
        saved_product = db_session.query(Product).filter_by(product_id=1001).first()
        assert saved_product is not None
        assert saved_product.department == "DAIRY"
        assert saved_product.brand == "STORE BRAND"
    
    def test_product_unique_id(self, db_session):
        """Test product ID uniqueness constraint"""
        product1 = Product(product_id=2001, department="TEST")
        product2 = Product(product_id=2001, department="TEST2")
        
        db_session.add(product1)
        db_session.commit()
        
        db_session.add(product2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_product_string_representation(self, db_session):
        """Test product string representation"""
        product = Product(product_id=1234, brand="TEST_BRAND")
        assert "1234" in str(product)
        assert "TEST_BRAND" in str(product)


class TestStoreModel:
    """Test Store entity model"""
    
    def test_store_creation(self, db_session):
        """Test basic store creation"""
        store = Store(store_id=100)
        
        db_session.add(store)
        db_session.commit()
        
        saved_store = db_session.query(Store).filter_by(store_id=100).first()
        assert saved_store is not None
        assert saved_store.store_id == 100
    
    def test_store_unique_id(self, db_session):
        """Test store ID uniqueness constraint"""
        store1 = Store(store_id=200)
        store2 = Store(store_id=200)
        
        db_session.add(store1)
        db_session.commit()
        
        db_session.add(store2)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestTransactionModel:
    """Test Transaction entity model"""
    
    def test_transaction_creation(self, populated_db_session):
        """Test basic transaction creation"""
        transaction = Transaction(
            household_key="HH_001",
            basket_id="BASKET_TEST",
            product_id=1001,
            store_id=1,
            day=100,
            week_no=15,
            trans_time=1200,
            quantity=2,
            sales_value=5.99,
            retail_disc=0.50,
            coupon_disc=0.25,
            coupon_match_disc=0.10
        )
        
        populated_db_session.add(transaction)
        populated_db_session.commit()
        
        saved_transaction = populated_db_session.query(Transaction).filter_by(basket_id="BASKET_TEST").first()
        assert saved_transaction is not None
        assert saved_transaction.sales_value == 5.99
        assert saved_transaction.quantity == 2
    
    def test_transaction_numeric_fields(self, populated_db_session):
        """Test transaction numeric field types"""
        transaction = Transaction(
            household_key="HH_001",
            basket_id="BASKET_NUMERIC",
            product_id=1001,
            store_id=1,
            day=1,
            week_no=1,
            quantity=3,
            sales_value=12.99
        )
        
        populated_db_session.add(transaction)
        populated_db_session.commit()
        
        saved = populated_db_session.query(Transaction).filter_by(basket_id="BASKET_NUMERIC").first()
        assert isinstance(saved.quantity, int)
        assert isinstance(saved.sales_value, float)
        assert isinstance(saved.day, int)
    
    def test_transaction_relationships(self, populated_db_session):
        """Test transaction can reference existing household and product"""
        # Existing data should be available from populated_db_session fixture
        households = populated_db_session.query(Household).all()
        products = populated_db_session.query(Product).all()
        
        assert len(households) > 0
        assert len(products) > 0
        
        # Transaction referencing existing entities should work
        transaction = Transaction(
            household_key=households[0].household_key,
            basket_id="BASKET_REL_TEST",
            product_id=products[0].product_id,
            store_id=1,
            day=1,
            week_no=1,
            quantity=1,
            sales_value=1.99
        )
        
        populated_db_session.add(transaction)
        populated_db_session.commit()
        
        saved = populated_db_session.query(Transaction).filter_by(basket_id="BASKET_REL_TEST").first()
        assert saved is not None


class TestPromotionModel:
    """Test Promotion entity model"""
    
    def test_promotion_creation(self, db_session):
        """Test basic promotion creation"""
        promotion = Promotion(
            product_id=1001,
            store_id=1,
            start_day=1,
            end_day=14,
            display_loc="FRONT",
            mailer_loc="PAGE 1",
            promotion_type="DISCOUNT"
        )
        
        db_session.add(promotion)
        db_session.commit()
        
        saved_promotion = db_session.query(Promotion).first()
        assert saved_promotion is not None
        assert saved_promotion.promotion_type == "DISCOUNT"
        assert saved_promotion.start_day == 1
        assert saved_promotion.end_day == 14
    
    def test_promotion_duration_calculation(self, db_session):
        """Test promotion duration logic"""
        promotion = Promotion(
            product_id=1001,
            store_id=1,
            start_day=10,
            end_day=24,
            promotion_type="BOGO"
        )
        
        db_session.add(promotion)
        db_session.commit()
        
        duration = promotion.end_day - promotion.start_day
        assert duration == 14
    
    def test_promotion_optional_fields(self, db_session):
        """Test promotion with optional fields"""
        promotion = Promotion(
            product_id=1002,
            store_id=2,
            start_day=5,
            end_day=19,
            promotion_type="COUPON"
        )
        
        db_session.add(promotion)
        db_session.commit()
        
        saved = db_session.query(Promotion).filter_by(product_id=1002).first()
        assert saved.display_loc is None or saved.display_loc == ""
        assert saved.mailer_loc is None or saved.mailer_loc == ""


class TestCampaignModel:
    """Test Campaign entity model"""
    
    def test_campaign_creation(self, db_session):
        """Test basic campaign creation"""
        campaign = Campaign(
            name="CAMP_001",
            description="Test Campaign",
            start_date=datetime(2024, 1, 1).date(),
            end_date=datetime(2024, 1, 31).date(),
            budget=10000.0
        )
        
        db_session.add(campaign)
        db_session.commit()
        
        saved_campaign = db_session.query(Campaign).filter_by(name="CAMP_001").first()
        assert saved_campaign is not None
        assert saved_campaign.description == "Test Campaign"
        assert saved_campaign.budget == 10000.0
    
    def test_campaign_date_fields(self, db_session):
        """Test campaign date handling"""
        start_date = datetime(2024, 2, 1).date()
        end_date = datetime(2024, 2, 28).date()
        
        campaign = Campaign(
            name="CAMP_DATE",
            start_date=start_date,
            end_date=end_date
        )
        
        db_session.add(campaign)
        db_session.commit()
        
        saved = db_session.query(Campaign).filter_by(name="CAMP_DATE").first()
        assert saved.start_date == start_date
        assert saved.end_date == end_date
    
    def test_campaign_string_representation(self, db_session):
        """Test campaign string representation"""
        campaign = Campaign(name="CAMP_STR", description="Test Description")
        assert "CAMP_STR" in str(campaign)


class TestModelRelationships:
    """Test relationships between models"""
    
    def test_transaction_references(self, populated_db_session):
        """Test that transactions properly reference other entities"""
        # Get existing data
        household = populated_db_session.query(Household).first()
        product = populated_db_session.query(Product).first()
        store = populated_db_session.query(Store).first()
        
        assert household is not None
        assert product is not None
        assert store is not None
        
        # Create transaction with valid references
        transaction = Transaction(
            household_key=household.household_key,
            basket_id="REL_TEST",
            product_id=product.product_id,
            store_id=store.store_id,
            day=1,
            week_no=1,
            quantity=1,
            sales_value=9.99
        )
        
        populated_db_session.add(transaction)
        populated_db_session.commit()
        
        # Verify transaction was created successfully
        saved = populated_db_session.query(Transaction).filter_by(basket_id="REL_TEST").first()
        assert saved is not None
        assert saved.household_key == household.household_key
        assert saved.product_id == product.product_id
        assert saved.store_id == store.store_id
    
    def test_promotion_product_reference(self, populated_db_session):
        """Test promotion references to products"""
        product = populated_db_session.query(Product).first()
        store = populated_db_session.query(Store).first()
        
        promotion = Promotion(
            product_id=product.product_id,
            store_id=store.store_id,
            start_day=1,
            end_day=7,
            promotion_type="DISPLAY"
        )
        
        populated_db_session.add(promotion)
        populated_db_session.commit()
        
        saved = populated_db_session.query(Promotion).first()
        assert saved.product_id == product.product_id
        assert saved.store_id == store.store_id


class TestModelValidation:
    """Test model field validation and constraints"""
    
    def test_household_key_not_null(self, db_session):
        """Test household key cannot be null"""
        with pytest.raises((IntegrityError, ValueError)):
            household = Household(household_key=None)
            db_session.add(household)
            db_session.commit()
    
    def test_product_id_not_null(self, db_session):
        """Test product ID cannot be null"""
        with pytest.raises((IntegrityError, ValueError)):
            product = Product(product_id=None)
            db_session.add(product)
            db_session.commit()
    
    def test_transaction_required_fields(self, db_session):
        """Test transaction requires certain fields"""
        # Missing required fields should fail
        with pytest.raises((IntegrityError, ValueError)):
            transaction = Transaction()  # No required fields
            db_session.add(transaction)
            db_session.commit()


class TestModelDefaults:
    """Test model default values"""
    
    def test_transaction_default_values(self, populated_db_session):
        """Test transaction default field values"""
        transaction = Transaction(
            household_key="HH_001",
            basket_id="DEFAULT_TEST",
            product_id=1001,
            store_id=1,
            day=1,
            week_no=1,
            quantity=1,
            sales_value=1.99
            # Not setting discount fields - should default to 0
        )
        
        populated_db_session.add(transaction)
        populated_db_session.commit()
        
        saved = populated_db_session.query(Transaction).filter_by(basket_id="DEFAULT_TEST").first()
        # Discount fields should default to 0
        assert saved.retail_disc == 0.0
        assert saved.coupon_disc == 0.0
        assert saved.coupon_match_disc == 0.0
    
    def test_promotion_default_values(self, db_session):
        """Test promotion default field values"""
        promotion = Promotion(
            product_id=1001,
            store_id=1,
            start_day=1,
            end_day=7
            # Not setting optional fields
        )
        
        db_session.add(promotion)
        db_session.commit()
        
        saved = db_session.query(Promotion).first()
        # Optional fields should be empty or None
        assert saved.display_loc in [None, ""]
        assert saved.mailer_loc in [None, ""]
        assert saved.promotion_type in [None, "", "Unknown"]


class TestModelQueries:
    """Test common query patterns on models"""
    
    def test_household_queries(self, populated_db_session):
        """Test household query patterns"""
        # Count households
        count = populated_db_session.query(Household).count()
        assert count > 0
        
        # Filter by age group
        age_filtered = populated_db_session.query(Household).filter_by(age_desc="35-44").all()
        assert isinstance(age_filtered, list)
        
        # Filter by income
        income_filtered = populated_db_session.query(Household).filter(
            Household.income_desc.like("%50-74K%")
        ).all()
        assert isinstance(income_filtered, list)
    
    def test_transaction_aggregations(self, populated_db_session):
        """Test transaction aggregation queries"""
        from sqlalchemy import func
        
        # Total sales value
        total_sales = populated_db_session.query(func.sum(Transaction.sales_value)).scalar()
        assert total_sales is not None
        
        # Transaction count by product
        product_transactions = populated_db_session.query(
            Transaction.product_id,
            func.count(Transaction.id).label('count')
        ).group_by(Transaction.product_id).all()
        
        assert len(product_transactions) > 0
    
    def test_promotion_date_queries(self, populated_db_session):
        """Test promotion date-based queries"""
        # Active promotions on a specific day
        day = 10
        active_promotions = populated_db_session.query(Promotion).filter(
            Promotion.start_day <= day,
            Promotion.end_day >= day
        ).all()
        
        assert isinstance(active_promotions, list)