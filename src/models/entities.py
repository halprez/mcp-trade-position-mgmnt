from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Household(Base):
    __tablename__ = "households"
    
    id = Column(Integer, primary_key=True)
    household_key = Column(String, unique=True, index=True)
    age_desc = Column(String)
    marital_status_code = Column(String)
    income_desc = Column(String)
    homeowner_desc = Column(String)
    hh_comp_desc = Column(String)
    household_size_desc = Column(String)
    kid_category_desc = Column(String)
    
    # Relationships  
    transactions = relationship("Transaction", back_populates="household")
    
    def __repr__(self):
        return f"<Household(key='{self.household_key}')>"

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, unique=True, index=True)
    department = Column(String)
    commodity_desc = Column(String)
    sub_commodity_desc = Column(String)
    manufacturer = Column(String)
    brand = Column(String)
    curr_size_of_product = Column(String)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="product")
    
    def __repr__(self):
        return f"<Product(id={self.product_id}, brand='{self.brand}')>"

class Store(Base):
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, unique=True, index=True)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="store")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    household_key = Column(String, ForeignKey("households.household_key"))
    basket_id = Column(String, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"))
    store_id = Column(Integer, ForeignKey("stores.store_id"))
    
    day = Column(Integer)
    week_no = Column(Integer)
    trans_time = Column(Integer)
    
    quantity = Column(Integer)
    sales_value = Column(Float)
    retail_disc = Column(Float)
    coupon_disc = Column(Float)
    coupon_match_disc = Column(Float)
    
    # Relationships
    household = relationship("Household", back_populates="transactions")
    product = relationship("Product", back_populates="transactions")
    store = relationship("Store", back_populates="transactions")

class Promotion(Base):
    __tablename__ = "promotions"
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.product_id"))
    store_id = Column(Integer, ForeignKey("stores.store_id"))
    
    start_day = Column(Integer)
    end_day = Column(Integer)
    display_loc = Column(String)
    mailer_loc = Column(String)
    
    # Calculated fields
    discount_percentage = Column(Float)
    promotion_type = Column(String)  # BOGO, Discount, Display, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    start_date = Column(Date)
    end_date = Column(Date)
    budget = Column(Float)
    status = Column(String, default="planned")  # planned, active, completed, cancelled
    
    # Performance metrics
    actual_spend = Column(Float, default=0.0)
    revenue_lift = Column(Float)
    roi = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Campaign(name='{self.name}')>"