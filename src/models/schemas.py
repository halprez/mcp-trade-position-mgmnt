from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Base schemas
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# Enums for type safety
class PromotionType(str, Enum):
    BOGO = "BOGO"
    DISCOUNT = "Discount" 
    DISPLAY = "Display"
    COUPON = "Coupon"
    BUNDLE = "Bundle"

class CampaignStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

# Household schemas
class HouseholdBase(BaseSchema):
    household_key: str
    age_desc: Optional[str] = None
    marital_status_code: Optional[str] = None
    income_desc: Optional[str] = None
    homeowner_desc: Optional[str] = None
    hh_comp_desc: Optional[str] = None
    household_size_desc: Optional[str] = None
    kid_category_desc: Optional[str] = None

class HouseholdResponse(HouseholdBase):
    id: int

# Product schemas
class ProductBase(BaseSchema):
    product_id: int
    department: Optional[str] = None
    commodity_desc: Optional[str] = None
    sub_commodity_desc: Optional[str] = None
    manufacturer: Optional[str] = None
    brand: Optional[str] = None
    curr_size_of_product: Optional[str] = None

class ProductResponse(ProductBase):
    id: int

class ProductSummary(BaseSchema):
    product_id: int
    brand: Optional[str] = None
    commodity_desc: Optional[str] = None
    total_sales: Optional[float] = None
    avg_price: Optional[float] = None

# Transaction schemas
class TransactionBase(BaseSchema):
    household_key: str
    basket_id: str
    product_id: int
    store_id: int
    day: int
    week_no: int
    trans_time: Optional[int] = 0
    quantity: int
    sales_value: float
    retail_disc: Optional[float] = 0.0
    coupon_disc: Optional[float] = 0.0
    coupon_match_disc: Optional[float] = 0.0

class TransactionResponse(TransactionBase):
    id: int

class TransactionSummary(BaseSchema):
    total_transactions: int
    total_revenue: float
    avg_basket_size: float
    unique_households: int
    unique_products: int

# Promotion schemas
class PromotionBase(BaseSchema):
    product_id: int
    store_id: int
    start_day: int
    end_day: int
    display_loc: Optional[str] = None
    mailer_loc: Optional[str] = None
    discount_percentage: Optional[float] = None
    promotion_type: Optional[PromotionType] = None

class PromotionCreate(PromotionBase):
    pass

class PromotionResponse(PromotionBase):
    id: int
    created_at: datetime

class PromotionPerformance(BaseSchema):
    promotion_id: int
    product_id: int
    promotion_type: PromotionType
    start_day: int
    end_day: int
    baseline_sales: float
    promoted_sales: float
    lift_percentage: float
    roi: float
    incremental_revenue: float

# Campaign schemas
class CampaignBase(BaseSchema):
    name: str
    description: Optional[str] = None
    start_date: date
    end_date: date
    budget: float
    status: CampaignStatus = CampaignStatus.PLANNED

class CampaignCreate(CampaignBase):
    pass

class CampaignUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    budget: Optional[float] = None
    status: Optional[CampaignStatus] = None

class CampaignResponse(CampaignBase):
    id: int
    actual_spend: float
    revenue_lift: Optional[float] = None
    roi: Optional[float] = None
    created_at: datetime
    updated_at: datetime

# Analytics request/response schemas
class AnalyticsTimeRange(BaseSchema):
    start_day: int
    end_day: int

class ProductAnalyticsRequest(BaseSchema):
    product_ids: List[int]
    time_range: Optional[AnalyticsTimeRange] = None
    include_promotions: bool = True

class ProductAnalyticsResponse(BaseSchema):
    product_id: int
    brand: Optional[str] = None
    commodity_desc: Optional[str] = None
    total_sales: float
    total_revenue: float
    avg_price: float
    promotion_count: int
    best_promotion_roi: Optional[float] = None

class PromotionROIRequest(BaseSchema):
    promotion_ids: Optional[List[int]] = None
    product_ids: Optional[List[int]] = None
    time_range: Optional[AnalyticsTimeRange] = None
    promotion_type: Optional[PromotionType] = None

class PromotionROIResponse(BaseSchema):
    promotion_id: int
    product_id: int
    product_name: Optional[str] = None
    promotion_type: PromotionType
    start_day: int
    end_day: int
    investment: float
    incremental_revenue: float
    roi: float
    lift_percentage: float

# ML Prediction schemas  
class LiftPredictionRequest(BaseSchema):
    product_id: int
    store_id: int
    discount_percentage: float
    duration_days: int
    display_location: Optional[str] = None
    promotion_type: PromotionType

class LiftPredictionResponse(BaseSchema):
    product_id: int
    predicted_lift_percentage: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    expected_incremental_units: int
    expected_incremental_revenue: float

class BudgetOptimizationRequest(BaseSchema):
    total_budget: float = Field(..., gt=0, description="Total budget to allocate")
    target_products: List[int] = Field(..., min_length=1, description="Product IDs to include")
    objective: str = Field(default="maximize_revenue", description="Optimization objective")
    constraints: Optional[Dict[str, Any]] = None

class BudgetOptimizationResponse(BaseSchema):
    total_budget: float
    allocated_budget: float
    recommendations: List[Dict[str, Any]]
    expected_roi: float
    expected_lift: float

# Dashboard summary schemas
class DashboardSummary(BaseSchema):
    total_households: int
    total_products: int
    total_transactions: int
    total_promotions: int
    active_campaigns: int
    total_revenue: float
    avg_promotion_roi: Optional[float] = None

class TopPerformer(BaseSchema):
    id: int
    name: str
    value: float
    metric: str

class DashboardResponse(BaseSchema):
    summary: DashboardSummary
    top_products: List[TopPerformer]
    top_promotions: List[TopPerformer]
    recent_campaigns: List[CampaignResponse]

# Error response schemas
class ErrorDetail(BaseSchema):
    field: Optional[str] = None
    message: str
    code: Optional[str] = None

class ErrorResponse(BaseSchema):
    error: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None