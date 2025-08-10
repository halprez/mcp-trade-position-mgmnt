from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.entities import Campaign, Product, Promotion, Store
from ..models.schemas import (
    CampaignCreate,
    CampaignResponse,
    CampaignStatus,
    CampaignUpdate,
    PromotionCreate,
    PromotionResponse,
    PromotionType,
)

router = APIRouter(prefix="/promotions", tags=["promotions"])


# Promotion CRUD endpoints
@router.get("/", response_model=list[PromotionResponse])
def list_promotions(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    promotion_type: PromotionType | None = Query(
        None, description="Filter by promotion type"
    ),
    product_id: int | None = Query(None, description="Filter by product ID"),
    store_id: int | None = Query(None, description="Filter by store ID"),
    db: Session = Depends(get_db),
):
    """Get list of promotions with optional filtering"""
    query = db.query(Promotion)

    if promotion_type:
        query = query.filter(Promotion.promotion_type == promotion_type.value)
    if product_id:
        query = query.filter(Promotion.product_id == product_id)
    if store_id:
        query = query.filter(Promotion.store_id == store_id)

    promotions = (
        query.order_by(desc(Promotion.created_at)).offset(skip).limit(limit).all()
    )
    return promotions


@router.get("/{promotion_id}", response_model=PromotionResponse)
def get_promotion(promotion_id: int, db: Session = Depends(get_db)):
    """Get a specific promotion by ID"""
    promotion = db.query(Promotion).filter(Promotion.id == promotion_id).first()
    if not promotion:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return promotion


@router.post("/", response_model=PromotionResponse, status_code=201)
def create_promotion(promotion: PromotionCreate, db: Session = Depends(get_db)):
    """Create a new promotion"""
    # Validate product exists
    product = (
        db.query(Product).filter(Product.product_id == promotion.product_id).first()
    )
    if not product:
        raise HTTPException(
            status_code=400, detail=f"Product {promotion.product_id} not found"
        )

    # Validate store exists
    store = db.query(Store).filter(Store.store_id == promotion.store_id).first()
    if not store:
        raise HTTPException(
            status_code=400, detail=f"Store {promotion.store_id} not found"
        )

    # Create promotion
    db_promotion = Promotion(**promotion.model_dump())
    db.add(db_promotion)
    db.commit()
    db.refresh(db_promotion)
    return db_promotion


@router.delete("/{promotion_id}", status_code=204)
def delete_promotion(promotion_id: int, db: Session = Depends(get_db)):
    """Delete a promotion"""
    promotion = db.query(Promotion).filter(Promotion.id == promotion_id).first()
    if not promotion:
        raise HTTPException(status_code=404, detail="Promotion not found")

    db.delete(promotion)
    db.commit()


# Campaign CRUD endpoints
@router.get("/campaigns/", response_model=list[CampaignResponse], tags=["campaigns"])
def list_campaigns(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: CampaignStatus | None = Query(
        None, description="Filter by campaign status"
    ),
    db: Session = Depends(get_db),
):
    """Get list of campaigns with optional filtering"""
    query = db.query(Campaign)

    if status:
        query = query.filter(Campaign.status == status.value)

    campaigns = (
        query.order_by(desc(Campaign.created_at)).offset(skip).limit(limit).all()
    )
    return campaigns


@router.get(
    "/campaigns/{campaign_id}", response_model=CampaignResponse, tags=["campaigns"]
)
def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Get a specific campaign by ID"""
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.post(
    "/campaigns/", response_model=CampaignResponse, status_code=201, tags=["campaigns"]
)
def create_campaign(campaign: CampaignCreate, db: Session = Depends(get_db)):
    """Create a new campaign"""
    # Validate budget is positive
    if campaign.budget <= 0:
        raise HTTPException(status_code=400, detail="Campaign budget must be positive")

    # Validate date range
    if campaign.end_date <= campaign.start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    db_campaign = Campaign(**campaign.model_dump())
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign


@router.put(
    "/campaigns/{campaign_id}", response_model=CampaignResponse, tags=["campaigns"]
)
def update_campaign(
    campaign_id: int, campaign_update: CampaignUpdate, db: Session = Depends(get_db)
):
    """Update an existing campaign"""
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Update only provided fields
    update_data = campaign_update.model_dump(exclude_unset=True)

    # Validate date range if both dates are being updated or set
    if update_data.get("end_date") and update_data.get("start_date"):
        if update_data["end_date"] <= update_data["start_date"]:
            raise HTTPException(
                status_code=400, detail="End date must be after start date"
            )
    elif update_data.get("end_date") and not update_data.get("start_date"):
        if update_data["end_date"] <= campaign.start_date:
            raise HTTPException(
                status_code=400, detail="End date must be after start date"
            )
    elif update_data.get("start_date") and not update_data.get("end_date"):
        if campaign.end_date <= update_data["start_date"]:
            raise HTTPException(
                status_code=400, detail="End date must be after start date"
            )

    for field, value in update_data.items():
        setattr(campaign, field, value)

    db.commit()
    db.refresh(campaign)
    return campaign


@router.delete("/campaigns/{campaign_id}", status_code=204, tags=["campaigns"])
def delete_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Delete a campaign"""
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Don't allow deletion of active campaigns
    if campaign.status == CampaignStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Cannot delete active campaigns")

    db.delete(campaign)
    db.commit()


# Utility endpoints
@router.get("/products/{product_id}/promotions", response_model=list[PromotionResponse])
def get_product_promotions(
    product_id: int,
    active_only: bool = Query(
        False, description="Return only currently active promotions"
    ),
    db: Session = Depends(get_db),
):
    """Get all promotions for a specific product"""
    # Validate product exists
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    query = db.query(Promotion).filter(Promotion.product_id == product_id)

    if active_only:
        # This is a simplified active check - in reality you'd compare against current day
        query = query.filter(Promotion.start_day <= 365, Promotion.end_day >= 1)

    promotions = query.order_by(desc(Promotion.start_day)).all()
    return promotions


@router.get("/stores/{store_id}/promotions", response_model=list[PromotionResponse])
def get_store_promotions(
    store_id: int,
    active_only: bool = Query(
        False, description="Return only currently active promotions"
    ),
    db: Session = Depends(get_db),
):
    """Get all promotions for a specific store"""
    # Validate store exists
    store = db.query(Store).filter(Store.store_id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    query = db.query(Promotion).filter(Promotion.store_id == store_id)

    if active_only:
        query = query.filter(Promotion.start_day <= 365, Promotion.end_day >= 1)

    promotions = query.order_by(desc(Promotion.start_day)).all()
    return promotions
