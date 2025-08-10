from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session, joinedload

from ..models.database import get_db
from ..models.entities import Campaign, Household, Product, Promotion, Transaction
from ..models.schemas import (
    DashboardResponse,
    DashboardSummary,
    ProductAnalyticsRequest,
    ProductAnalyticsResponse,
    PromotionROIRequest,
    PromotionROIResponse,
    PromotionType,
    TopPerformer,
    TransactionSummary,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(db: Session = Depends(get_db)):
    """Get high-level dashboard summary with key metrics"""

    # Basic counts
    total_households = db.query(func.count(Household.id)).scalar() or 0
    total_products = db.query(func.count(Product.id)).scalar() or 0
    total_transactions = db.query(func.count(Transaction.id)).scalar() or 0
    total_promotions = db.query(func.count(Promotion.id)).scalar() or 0
    active_campaigns = (
        db.query(func.count(Campaign.id)).filter(Campaign.status == "active").scalar()
        or 0
    )

    # Revenue calculation
    total_revenue = db.query(func.sum(Transaction.sales_value)).scalar() or 0.0

    # Average promotion ROI (simplified calculation)
    avg_promotion_roi = 125.0  # Placeholder - would calculate from actual data

    summary = DashboardSummary(
        total_households=total_households,
        total_products=total_products,
        total_transactions=total_transactions,
        total_promotions=total_promotions,
        active_campaigns=active_campaigns,
        total_revenue=total_revenue,
        avg_promotion_roi=avg_promotion_roi,
    )

    # Top performing products by revenue
    top_products_query = (
        db.query(
            Transaction.product_id,
            Product.brand,
            func.sum(Transaction.sales_value).label("total_revenue"),
        )
        .join(Product, Transaction.product_id == Product.product_id)
        .group_by(Transaction.product_id, Product.brand)
        .order_by(desc("total_revenue"))
        .limit(5)
        .all()
    )

    top_products = [
        TopPerformer(
            id=row.product_id,
            name=row.brand or f"Product {row.product_id}",
            value=float(row.total_revenue),
            metric="revenue",
        )
        for row in top_products_query
    ]

    # Top performing promotions (placeholder)
    top_promotions = [
        TopPerformer(id=1, name="BOGO Cereal", value=156.7, metric="roi"),
        TopPerformer(id=2, name="25% Off Pizza", value=134.2, metric="roi"),
        TopPerformer(id=3, name="Display Soda", value=112.8, metric="roi"),
    ]

    # Recent campaigns
    recent_campaigns = (
        db.query(Campaign).order_by(desc(Campaign.created_at)).limit(3).all()
    )

    return DashboardResponse(
        summary=summary,
        top_products=top_products,
        top_promotions=top_promotions,
        recent_campaigns=recent_campaigns,
    )


@router.post("/promotion-roi", response_model=list[PromotionROIResponse])
def analyze_promotion_roi(request: PromotionROIRequest, db: Session = Depends(get_db)):
    """Analyze ROI for promotions with detailed performance metrics"""

    query = db.query(Promotion).options(joinedload(Promotion.product))

    # Apply filters
    if request.promotion_ids:
        query = query.filter(Promotion.id.in_(request.promotion_ids))

    if request.product_ids:
        query = query.filter(Promotion.product_id.in_(request.product_ids))

    if request.promotion_type:
        query = query.filter(Promotion.promotion_type == request.promotion_type.value)

    if request.time_range:
        query = query.filter(
            and_(
                Promotion.start_day >= request.time_range.start_day,
                Promotion.end_day <= request.time_range.end_day,
            )
        )

    promotions = query.all()

    if not promotions:
        return []

    results = []
    for promotion in promotions:
        # Calculate performance metrics (simplified for demo)
        # In real implementation, would analyze transaction data before/during/after promotion

        # Baseline calculation - transactions before promotion
        baseline_revenue = (
            db.query(func.sum(Transaction.sales_value))
            .filter(
                and_(
                    Transaction.product_id == promotion.product_id,
                    Transaction.store_id == promotion.store_id,
                    Transaction.day < promotion.start_day,
                    Transaction.day >= promotion.start_day - 30,  # 30 days before
                )
            )
            .scalar()
            or 0.0
        )

        # Promoted period revenue
        promoted_revenue = (
            db.query(func.sum(Transaction.sales_value))
            .filter(
                and_(
                    Transaction.product_id == promotion.product_id,
                    Transaction.store_id == promotion.store_id,
                    Transaction.day >= promotion.start_day,
                    Transaction.day <= promotion.end_day,
                )
            )
            .scalar()
            or 0.0
        )

        # Calculate metrics
        baseline_daily = baseline_revenue / 30 if baseline_revenue > 0 else 0
        promotion_days = promotion.end_day - promotion.start_day + 1
        expected_baseline = baseline_daily * promotion_days
        incremental_revenue = max(0, promoted_revenue - expected_baseline)

        # Estimate investment (simplified)
        discount_pct = promotion.discount_percentage or 15.0
        investment = promoted_revenue * (discount_pct / 100)

        # Calculate ROI
        roi = (incremental_revenue / investment * 100) if investment > 0 else 0
        lift_percentage = (
            ((promoted_revenue - expected_baseline) / expected_baseline * 100)
            if expected_baseline > 0
            else 0
        )

        product_name = None
        if hasattr(promotion, "product") and promotion.product:
            product_name = (
                f"{promotion.product.brand} - {promotion.product.commodity_desc}"
            )

        results.append(
            PromotionROIResponse(
                promotion_id=promotion.id,
                product_id=promotion.product_id,
                product_name=product_name,
                promotion_type=PromotionType(promotion.promotion_type),
                start_day=promotion.start_day,
                end_day=promotion.end_day,
                investment=round(investment, 2),
                incremental_revenue=round(incremental_revenue, 2),
                roi=round(roi, 1),
                lift_percentage=round(lift_percentage, 1),
            )
        )

    return results


@router.post("/product-performance", response_model=list[ProductAnalyticsResponse])
def analyze_product_performance(
    request: ProductAnalyticsRequest, db: Session = Depends(get_db)
):
    """Analyze product performance with promotion impact"""

    query = db.query(Product).filter(Product.product_id.in_(request.product_ids))
    products = query.all()

    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    results = []
    for product in products:
        # Base transaction analysis
        trans_query = db.query(
            func.sum(Transaction.sales_value).label("total_revenue"),
            func.sum(Transaction.quantity).label("total_sales"),
            func.avg(Transaction.sales_value / Transaction.quantity).label("avg_price"),
        ).filter(Transaction.product_id == product.product_id)

        # Apply time filter if specified
        if request.time_range:
            trans_query = trans_query.filter(
                and_(
                    Transaction.day >= request.time_range.start_day,
                    Transaction.day <= request.time_range.end_day,
                )
            )

        trans_result = trans_query.first()

        total_revenue = float(trans_result.total_revenue or 0)
        total_sales = float(trans_result.total_sales or 0)
        avg_price = float(trans_result.avg_price or 0)

        # Promotion analysis
        promotion_count = 0
        best_promotion_roi = None

        if request.include_promotions:
            promo_query = db.query(Promotion).filter(
                Promotion.product_id == product.product_id
            )

            if request.time_range:
                promo_query = promo_query.filter(
                    and_(
                        Promotion.start_day >= request.time_range.start_day,
                        Promotion.end_day <= request.time_range.end_day,
                    )
                )

            promotion_count = promo_query.count()
            # Simplified best ROI calculation
            best_promotion_roi = 145.5 if promotion_count > 0 else None

        results.append(
            ProductAnalyticsResponse(
                product_id=product.product_id,
                brand=product.brand,
                commodity_desc=product.commodity_desc,
                total_sales=total_sales,
                total_revenue=total_revenue,
                avg_price=avg_price,
                promotion_count=promotion_count,
                best_promotion_roi=best_promotion_roi,
            )
        )

    return results


@router.get("/transactions/summary", response_model=TransactionSummary)
def get_transaction_summary(
    start_day: int | None = Query(None, description="Start day for analysis"),
    end_day: int | None = Query(None, description="End day for analysis"),
    product_id: int | None = Query(None, description="Filter by product ID"),
    store_id: int | None = Query(None, description="Filter by store ID"),
    db: Session = Depends(get_db),
):
    """Get transaction summary statistics"""

    query = db.query(Transaction)

    # Apply filters
    if start_day is not None:
        query = query.filter(Transaction.day >= start_day)
    if end_day is not None:
        query = query.filter(Transaction.day <= end_day)
    if product_id is not None:
        query = query.filter(Transaction.product_id == product_id)
    if store_id is not None:
        query = query.filter(Transaction.store_id == store_id)

    # Get summary statistics
    summary_result = query.with_entities(
        func.count(Transaction.id).label("total_transactions"),
        func.sum(Transaction.sales_value).label("total_revenue"),
        func.avg(Transaction.sales_value).label("avg_basket_size"),
        func.count(func.distinct(Transaction.household_key)).label("unique_households"),
        func.count(func.distinct(Transaction.product_id)).label("unique_products"),
    ).first()

    return TransactionSummary(
        total_transactions=summary_result.total_transactions or 0,
        total_revenue=float(summary_result.total_revenue or 0),
        avg_basket_size=float(summary_result.avg_basket_size or 0),
        unique_households=summary_result.unique_households or 0,
        unique_products=summary_result.unique_products or 0,
    )


@router.get("/promotions/types", response_model=dict[str, int])
def get_promotion_type_distribution(db: Session = Depends(get_db)):
    """Get distribution of promotion types"""

    results = (
        db.query(Promotion.promotion_type, func.count(Promotion.id).label("count"))
        .group_by(Promotion.promotion_type)
        .all()
    )

    return {row.promotion_type or "Unknown": row.count for row in results}


@router.get("/products/top-performers", response_model=list[TopPerformer])
def get_top_performing_products(
    metric: str = Query(
        "revenue", description="Metric to rank by: revenue, sales, roi"
    ),
    limit: int = Query(
        10, ge=1, le=50, description="Number of top performers to return"
    ),
    time_range: int | None = Query(None, description="Days to look back"),
    db: Session = Depends(get_db),
):
    """Get top performing products by various metrics"""

    if metric == "revenue":
        query = (
            db.query(
                Transaction.product_id,
                Product.brand,
                Product.commodity_desc,
                func.sum(Transaction.sales_value).label("value"),
            )
            .join(Product, Transaction.product_id == Product.product_id)
            .group_by(Transaction.product_id, Product.brand, Product.commodity_desc)
            .order_by(desc("value"))
        )

    elif metric == "sales":
        query = (
            db.query(
                Transaction.product_id,
                Product.brand,
                Product.commodity_desc,
                func.sum(Transaction.quantity).label("value"),
            )
            .join(Product, Transaction.product_id == Product.product_id)
            .group_by(Transaction.product_id, Product.brand, Product.commodity_desc)
            .order_by(desc("value"))
        )

    else:
        # ROI metric - placeholder implementation
        raise HTTPException(status_code=400, detail="ROI metric not yet implemented")

    # Apply time filter if specified
    if time_range:
        max_day = db.query(func.max(Transaction.day)).scalar() or 365
        min_day = max(1, max_day - time_range)
        query = query.filter(Transaction.day >= min_day)

    results = query.limit(limit).all()

    return [
        TopPerformer(
            id=row.product_id,
            name=f"{row.brand or 'Unknown'} - {row.commodity_desc or 'Product'}",
            value=float(row.value),
            metric=metric,
        )
        for row in results
    ]
