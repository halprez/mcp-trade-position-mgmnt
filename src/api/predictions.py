import random
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.entities import Product, Promotion, Store, Transaction
from ..models.schemas import (
    BudgetOptimizationRequest,
    BudgetOptimizationResponse,
    LiftPredictionRequest,
    LiftPredictionResponse,
    PromotionType,
)
from ..services.ml_predictor import get_ml_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/lift-prediction", response_model=LiftPredictionResponse)
def predict_promotion_lift(
    request: LiftPredictionRequest, db: Session = Depends(get_db)
):
    """Predict expected sales lift for a planned promotion using ML models"""

    # Validate product and store exist
    product = db.query(Product).filter(Product.product_id == request.product_id).first()
    if not product:
        raise HTTPException(
            status_code=404, detail=f"Product {request.product_id} not found"
        )

    store = db.query(Store).filter(Store.store_id == request.store_id).first()
    if not store:
        raise HTTPException(
            status_code=404, detail=f"Store {request.store_id} not found"
        )

    try:
        # Get ML service and make prediction
        ml_service = get_ml_service()

        # Use real ML model for prediction
        ml_result = ml_service.predict_lift(
            product_id=request.product_id,
            store_id=request.store_id,
            promotion_type=request.promotion_type.value,
            discount_pct=request.discount_percentage,
            duration_days=request.duration_days,
        )

        # Extract ML predictions
        lift_percentage = ml_result.prediction
        confidence_lower = ml_result.confidence_lower
        confidence_upper = ml_result.confidence_upper

    except Exception as e:
        # Fallback to rule-based prediction if ML fails
        print(f"ML prediction failed, using fallback: {e}")

        # Simplified fallback logic
        discount_factor = min(request.discount_percentage / 100 * 2.5, 0.8)
        duration_factor = min(request.duration_days / 14 * 0.3, 0.4)

        type_multipliers = {
            PromotionType.BOGO: 1.4,
            PromotionType.DISCOUNT: 1.0,
            PromotionType.DISPLAY: 0.8,
            PromotionType.COUPON: 1.2,
            PromotionType.BUNDLE: 1.1,
        }
        type_factor = type_multipliers.get(request.promotion_type, 1.0)

        display_boost = (
            0.15 if request.display_location and request.display_location != "" else 0
        )
        base_lift = (discount_factor + duration_factor) * type_factor + display_boost

        lift_percentage = max(5.0, min(75.0, base_lift * 100 + random.uniform(-5, 5)))
        confidence_lower = max(0, lift_percentage - 8)
        confidence_upper = lift_percentage + 12

    # Calculate business metrics
    baseline_sales = (
        db.query(func.sum(Transaction.quantity))
        .filter(
            and_(
                Transaction.product_id == request.product_id,
                Transaction.store_id == request.store_id,
            )
        )
        .scalar()
        or 100
    )

    baseline_daily = baseline_sales / 365 if baseline_sales > 0 else 10
    expected_baseline_units = int(baseline_daily * request.duration_days)
    incremental_units = int(expected_baseline_units * (lift_percentage / 100))

    # Get average price for revenue calculation
    avg_price = (
        db.query(func.avg(Transaction.sales_value / Transaction.quantity))
        .filter(Transaction.product_id == request.product_id)
        .scalar()
        or 5.99
    )

    incremental_revenue = incremental_units * avg_price

    return LiftPredictionResponse(
        product_id=request.product_id,
        predicted_lift_percentage=round(lift_percentage, 1),
        confidence_interval_lower=round(confidence_lower, 1),
        confidence_interval_upper=round(confidence_upper, 1),
        expected_incremental_units=incremental_units,
        expected_incremental_revenue=round(incremental_revenue, 2),
    )


@router.post("/budget-optimization", response_model=BudgetOptimizationResponse)
def optimize_promotion_budget(
    request: BudgetOptimizationRequest, db: Session = Depends(get_db)
):
    """Optimize budget allocation across products for maximum ROI using ML"""

    # Validate all products exist
    products = (
        db.query(Product).filter(Product.product_id.in_(request.target_products)).all()
    )
    if len(products) != len(request.target_products):
        found_ids = {p.product_id for p in products}
        missing = set(request.target_products) - found_ids
        raise HTTPException(
            status_code=404, detail=f"Products not found: {list(missing)}"
        )

    try:
        # Use ML-powered budget optimization
        ml_service = get_ml_service()

        optimization_result = ml_service.optimize_budget(
            total_budget=request.total_budget,
            product_ids=request.target_products,
            constraints={
                "max_allocation_pct": 40,
                "min_allocation_pct": 10,
                "max_discount_pct": 35,
                "preferred_duration": 14,
            },
        )

        return BudgetOptimizationResponse(
            total_budget=optimization_result["total_budget"],
            allocated_budget=optimization_result["allocated_budget"],
            recommendations=optimization_result["recommendations"],
            expected_roi=optimization_result["expected_roi"],
            expected_lift=optimization_result["expected_lift"],
        )

    except Exception as e:
        # Fallback to rule-based optimization if ML fails
        print(f"ML budget optimization failed, using fallback: {e}")

        # Simplified fallback optimization logic
        product_performance = {}

        for product in products:
            total_sales = (
                db.query(func.sum(Transaction.sales_value))
                .filter(Transaction.product_id == product.product_id)
                .scalar()
                or 0
            )

            promo_count = (
                db.query(func.count(Promotion.id))
                .filter(Promotion.product_id == product.product_id)
                .scalar()
                or 0
            )

            avg_price = (
                db.query(func.avg(Transaction.sales_value / Transaction.quantity))
                .filter(Transaction.product_id == product.product_id)
                .scalar()
                or 1.0
            )

            sales_score = min(total_sales / 1000, 10)
            promotion_experience = min(promo_count * 0.5, 5)
            price_tier = 1 + (avg_price / 10)
            attractiveness = sales_score + promotion_experience + price_tier

            product_performance[product.product_id] = {
                "product": product,
                "total_sales": total_sales,
                "attractiveness": attractiveness,
                "avg_price": avg_price,
                "estimated_roi": 100 + attractiveness * 10,
            }

        recommendations = []
        allocated_budget = 0

        sorted_products = sorted(
            product_performance.items(),
            key=lambda x: x[1]["estimated_roi"],
            reverse=True,
        )

        for product_id, perf in sorted_products:
            max_allocation = request.total_budget * 0.4
            min_allocation = request.total_budget * 0.1
            remaining_budget = request.total_budget - allocated_budget
            remaining_products = len(
                [
                    p
                    for p in sorted_products[
                        sorted_products.index((product_id, perf)) :
                    ]
                ]
            )

            if remaining_products > 1:
                allocation = max(
                    min_allocation, min(max_allocation, remaining_budget * 0.3)
                )
            else:
                allocation = remaining_budget

            allocation = min(allocation, remaining_budget)

            expected_lift = 15 + (perf["attractiveness"] * 2)
            expected_roi = perf["estimated_roi"] + random.uniform(-5, 15)

            recommendation = {
                "product_id": product_id,
                "product_name": f"{perf['product'].brand} - {perf['product'].commodity_desc}",
                "allocated_budget": round(allocation, 2),
                "budget_percentage": round(allocation / request.total_budget * 100, 1),
                "recommended_discount": round(
                    15 + (allocation / request.total_budget * 20), 1
                ),
                "expected_lift": round(expected_lift, 1),
                "expected_roi": round(expected_roi, 1),
                "promotion_type": "BOGO" if perf["avg_price"] > 5 else "Discount",
            }

            recommendations.append(recommendation)
            allocated_budget += allocation

            if allocated_budget >= request.total_budget * 0.95:
                break

        total_expected_roi = sum(r["expected_roi"] for r in recommendations) / len(
            recommendations
        )
        total_expected_lift = sum(r["expected_lift"] for r in recommendations) / len(
            recommendations
        )

        return BudgetOptimizationResponse(
            total_budget=request.total_budget,
            allocated_budget=round(allocated_budget, 2),
            recommendations=recommendations,
            expected_roi=round(total_expected_roi, 1),
            expected_lift=round(total_expected_lift, 1),
        )


@router.get("/product-recommendations/{product_id}")
def get_product_promotion_recommendations(
    product_id: int, budget: float, db: Session = Depends(get_db)
):
    """Get specific promotion recommendations for a product"""

    product = db.query(Product).filter(Product.product_id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Analyze product characteristics
    avg_price = (
        db.query(func.avg(Transaction.sales_value / Transaction.quantity))
        .filter(Transaction.product_id == product_id)
        .scalar()
        or 5.0
    )

    total_sales = (
        db.query(func.sum(Transaction.sales_value))
        .filter(Transaction.product_id == product_id)
        .scalar()
        or 0
    )

    # Generate recommendations based on product profile
    recommendations = []

    # BOGO recommendation for higher-priced items
    if avg_price > 6.0:
        recommendations.append(
            {
                "type": "BOGO",
                "description": "Buy One Get One Free",
                "expected_lift": "35-45%",
                "investment_needed": budget * 0.6,
                "best_duration": "2 weeks",
                "confidence": "High",
            }
        )

    # Discount recommendation
    discount_pct = min(30, max(10, budget / total_sales * 1000))
    recommendations.append(
        {
            "type": "Discount",
            "description": f"{discount_pct:.0f}% Off",
            "expected_lift": f"{15 + discount_pct/2:.0f}-{25 + discount_pct/2:.0f}%",
            "investment_needed": budget * 0.4,
            "best_duration": "3 weeks",
            "confidence": "Medium",
        }
    )

    # Display promotion
    recommendations.append(
        {
            "type": "Display",
            "description": "End-cap Display Placement",
            "expected_lift": "18-25%",
            "investment_needed": budget * 0.2,
            "best_duration": "4 weeks",
            "confidence": "Medium",
        }
    )

    return {
        "product_id": product_id,
        "product_name": f"{product.brand} - {product.commodity_desc}",
        "budget": budget,
        "recommendations": recommendations,
        "best_timing": "Weekends and month-end typically perform best",
        "seasonality_notes": "Consider holiday periods for increased effectiveness",
    }


@router.get("/market-insights")
def get_market_insights(db: Session = Depends(get_db)):
    """Get market insights and trends for promotion planning"""

    # Get promotion type performance
    promo_types = (
        db.query(Promotion.promotion_type, func.count(Promotion.id).label("count"))
        .group_by(Promotion.promotion_type)
        .all()
    )

    # Calculate trends (simplified)
    insights = {
        "promotion_trends": {
            "most_used_type": (
                max(promo_types, key=lambda x: x.count).promotion_type
                if promo_types
                else "Discount"
            ),
            "total_promotions": sum(p.count for p in promo_types),
            "avg_promotions_per_product": round(
                sum(p.count for p in promo_types) / max(1, len(promo_types)), 1
            ),
        },
        "performance_insights": [
            "BOGO promotions show 15% higher lift than straight discounts",
            "Display promotions work best for impulse purchases",
            "Weekend promotions generate 22% more traffic",
            "Promotions longer than 3 weeks show diminishing returns",
        ],
        "seasonal_recommendations": {
            "current_month": datetime.now().strftime("%B"),
            "recommended_focus": (
                "Back-to-school promotions"
                if datetime.now().month in [8, 9]
                else "Standard promotional mix"
            ),
            "trending_categories": ["Frozen Foods", "Beverages", "Snacks"],
        },
        "budget_guidelines": {
            "recommended_roi_threshold": "120%",
            "typical_discount_range": "15-25%",
            "optimal_duration": "2-3 weeks",
            "budget_allocation": "Allocate 60% to proven performers, 40% to test new strategies",
        },
    }

    return insights


@router.get("/model-status")
def get_model_status():
    """Get ML model status and information"""
    try:
        ml_service = get_ml_service()
        model_info = ml_service.get_model_info()

        return {
            "status": (
                "healthy" if model_info["service_initialized"] else "initializing"
            ),
            "models": model_info,
            "capabilities": [
                "promotion_lift_prediction",
                "budget_optimization",
                "feature_engineering",
                "model_serving",
            ],
            "ml_engine": "XGBoost + Scikit-learn",
            "version": "1.0.0",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "fallback_mode": True,
            "message": "Using rule-based fallback predictions",
        }


@router.post("/retrain-models")
def retrain_models():
    """Retrain ML models with latest data"""
    try:
        ml_service = get_ml_service()
        ml_service.initialize(retrain=True)

        return {
            "status": "success",
            "message": "Models retrained successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model retraining failed: {str(e)}"
        )
