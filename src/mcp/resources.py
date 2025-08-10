"""
MCP Resources - Structured data access for Claude Desktop
Provides real-time access to TPM data through resource URIs
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastmcp import FastMCP
from ..models.database import get_db_session
from ..models.entities import (
    Product,
    Promotion,
    Transaction,
    Campaign,
    Store,
    Household,
)
from sqlalchemy import func, desc, and_
from sqlalchemy.orm import Session


def register_data_resources(mcp: FastMCP) -> None:
    """Register data access resources with the FastMCP server"""

    @mcp.resource("tpm://products/{category}")
    def get_products_by_category(category: str) -> str:
        """Get all products in a specific category with performance metrics"""
        try:
            with get_db_session() as db:
                products = (
                    db.query(
                        Product.product_id,
                        Product.brand,
                        Product.commodity_desc,
                        Product.manufacturer,
                        func.sum(Transaction.sales_value).label("total_revenue"),
                        func.avg(Transaction.sales_value / Transaction.quantity).label(
                            "avg_price"
                        ),
                        func.count(Transaction.id).label("transaction_count"),
                    )
                    .join(Transaction, Product.product_id == Transaction.product_id)
                    .filter(
                        Product.commodity_desc.ilike(f"%{category}%")
                        | Product.department.ilike(f"%{category}%")
                    )
                    .group_by(
                        Product.product_id,
                        Product.brand,
                        Product.commodity_desc,
                        Product.manufacturer,
                    )
                    .order_by(desc("total_revenue"))
                    .limit(20)
                    .all()
                )

                if not products:
                    return f"No products found in category: {category}"

                result = {
                    "category": category,
                    "product_count": len(products),
                    "total_category_revenue": sum(
                        p.total_revenue or 0 for p in products
                    ),
                    "products": [],
                }

                for product in products:
                    result["products"].append(
                        {
                            "product_id": product.product_id,
                            "brand": product.brand,
                            "description": product.commodity_desc,
                            "manufacturer": product.manufacturer,
                            "total_revenue": float(product.total_revenue or 0),
                            "avg_price": float(product.avg_price or 0),
                            "transaction_count": product.transaction_count,
                            "market_share_pct": (
                                round(
                                    (product.total_revenue or 0)
                                    / result["total_category_revenue"]
                                    * 100,
                                    2,
                                )
                                if result["total_category_revenue"] > 0
                                else 0
                            ),
                        }
                    )

                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error fetching products for category {category}: {str(e)}"

    @mcp.resource("tpm://promotions/{time_period}")
    def get_promotions_by_period(time_period: str) -> str:
        """Get promotion performance data for a specific time period

        time_period options: "current", "last_30_days", "last_quarter", "all_time"
        """
        try:
            with get_db_session() as db:
                base_query = db.query(
                    Promotion.id,
                    Promotion.product_id,
                    Promotion.promotion_type,
                    Promotion.discount_percentage,
                    Promotion.duration_days,
                    Promotion.start_day,
                    Promotion.end_day,
                    Product.brand,
                    Product.commodity_desc,
                ).join(Product, Promotion.product_id == Product.product_id)

                # Apply time filtering based on period
                if time_period == "current":
                    # Active promotions
                    current_day = 365  # Simplified current day
                    promotions = base_query.filter(
                        and_(
                            Promotion.start_day <= current_day,
                            Promotion.end_day >= current_day,
                        )
                    ).all()
                elif time_period == "last_30_days":
                    current_day = 365
                    promotions = (
                        base_query.filter(Promotion.end_day >= current_day - 30)
                        .order_by(desc(Promotion.start_day))
                        .limit(50)
                        .all()
                    )
                elif time_period == "last_quarter":
                    current_day = 365
                    promotions = (
                        base_query.filter(Promotion.end_day >= current_day - 90)
                        .order_by(desc(Promotion.start_day))
                        .limit(100)
                        .all()
                    )
                else:  # all_time
                    promotions = (
                        base_query.order_by(desc(Promotion.start_day)).limit(200).all()
                    )

                if not promotions:
                    return f"No promotions found for period: {time_period}"

                # Calculate summary statistics
                total_promotions = len(promotions)
                avg_discount = (
                    sum(p.discount_percentage for p in promotions) / total_promotions
                )
                avg_duration = (
                    sum(p.duration_days for p in promotions) / total_promotions
                )

                # Promotion type distribution
                type_counts = {}
                for promo in promotions:
                    type_counts[promo.promotion_type] = (
                        type_counts.get(promo.promotion_type, 0) + 1
                    )

                result = {
                    "time_period": time_period,
                    "summary": {
                        "total_promotions": total_promotions,
                        "avg_discount_pct": round(avg_discount, 1),
                        "avg_duration_days": round(avg_duration, 1),
                        "promotion_types": type_counts,
                    },
                    "promotions": [],
                }

                for promo in promotions:
                    result["promotions"].append(
                        {
                            "promotion_id": promo.id,
                            "product_id": promo.product_id,
                            "product_name": f"{promo.brand} - {promo.commodity_desc}",
                            "type": promo.promotion_type,
                            "discount_pct": promo.discount_percentage,
                            "duration_days": promo.duration_days,
                            "start_day": promo.start_day,
                            "end_day": promo.end_day,
                            "status": (
                                "active"
                                if promo.start_day <= 365 <= promo.end_day
                                else "completed"
                            ),
                        }
                    )

                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error fetching promotions for period {time_period}: {str(e)}"

    @mcp.resource("tpm://analytics/top-performers/{metric}")
    def get_top_performers(metric: str) -> str:
        """Get top performing products by various metrics

        metric options: "revenue", "units", "profit_margin", "promotion_lift"
        """
        try:
            with get_db_session() as db:
                if metric == "revenue":
                    performers = (
                        db.query(
                            Product.product_id,
                            Product.brand,
                            Product.commodity_desc,
                            func.sum(Transaction.sales_value).label("total_revenue"),
                            func.sum(Transaction.quantity).label("total_units"),
                            func.avg(
                                Transaction.sales_value / Transaction.quantity
                            ).label("avg_price"),
                        )
                        .join(Transaction)
                        .group_by(
                            Product.product_id, Product.brand, Product.commodity_desc
                        )
                        .order_by(desc("total_revenue"))
                        .limit(15)
                        .all()
                    )

                elif metric == "units":
                    performers = (
                        db.query(
                            Product.product_id,
                            Product.brand,
                            Product.commodity_desc,
                            func.sum(Transaction.quantity).label("total_units"),
                            func.sum(Transaction.sales_value).label("total_revenue"),
                            func.avg(
                                Transaction.sales_value / Transaction.quantity
                            ).label("avg_price"),
                        )
                        .join(Transaction)
                        .group_by(
                            Product.product_id, Product.brand, Product.commodity_desc
                        )
                        .order_by(desc("total_units"))
                        .limit(15)
                        .all()
                    )

                elif metric == "profit_margin":
                    # Simplified profit margin calculation
                    performers = (
                        db.query(
                            Product.product_id,
                            Product.brand,
                            Product.commodity_desc,
                            func.avg(
                                Transaction.sales_value / Transaction.quantity
                            ).label("avg_price"),
                            func.sum(Transaction.sales_value).label("total_revenue"),
                            func.sum(Transaction.quantity).label("total_units"),
                        )
                        .join(Transaction)
                        .group_by(
                            Product.product_id, Product.brand, Product.commodity_desc
                        )
                        .order_by(desc("avg_price"))
                        .limit(15)
                        .all()
                    )

                else:  # promotion_lift
                    # Products with most promotions (proxy for lift performance)
                    performers = (
                        db.query(
                            Product.product_id,
                            Product.brand,
                            Product.commodity_desc,
                            func.count(Promotion.id).label("promotion_count"),
                            func.avg(Promotion.discount_percentage).label(
                                "avg_discount"
                            ),
                            func.sum(Transaction.sales_value).label("total_revenue"),
                        )
                        .join(Promotion, Product.product_id == Promotion.product_id)
                        .join(Transaction, Product.product_id == Transaction.product_id)
                        .group_by(
                            Product.product_id, Product.brand, Product.commodity_desc
                        )
                        .order_by(desc("promotion_count"))
                        .limit(15)
                        .all()
                    )

                if not performers:
                    return f"No performance data found for metric: {metric}"

                result = {
                    "metric": metric,
                    "analysis_date": datetime.now().isoformat(),
                    "top_performers": [],
                }

                for i, performer in enumerate(performers, 1):
                    perf_data = {
                        "rank": i,
                        "product_id": performer.product_id,
                        "product_name": f"{performer.brand} - {performer.commodity_desc}",
                        "brand": performer.brand,
                    }

                    if metric == "revenue":
                        perf_data.update(
                            {
                                "total_revenue": float(performer.total_revenue),
                                "total_units": performer.total_units,
                                "avg_price": float(performer.avg_price),
                            }
                        )
                    elif metric == "units":
                        perf_data.update(
                            {
                                "total_units": performer.total_units,
                                "total_revenue": float(performer.total_revenue),
                                "avg_price": float(performer.avg_price),
                            }
                        )
                    elif metric == "profit_margin":
                        perf_data.update(
                            {
                                "avg_price": float(performer.avg_price),
                                "total_revenue": float(performer.total_revenue),
                                "total_units": performer.total_units,
                                "estimated_margin_pct": min(
                                    50, max(10, performer.avg_price * 15)
                                ),  # Simplified estimate
                            }
                        )
                    else:  # promotion_lift
                        perf_data.update(
                            {
                                "promotion_count": performer.promotion_count,
                                "avg_discount": float(performer.avg_discount),
                                "total_revenue": float(performer.total_revenue),
                            }
                        )

                    result["top_performers"].append(perf_data)

                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error fetching top performers for {metric}: {str(e)}"

    @mcp.resource("tpm://market-data/category-analysis/{category}")
    def get_category_analysis(category: str) -> str:
        """Comprehensive category analysis including competition and trends"""
        try:
            with get_db_session() as db:
                # Get all products in category
                products = (
                    db.query(Product)
                    .filter(
                        Product.commodity_desc.ilike(f"%{category}%")
                        | Product.department.ilike(f"%{category}%")
                    )
                    .all()
                )

                if not products:
                    return f"Category '{category}' not found"

                product_ids = [p.product_id for p in products]

                # Revenue and volume analysis
                category_stats = (
                    db.query(
                        func.sum(Transaction.sales_value).label("total_revenue"),
                        func.sum(Transaction.quantity).label("total_units"),
                        func.avg(Transaction.sales_value / Transaction.quantity).label(
                            "avg_price"
                        ),
                        func.count(func.distinct(Transaction.household_key)).label(
                            "unique_customers"
                        ),
                    )
                    .filter(Transaction.product_id.in_(product_ids))
                    .first()
                )

                # Promotion analysis
                promo_stats = (
                    db.query(
                        func.count(Promotion.id).label("total_promotions"),
                        func.avg(Promotion.discount_percentage).label("avg_discount"),
                        func.avg(Promotion.duration_days).label("avg_duration"),
                    )
                    .filter(Promotion.product_id.in_(product_ids))
                    .first()
                )

                # Brand competition
                brand_performance = (
                    db.query(
                        Product.brand,
                        func.sum(Transaction.sales_value).label("brand_revenue"),
                        func.count(func.distinct(Product.product_id)).label(
                            "product_count"
                        ),
                    )
                    .join(Transaction)
                    .filter(Product.product_id.in_(product_ids))
                    .group_by(Product.brand)
                    .order_by(desc("brand_revenue"))
                    .limit(10)
                    .all()
                )

                # Market concentration (HHI calculation)
                total_category_revenue = float(category_stats.total_revenue or 1)
                brand_shares = [
                    (b.brand_revenue / total_category_revenue * 100)
                    for b in brand_performance
                ]
                hhi = sum(share**2 for share in brand_shares) if brand_shares else 0

                result = {
                    "category": category,
                    "analysis_date": datetime.now().isoformat(),
                    "market_overview": {
                        "total_products": len(products),
                        "total_revenue": float(category_stats.total_revenue or 0),
                        "total_units": category_stats.total_units or 0,
                        "avg_price": float(category_stats.avg_price or 0),
                        "unique_customers": category_stats.unique_customers or 0,
                        "market_concentration_hhi": round(hhi, 1),
                    },
                    "promotion_activity": {
                        "total_promotions": promo_stats.total_promotions or 0,
                        "avg_discount_pct": float(promo_stats.avg_discount or 0),
                        "avg_duration_days": float(promo_stats.avg_duration or 0),
                        "promotion_intensity": round(
                            (promo_stats.total_promotions or 0) / len(products), 2
                        ),
                    },
                    "competitive_landscape": {
                        "total_brands": len(brand_performance),
                        "market_leader": (
                            brand_performance[0].brand if brand_performance else "N/A"
                        ),
                        "market_leader_share_pct": (
                            round(brand_shares[0], 1) if brand_shares else 0
                        ),
                        "top_brands": [
                            {
                                "brand": b.brand,
                                "revenue": float(b.brand_revenue),
                                "market_share_pct": round(
                                    b.brand_revenue / total_category_revenue * 100, 1
                                ),
                                "product_count": b.product_count,
                            }
                            for b in brand_performance[:5]
                        ],
                    },
                    "insights": {
                        "market_maturity": (
                            "Mature"
                            if hhi > 2500
                            else "Fragmented" if hhi < 1000 else "Competitive"
                        ),
                        "promotional_intensity": (
                            "High"
                            if (promo_stats.total_promotions or 0) > len(products) * 0.5
                            else "Moderate"
                        ),
                        "pricing_strategy": (
                            "Premium"
                            if (category_stats.avg_price or 0) > 8
                            else (
                                "Value"
                                if (category_stats.avg_price or 0) < 3
                                else "Mid-tier"
                            )
                        ),
                    },
                }

                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error analyzing category {category}: {str(e)}"

    @mcp.resource("tpm://campaigns/{campaign_id}/detailed-metrics")
    def get_detailed_campaign_metrics(campaign_id: str) -> str:
        """Get comprehensive campaign performance metrics and analysis"""
        try:
            with get_db_session() as db:
                try:
                    campaign_id_int = int(campaign_id)
                except ValueError:
                    return f"Invalid campaign ID: {campaign_id}"

                campaign = (
                    db.query(Campaign).filter(Campaign.id == campaign_id_int).first()
                )
                if not campaign:
                    return f"Campaign {campaign_id} not found"

                # Get related promotions
                promotions = (
                    db.query(Promotion)
                    .filter(Promotion.id == campaign_id_int)  # Simplified relationship
                    .all()
                )

                result = {
                    "campaign_id": campaign.id,
                    "campaign_name": campaign.name,
                    "analysis_date": datetime.now().isoformat(),
                    "basic_metrics": {
                        "status": campaign.status,
                        "budget": float(campaign.budget),
                        "actual_spend": float(campaign.actual_spend),
                        "spend_efficiency_pct": (
                            round(campaign.actual_spend / campaign.budget * 100, 1)
                            if campaign.budget > 0
                            else 0
                        ),
                        "roi_pct": campaign.roi,
                        "revenue_lift_pct": campaign.revenue_lift,
                    },
                    "timeline": {
                        "start_date": (
                            campaign.start_date.isoformat()
                            if campaign.start_date
                            else None
                        ),
                        "end_date": (
                            campaign.end_date.isoformat() if campaign.end_date else None
                        ),
                        "duration_days": (
                            (campaign.end_date - campaign.start_date).days
                            if campaign.start_date and campaign.end_date
                            else None
                        ),
                        "created_at": (
                            campaign.created_at.isoformat()
                            if campaign.created_at
                            else None
                        ),
                    },
                    "performance_assessment": {
                        "roi_grade": (
                            "A"
                            if (campaign.roi or 0) > 150
                            else (
                                "B"
                                if (campaign.roi or 0) > 120
                                else "C" if (campaign.roi or 0) > 100 else "D"
                            )
                        ),
                        "budget_utilization": (
                            "Efficient"
                            if 0.8
                            <= (
                                campaign.actual_spend / campaign.budget
                                if campaign.budget > 0
                                else 0
                            )
                            <= 1.0
                            else (
                                "Over"
                                if campaign.actual_spend > campaign.budget
                                else "Under"
                            )
                        ),
                        "overall_success": (
                            "High"
                            if (campaign.roi or 0) > 130
                            and (campaign.revenue_lift or 0) > 20
                            else "Moderate"
                        ),
                    },
                    "recommendations": [],
                }

                # Generate recommendations based on performance
                if (campaign.roi or 0) < 120:
                    result["recommendations"].append(
                        "Consider optimization strategies to improve ROI"
                    )
                if campaign.actual_spend < campaign.budget * 0.8:
                    result["recommendations"].append(
                        "Opportunity to increase spend for greater impact"
                    )
                if (campaign.revenue_lift or 0) > 25:
                    result["recommendations"].append(
                        "Strong performance - consider scaling similar campaigns"
                    )

                return json.dumps(result, indent=2)

        except Exception as e:
            return (
                f"Error fetching detailed metrics for campaign {campaign_id}: {str(e)}"
            )
