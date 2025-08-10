"""
Advanced MCP Tools for TPM - AI-powered conversation tools for Claude Desktop
These tools enable natural language interaction with the TPM ML prediction engine
"""
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from fastmcp import FastMCP
from ..models.database import get_db_session
from ..models.entities import Product, Promotion, Transaction, Campaign, Store
from ..services.ml_predictor import get_ml_service
from sqlalchemy import func, and_, desc
from sqlalchemy.orm import Session


def register_ml_tools(mcp: FastMCP) -> None:
    """Register AI-powered MCP tools with the FastMCP server"""

    @mcp.tool()
    def predict_promotion_lift(
        product_name: str,
        discount_percentage: float,
        duration_days: int = 14,
        promotion_type: str = "DISCOUNT",
        store_name: str = None
    ) -> str:
        """AI-powered promotion lift prediction for any product by name
        
        Args:
            product_name: Product name or brand (e.g., "Cheerios", "Coca Cola")
            discount_percentage: Discount percentage (5-50%)
            duration_days: Promotion duration in days (7-28)
            promotion_type: Type - DISCOUNT, BOGO, DISPLAY, COUPON, BUNDLE
            store_name: Optional store name for location-specific prediction
        """
        try:
            with get_db_session() as db:
                # Find product by name
                product = db.query(Product).filter(
                    Product.brand.ilike(f"%{product_name}%") |
                    Product.commodity_desc.ilike(f"%{product_name}%")
                ).first()
                
                if not product:
                    return f"❌ Product '{product_name}' not found. Try a different name or brand."
                
                # Find store if specified
                store_id = None
                if store_name:
                    store = db.query(Store).filter(
                        Store.store_id.ilike(f"%{store_name}%")
                    ).first()
                    store_id = store.store_id if store else None
                
                # Get ML prediction
                ml_service = get_ml_service()
                prediction = ml_service.predict_lift(
                    product_id=product.product_id,
                    store_id=store_id,
                    promotion_type=promotion_type.upper(),
                    discount_pct=discount_percentage,
                    duration_days=duration_days
                )
                
                # Calculate business impact
                baseline_sales = db.query(func.sum(Transaction.quantity)).filter(
                    Transaction.product_id == product.product_id
                ).scalar() or 100
                
                daily_baseline = baseline_sales / 365
                incremental_units = int(daily_baseline * duration_days * (prediction.prediction / 100))
                
                avg_price = db.query(func.avg(Transaction.sales_value / Transaction.quantity)).filter(
                    Transaction.product_id == product.product_id
                ).scalar() or 5.0
                
                incremental_revenue = incremental_units * avg_price
                investment = incremental_units * avg_price * (discount_percentage / 100)
                roi = (incremental_revenue / max(investment, 1)) * 100 if investment > 0 else 0
                
                # Format confidence level
                confidence = "High" if abs(prediction.confidence_upper - prediction.confidence_lower) < 15 else "Medium"
                
                result = f"""🤖 AI Promotion Prediction for {product.brand} - {product.commodity_desc}

📊 **Predicted Performance:**
• Lift: {prediction.prediction:.1f}% (Range: {prediction.confidence_lower:.1f}% - {prediction.confidence_upper:.1f}%)
• Confidence: {confidence}
• Duration: {duration_days} days at {discount_percentage}% discount

💰 **Business Impact:**
• Incremental Units: {incremental_units:,}
• Incremental Revenue: ${incremental_revenue:,.2f}
• Investment Required: ${investment:,.2f}
• Expected ROI: {roi:.0f}%

🎯 **AI Insights:**"""
                
                # Add top feature insights
                if prediction.feature_importance:
                    top_features = sorted(prediction.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    for feature, importance in top_features:
                        if importance > 0.1:
                            result += f"\n• {feature.replace('_', ' ').title()}: {importance:.0%} influence"
                
                result += f"""

🚀 **Recommendation:** {"✅ Strong promotion candidate" if prediction.prediction > 20 else "⚠️ Consider alternative strategies"}

Model: {prediction.model_version}"""
                
                return result
                
        except Exception as e:
            return f"❌ AI prediction failed: {str(e)}. Check product name and try again."

    @mcp.tool()
    def optimize_promotion_budget(
        total_budget: float,
        product_categories: str,
        max_products: int = 5,
        objectives: str = "maximize_roi"
    ) -> str:
        """AI-powered budget optimization across product categories
        
        Args:
            total_budget: Total promotion budget ($1000-$100000)
            product_categories: Categories to focus on (e.g., "frozen, cereal, beverages")
            max_products: Maximum number of products to include (3-10)
            objectives: Optimization goal - "maximize_roi", "maximize_lift", "balanced"
        """
        try:
            with get_db_session() as db:
                # Find products in specified categories
                categories = [cat.strip().lower() for cat in product_categories.split(",")]
                
                products = []
                for category in categories:
                    category_products = db.query(Product).filter(
                        Product.commodity_desc.ilike(f"%{category}%") |
                        Product.department.ilike(f"%{category}%")
                    ).limit(max_products // len(categories) + 1).all()
                    products.extend(category_products)
                
                if not products:
                    return f"❌ No products found for categories: {product_categories}"
                
                # Limit to max_products
                products = products[:max_products]
                product_ids = [p.product_id for p in products]
                
                # Get ML optimization
                ml_service = get_ml_service()
                optimization = ml_service.optimize_budget(
                    total_budget=total_budget,
                    product_ids=product_ids
                )
                
                result = f"""🤖 AI Budget Optimization for ${total_budget:,.2f}

📊 **Optimized Allocation:**
• Target Categories: {product_categories}
• Products Analyzed: {len(products)}
• Budget Allocated: ${optimization['allocated_budget']:,.2f}
• Expected ROI: {optimization['expected_roi']:.1f}%
• Expected Lift: {optimization['expected_lift']:.1f}%

💡 **AI Recommendations:**"""
                
                for i, rec in enumerate(optimization['recommendations'][:5], 1):
                    result += f"""

{i}. **{rec['product_name']}**
   • Budget: ${rec['allocated_budget']:,.2f} ({rec['budget_percentage']}%)
   • Discount: {rec['recommended_discount']}%
   • Expected Lift: {rec['expected_lift']:.1f}%
   • Expected ROI: {rec['expected_roi']:.1f}%"""
                
                # Optimization score interpretation
                score = optimization.get('optimization_score', 0.5)
                if score > 0.8:
                    status = "🎯 Excellent optimization potential"
                elif score > 0.6:
                    status = "✅ Good optimization balance"
                else:
                    status = "⚠️ Consider different product mix"
                
                result += f"""

🎯 **Optimization Quality:** {status}
📈 **Success Probability:** {min(95, 60 + (score * 30)):.0f}%

🚀 **Next Steps:**
1. Review individual product recommendations
2. Consider seasonal timing factors  
3. Monitor competitor activities
4. Plan execution timeline"""
                
                return result
                
        except Exception as e:
            return f"❌ Budget optimization failed: {str(e)}. Check parameters and try again."

    @mcp.tool()
    def analyze_competitive_impact(
        product_name: str,
        competitor_actions: str = "general_market"
    ) -> str:
        """Analyze how competitor promotions might impact your promotion strategy
        
        Args:
            product_name: Your product name
            competitor_actions: Description of competitor activities
        """
        try:
            with get_db_session() as db:
                # Find the product
                product = db.query(Product).filter(
                    Product.brand.ilike(f"%{product_name}%") |
                    Product.commodity_desc.ilike(f"%{product_name}%")
                ).first()
                
                if not product:
                    return f"❌ Product '{product_name}' not found."
                
                # Get historical performance data
                recent_promotions = db.query(Promotion).filter(
                    Promotion.product_id == product.product_id
                ).order_by(desc(Promotion.start_day)).limit(5).all()
                
                # Analyze competitive context
                category_products = db.query(Product).filter(
                    Product.commodity_desc.ilike(f"%{product.commodity_desc.split()[0]}%")
                ).count()
                
                avg_category_price = db.query(
                    func.avg(Transaction.sales_value / Transaction.quantity)
                ).join(Product).filter(
                    Product.commodity_desc.ilike(f"%{product.commodity_desc.split()[0]}%")
                ).scalar() or 0
                
                product_price = db.query(
                    func.avg(Transaction.sales_value / Transaction.quantity)
                ).filter(Transaction.product_id == product.product_id).scalar() or 0
                
                # Competitive positioning
                price_position = "Premium" if product_price > avg_category_price * 1.2 else \
                               "Value" if product_price < avg_category_price * 0.8 else "Mid-tier"
                
                result = f"""🔍 Competitive Analysis: {product.brand} - {product.commodity_desc}

📊 **Market Position:**
• Category: {product.commodity_desc}
• Price Position: {price_position} (${product_price:.2f} vs ${avg_category_price:.2f} avg)
• Market Competitors: {category_products} similar products

⚔️ **Competitive Response Strategy:**"""
                
                if competitor_actions.lower() in ["aggressive", "heavy_discount"]:
                    result += """
• 🎯 Recommendation: DEFENSIVE strategy
• Focus on BOGO or bundle promotions vs straight discounts
• Emphasize unique value propositions
• Consider loyalty program integration"""
                elif competitor_actions.lower() in ["seasonal", "holiday"]:
                    result += """
• 🎯 Recommendation: SEASONAL alignment
• Time promotions with competitor schedules
• Differentiate through promotion mechanics
• Leverage seasonal messaging"""
                else:
                    result += """
• 🎯 Recommendation: PROACTIVE strategy
• Test innovative promotion formats
• Focus on customer acquisition
• Build market share through value"""
                
                # Historical performance context
                if recent_promotions:
                    avg_discount = sum(p.discount_percentage for p in recent_promotions) / len(recent_promotions)
                    result += f"""

📈 **Historical Context:**
• Recent promotions: {len(recent_promotions)}
• Average discount used: {avg_discount:.1f}%
• Last promotion: {recent_promotions[0].start_day} days ago"""
                
                result += f"""

🤖 **AI Insights:**
• Market saturation: {"High" if category_products > 20 else "Medium"}
• Competitive intensity: {"High" if price_position == "Value" else "Medium"}
• Differentiation opportunity: {"Strong" if price_position == "Premium" else "Moderate"}

💡 **Strategic Recommendations:**
1. Monitor competitor promotion calendars
2. Test counter-positioning strategies
3. Focus on unique brand benefits
4. Consider partnership opportunities"""
                
                return result
                
        except Exception as e:
            return f"❌ Competitive analysis failed: {str(e)}"

    @mcp.tool()
    def market_intelligence_summary(
        time_horizon: str = "current_quarter",
        focus_categories: str = "all"
    ) -> str:
        """Generate AI-powered market intelligence summary for strategic planning
        
        Args:
            time_horizon: Analysis period - "current_quarter", "next_quarter", "year_ahead"
            focus_categories: Categories to focus on or "all"
        """
        try:
            with get_db_session() as db:
                # Get market overview data
                total_products = db.query(Product).count()
                total_promotions = db.query(Promotion).count()
                total_revenue = db.query(func.sum(Transaction.sales_value)).scalar() or 0
                
                # Category performance
                category_performance = db.query(
                    Product.commodity_desc,
                    func.sum(Transaction.sales_value).label('revenue'),
                    func.count(Promotion.id).label('promo_count')
                ).join(Transaction).outerjoin(
                    Promotion, Product.product_id == Promotion.product_id
                ).group_by(Product.commodity_desc).order_by(
                    desc('revenue')
                ).limit(10).all()
                
                # Promotion effectiveness
                avg_promotion_performance = db.query(
                    func.avg(Promotion.discount_percentage).label('avg_discount'),
                    func.count(Promotion.id).label('total_promos')
                ).first()
                
                result = f"""🧠 AI Market Intelligence Report - {time_horizon.replace('_', ' ').title()}

📊 **Market Overview:**
• Total Products: {total_products:,}
• Active Promotions: {total_promotions:,}
• Market Size: ${total_revenue:,.2f}
• Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

🏆 **Top Performing Categories:**"""
                
                for i, (category, revenue, promo_count) in enumerate(category_performance[:5], 1):
                    market_share = (revenue / total_revenue * 100) if total_revenue > 0 else 0
                    result += f"""
{i}. {category[:30]}...
   • Revenue: ${revenue:,.2f} ({market_share:.1f}% share)
   • Promotions: {promo_count} campaigns"""
                
                result += f"""

📈 **Promotion Landscape:**
• Average Discount: {avg_promotion_performance.avg_discount:.1f}%
• Campaign Activity: {avg_promotion_performance.total_promos} active
• Market Intensity: {"High" if avg_promotion_performance.total_promos > 100 else "Moderate"}

🤖 **AI Strategic Insights:**"""
                
                # Generate insights based on data
                if avg_promotion_performance.avg_discount > 25:
                    result += "\n• ⚠️ High average discounts suggest price-sensitive market"
                else:
                    result += "\n• ✅ Moderate discount levels indicate brand value retention"
                
                if total_promotions > total_products * 0.3:
                    result += "\n• 🎯 Active promotional environment - timing is critical"
                else:
                    result += "\n• 📊 Stable market - opportunity for innovative campaigns"
                
                result += f"""

🎯 **Strategic Recommendations for {time_horizon.replace('_', ' ').title()}:**
1. Focus on top-performing categories for maximum impact
2. {"Defensive positioning due to high competitive activity" if total_promotions > 50 else "Aggressive growth strategy opportunity"}
3. {"Premium pricing strategy viable" if avg_promotion_performance.avg_discount < 20 else "Value-focused approach recommended"}
4. Consider seasonal factors and competitor calendar alignment

📊 **Market Opportunities:**
• Underserved categories with growth potential
• Innovation gaps in promotion mechanics  
• Partnership and bundle opportunities
• Digital and omnichannel integration

⏰ **Timing Insights:**
• {"Peak promotional season - compete for visibility" if datetime.now().month in [11, 12] else "Standard market conditions"}
• Consider quarterly budget cycles and seasonal patterns"""
                
                return result
                
        except Exception as e:
            return f"❌ Market intelligence analysis failed: {str(e)}"

    @mcp.tool()
    def promotion_performance_diagnostics(
        campaign_identifier: str,
        analysis_depth: str = "comprehensive"
    ) -> str:
        """Deep AI analysis of promotion performance with actionable insights
        
        Args:
            campaign_identifier: Campaign ID, name, or product name
            analysis_depth: "quick", "standard", or "comprehensive"
        """
        try:
            with get_db_session() as db:
                # Try to find campaign by ID first, then by name/product
                campaign = None
                try:
                    campaign_id = int(campaign_identifier)
                    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
                except ValueError:
                    # Search by name or product
                    campaign = db.query(Campaign).filter(
                        Campaign.name.ilike(f"%{campaign_identifier}%")
                    ).first()
                    
                    if not campaign:
                        # Search by product
                        product = db.query(Product).filter(
                            Product.brand.ilike(f"%{campaign_identifier}%") |
                            Product.commodity_desc.ilike(f"%{campaign_identifier}%")
                        ).first()
                        
                        if product:
                            # Find recent promotions for this product
                            recent_promotion = db.query(Promotion).filter(
                                Promotion.product_id == product.product_id
                            ).order_by(desc(Promotion.start_day)).first()
                            
                            if recent_promotion:
                                return f"""📊 Recent Promotion Analysis: {product.brand} - {product.commodity_desc}

⚡ **Quick Analysis:**
• Discount: {recent_promotion.discount_percentage}%
• Type: {recent_promotion.promotion_type}
• Duration: {recent_promotion.duration_days} days

💡 **For comprehensive campaign analysis, create a formal campaign and track metrics.""""""
                
                if not campaign:
                    return f"❌ Campaign '{campaign_identifier}' not found. Try campaign ID, name, or product name."
                
                # Comprehensive analysis
                budget_str = f"${campaign.budget:,.2f}"
                spend_str = f"${campaign.actual_spend:,.2f}"
                efficiency = f"{(campaign.actual_spend/campaign.budget*100):.1f}%" if campaign.budget > 0 else "N/A"
                
                result = f"""AI Performance Diagnostics: {campaign.name}

**Campaign Overview:**
- Status: {campaign.status.upper()}
- Budget: {budget_str}
- Actual Spend: {spend_str}
- Efficiency: {efficiency} of budget used

**Performance Metrics:**"""
                
                if campaign.roi:
                    roi_status = "Excellent" if campaign.roi > 150 else "Good" if campaign.roi > 120 else "Below target"
                    result += f"\n- ROI: {campaign.roi}% ({roi_status})"
                
                if campaign.revenue_lift:
                    lift_status = "Outstanding" if campaign.revenue_lift > 30 else "Strong" if campaign.revenue_lift > 20 else "Moderate"
                    result += f"\n- Revenue Lift: {campaign.revenue_lift}% ({lift_status})"
                
                result += f"""

**Timeline Analysis:**
- Start: {campaign.start_date}
- End: {campaign.end_date}
- Duration: {(campaign.end_date - campaign.start_date).days} days
- Phase: {"Active" if campaign.status == "active" else "Completed"}"""
                
                # AI Insights based on performance
                result += "\n\n**AI Diagnostic Insights:**"
                
                if campaign.roi and campaign.roi < 120:
                    result += "\n- ROI below benchmark - consider optimization strategies"
                elif campaign.roi and campaign.roi > 150:
                    result += "\n- Exceptional ROI - replicate successful elements"
                    
                if campaign.actual_spend < campaign.budget * 0.8:
                    result += "\n- Underutilized budget - opportunity for expansion"
                elif campaign.actual_spend > campaign.budget * 1.1:
                    result += "\n- Over budget - review spend controls"
                
                if analysis_depth == "comprehensive":
                    result += f"""

**Optimization Recommendations:**
1. {"Scale successful elements" if (campaign.roi or 0) > 130 else "Test alternative promotion mechanics"}
2. {"Increase budget allocation" if campaign.actual_spend < campaign.budget * 0.9 else "Optimize spend efficiency"}
3. {"Extend successful campaigns" if (campaign.roi or 0) > 140 else "Consider early termination if underperforming"}
4. {"Replicate to similar products" if (campaign.revenue_lift or 0) > 25 else "Analyze root causes of performance gaps"}

**Strategic Next Steps:**
- A/B test variations of successful elements
- Analyze customer segment responses
- Review competitive landscape changes
- Plan follow-up campaigns or extensions"""
                
                return result
                
        except Exception as e:
            return f"❌ Performance diagnostics failed: {str(e)}"
