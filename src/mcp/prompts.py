"""
Strategic MCP Prompts - AI-powered planning templates for TPM
Provides contextual prompts to enhance Claude's understanding of trade promotion management
"""

from typing import Dict, List, Optional
from datetime import datetime

from fastmcp import FastMCP


def register_strategic_prompts(mcp: FastMCP) -> None:
    """Register strategic planning prompts with the FastMCP server"""

    @mcp.prompt()
    def create_promotion_strategy_prompt(
        business_objective: str,
        market_conditions: str,
        budget_constraints: str,
        time_horizon: str = "quarterly",
    ) -> str:
        """Generate strategic promotion planning prompt for comprehensive analysis

        Args:
            business_objective: Primary goal (e.g., "increase market share", "maximize profit")
            market_conditions: Current market state (e.g., "competitive", "seasonal surge")
            budget_constraints: Budget parameters (e.g., "$50k total", "10% increase from last quarter")
            time_horizon: Planning period (e.g., "quarterly", "annual", "campaign-specific")
        """

        current_date = datetime.now().strftime("%B %Y")

        return f"""You are an expert Trade Promotion Manager with access to comprehensive TPM analytics and AI-powered prediction tools. 

**STRATEGIC CONTEXT ({current_date})**
• Business Objective: {business_objective}
• Market Conditions: {market_conditions}  
• Budget Parameters: {budget_constraints}
• Planning Horizon: {time_horizon}

**AVAILABLE AI TOOLS & DATA**
You have access to:
- ML-powered promotion lift prediction models
- AI budget optimization algorithms  
- Competitive analysis frameworks
- Historical performance databases
- Real-time market intelligence

**STRATEGIC PLANNING FRAMEWORK**

1. **Market Analysis** - Use `market_intelligence_summary` to understand:
   - Category performance trends
   - Competitive landscape shifts
   - Consumer behavior patterns
   - Seasonal/cyclical factors

2. **Opportunity Assessment** - Leverage AI tools to identify:
   - High-potential products for promotion
   - Optimal discount levels and mechanics
   - Underexploited market segments
   - Timing opportunities vs competition

3. **Resource Optimization** - Apply ML algorithms to:
   - Maximize ROI across product portfolio
   - Balance risk vs reward in promotion mix
   - Allocate budget for optimal impact
   - Sequence promotions for sustained growth

4. **Performance Prediction** - Use predictive models to:
   - Forecast promotion lift and incrementality
   - Estimate customer acquisition vs retention
   - Project competitive response scenarios
   - Calculate break-even and profit scenarios

5. **Strategic Recommendations** - Synthesize analysis into:
   - Prioritized promotion calendar
   - Risk mitigation strategies
   - Success metrics and KPIs
   - Contingency plans and alternatives

**KEY QUESTIONS TO ADDRESS**
- What promotion mix will achieve {business_objective} given {market_conditions}?
- How should budget be allocated across categories/products/time periods?
- What competitive responses should be anticipated and countered?
- Which metrics will best measure strategic success?
- What are the primary risks and how can they be mitigated?

**DELIVERABLE FORMAT**
Provide a comprehensive strategy document including:
- Executive summary with key recommendations
- Detailed tactical plans with timelines
- Expected outcomes with confidence intervals
- Risk assessment and mitigation plans
- Success metrics and monitoring protocols

Begin your analysis by gathering current market intelligence and historical performance data."""

    @mcp.prompt()
    def promotion_optimization_consultant_prompt(
        current_campaign: str, performance_issues: str = "general optimization"
    ) -> str:
        """Generate consultation prompt for optimizing existing promotions

        Args:
            current_campaign: Description of current campaign or promotion
            performance_issues: Specific issues or optimization goals
        """

        return f"""You are a Senior Promotion Optimization Consultant with deep expertise in retail analytics and consumer behavior. A client has approached you about their current promotional campaign.

**CLIENT SITUATION**
Current Campaign: {current_campaign}
Optimization Focus: {performance_issues}

**YOUR CONSULTATION APPROACH**

**1. DIAGNOSTIC ANALYSIS**
First, conduct a comprehensive performance audit:
- Use `promotion_performance_diagnostics` for detailed campaign analysis  
- Compare against industry benchmarks and historical performance
- Identify specific performance gaps and root causes
- Assess competitive context and market dynamics

**2. AI-POWERED INSIGHTS** 
Leverage advanced analytics to uncover:
- Hidden patterns in customer response data
- Optimal promotion mechanics and timing
- Cross-category and cannibalization effects  
- Predictive models for performance improvement

**3. OPTIMIZATION STRATEGIES**
Develop data-driven recommendations for:
- Tactical adjustments to current campaigns
- Strategic pivots for underperforming elements
- Budget reallocation for maximum impact
- Advanced targeting and personalization

**4. COMPETITIVE INTELLIGENCE**
Use `analyze_competitive_impact` to:
- Understand competitive responses and threats
- Identify differentiation opportunities  
- Anticipate market shifts and trends
- Position promotions for sustainable advantage

**5. IMPLEMENTATION ROADMAP**
Create actionable plans including:
- Immediate tactical optimizations
- Medium-term strategic adjustments  
- Long-term capability building
- Performance monitoring and iteration

**CONSULTATION FRAMEWORK**
- Ask clarifying questions to understand specific challenges
- Use AI tools to gather relevant data and insights
- Apply advanced analytics to identify optimization opportunities
- Provide specific, actionable recommendations with expected outcomes
- Include risk assessment and contingency planning

**SUCCESS METRICS**
Focus on measurable improvements in:
- ROI and profitability
- Incremental sales lift
- Market share gains
- Customer acquisition/retention
- Competitive positioning

Begin by conducting a thorough diagnostic of the current situation using available analytics tools."""
