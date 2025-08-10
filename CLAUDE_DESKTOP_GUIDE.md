# 🚀 Claude Desktop Integration Guide - TPM AI Assistant

> **← Back to:** [📋 Main README](./README.md) | [🛠️ Development Setup](./README.md#development-quick-start)

## 🎯 **Quick Setup**

### 1. Install Configuration
```bash
cd /path/to/mcp-trade-position-mgmnt
./install_config.sh
```

### 2. Alternative Manual Setup
```bash
# Copy template and customize paths
cp claude_desktop_config_template.json ~/.config/Claude/claude_desktop_config.json
# Edit the [PROJECT_DIR] placeholder with your actual project path
```

### 3. Start Using
- Restart Claude Desktop
- The TPM Assistant will auto-connect and show a welcome message

## 🎉 **First Time Experience**

When you first connect, Claude Desktop will automatically show:
- **Welcome Message** with full capabilities overview
- **Quick Start Examples** to get you going immediately  
- **Available Tools** and data access points
- **Pro Tips** for effective usage

## 💬 **How to Interact**

### **🤖 Discovery Commands**
Start with these to understand what's available:

- **"Welcome"** or call `claude_desktop_welcome()` → Initial setup and overview
- **"What can this TPM system do?"** → Full capabilities discovery  
- **"Show me available data"** → Dataset overview and categories
- **"Help and examples"** → Comprehensive guide with use cases

### **🎯 Core AI Capabilities**

#### **Promotion Prediction (ML-Powered)**
```
"Predict lift for Cheerios with 25% discount for 2 weeks"
"What would happen with BOGO on frozen pizza?"  
"Compare 20% vs 30% discount for premium brands"
```

#### **Budget Optimization (AI-Powered)**  
```
"Optimize $50k budget across cereal, frozen, beverages"
"How should I allocate my Q4 promotion budget?"
"Find best ROI distribution for holiday campaigns"
```

#### **Market Intelligence**
```
"Analyze cereal category market landscape" 
"Give me competitive intelligence on beverage promotions"
"What are the top performing categories this quarter?"
```

#### **Performance Analysis**
```
"Diagnose my current Doritos campaign performance"
"Why is my frozen food promotion underperforming?"  
"Compare ROI across all active promotions"
```

## 🔍 **Advanced Usage**

### **Scenario Analysis**
- "What if Pepsi launches a counter-promotion during my campaign?"
- "How would economic downturn affect promotion sensitivity?"
- "Analyze seasonal impact on frozen food promotions"

### **Strategic Planning**  
- "Develop Q4 promotion strategy for premium brands"
- "Create defensive plan against aggressive competitor pricing"
- "Design customer acquisition campaign for new market"

### **Competitive Intelligence**
- "How does our cereal promotion effectiveness compare to Kellogg's?"
- "Analyze General Mills' recent promotional strategy shifts"
- "What promotion mechanics work best in our category?"

## 🛠 **Available Tools & Resources**

### **Core MCP Tools**
- `predict_promotion_lift()` - AI-powered promotion predictions
- `optimize_promotion_budget()` - Budget allocation optimization  
- `analyze_competitive_impact()` - Competitive analysis
- `market_intelligence_summary()` - Market insights
- `promotion_performance_diagnostics()` - Campaign analysis

### **Data Resources**
- `tpm://products/{category}` - Product performance data
- `tpm://promotions/{time_period}` - Historical promotion analysis
- `tpm://analytics/top-performers/{metric}` - Performance rankings
- `tpm://market-data/category-analysis/{category}` - Market intelligence

### **REST API Endpoints**
- `/predictions/lift-prediction` - ML prediction service
- `/predictions/budget-optimization` - AI optimization  
- `/analytics/dashboard` - Performance analytics
- `/promotions/campaigns` - Campaign management

## 💡 **Best Practices**

### **For Accurate Predictions**
- Use specific product names ("Cheerios" not "cereal")
- Provide realistic discount percentages (5-50%)
- Specify reasonable durations (1-8 weeks)
- Include store context when relevant

### **For Effective Analysis**  
- Ask follow-up questions for deeper insights
- Request scenario analysis and sensitivity testing
- Get competitive context for strategic decisions
- Use data resources for comprehensive understanding

### **For Strategic Planning**
- Combine multiple tools for comprehensive analysis  
- Request both quantitative predictions and qualitative insights
- Consider competitive responses and market dynamics
- Plan for seasonal and economic factors

## 🔗 **Global Shortcuts**
Claude Desktop shortcuts for quick access:
- **tpm-welcome** → Welcome and overview
- **tpm-discover** → Full capabilities  
- **tpm-help** → Help and examples
- **tpm-data** → Data overview

## 🎪 **Example Conversation Flow**

**User:** "I need help planning a cereal promotion"

**Claude:** *[Calls discover_tpm_capabilities()]*
"I can help you with AI-powered cereal promotion planning! Let me show you what's available..."

**User:** "Predict performance for Cheerios 20% off for 2 weeks"  

**Claude:** *[Calls predict_promotion_lift()]*
"Using ML models trained on historical data... Here's the predicted performance..."

**User:** "How does this compare to BOGO promotions?"

**Claude:** *[Calls predict_promotion_lift() with BOGO]*
"Comparing the two strategies... BOGO typically generates..."

**User:** "Optimize my $25k budget across top cereal brands"

**Claude:** *[Calls optimize_promotion_budget()]*
"AI optimization suggests the following allocation..."

## 🚨 **Troubleshooting**

### **If Tools Don't Work**
- Check TPM server is running on port 8000
- Verify Claude Desktop config is correct
- Restart Claude Desktop after config changes

### **If Predictions Seem Off**  
- Verify product names are spelled correctly
- Check discount percentages are realistic
- Ensure promotion duration is reasonable

### **If No Data Shows**
- Run `get_sample_data_overview()` to check data availability
- Verify database connection in server logs
- Check that sample data has been processed

## 🎯 **Success Metrics**

Track your promotion optimization success:
- **ROI Improvement** through AI-optimized allocations
- **Prediction Accuracy** vs actual campaign performance  
- **Strategic Advantage** through competitive intelligence
- **Time Savings** through automated analysis
- **Decision Quality** through data-driven insights

---

**Ready to revolutionize your trade promotion management through AI conversation with Claude Desktop!** 🚀

---

## 🔗 **Related Documentation**

- **[← Main README](./README.md)** - Project overview and development setup
- **[🛠️ Development Commands](./README.md#development-commands)** - Build, test, and deployment commands  
- **[📊 Architecture](./README.md#technology-architecture)** - Technical architecture and design
- **[🎯 Demo Scenarios](./README.md#claude-desktop-demo-scenarios)** - Detailed conversation examples