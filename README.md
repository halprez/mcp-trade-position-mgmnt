# Trade Promotion Management MCP Server
## AI-Powered Trade Promotion Manager for Claude Desktop

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-XGBoost%20%7C%20ML-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Protocol](https://img.shields.io/badge/protocol-MCP-red?style=for-the-badge)
![Status](https://img.shields.io/badge/status-beta-yellow?style=for-the-badge)
![Development](https://img.shields.io/badge/development-active-brightgreen?style=for-the-badge)

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/postgresql-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Claude](https://img.shields.io/badge/Claude%20Desktop-integrated-9f7aea?style=for-the-badge)

## 🎬 **See It In Action**

<p align="center">
  <img src="media/mcp_tpm_claude_desktop.gif" alt="TPM Claude Desktop Demo" width="100%"/>
</p>

> **🚀 Watch the AI-powered Trade Promotion Management in action!** Natural language conversation with Claude Desktop for ML predictions, budget optimization, and strategic insights.

**Demo Features:**
- 🤖 **Natural Language Queries** - Ask about promotions in plain English
- 📊 **ML-Powered Predictions** - Real-time promotion lift forecasting
- 💰 **Budget Optimization** - AI-driven allocation strategies
- 📈 **Strategic Insights** - Competitive analysis and recommendations

---

## 🎯 Project Overview

**Project Name**: TPM-MCP - Trade Promotion Management MCP Server

**Objective**: Build an MCP (Model Context Protocol) server that connects FastAPI-powered trade promotion management directly to Claude Desktop, showcasing cutting-edge AI integration, Python expertise, and modern architecture.

**Revolutionary Approach**: Instead of traditional UIs, create an intelligent TPM assistant that Claude can interact with directly through MCP tools, making trade promotion planning as simple as having a conversation.

**Business Context**: Enable FMCG managers to optimize promotional campaigns through natural language interactions with Claude, powered by real data and AI insights.

---

## 📊 Dataset Strategy

### Primary Dataset: Dunnhumby - The Complete Journey
**Source**: https://www.dunnhumby.com/source-files/
**Why Perfect for TPM**:
- Real retail data from 2,500 households over 2 years
- Contains promotional data (sale tags, displays, coupons)
- Base price vs. shelf price (discount calculation)
- Multiple categories: mouthwash, pretzels, frozen pizza, cereal
- Transaction-level detail for promotion effectiveness analysis

**Key Data Elements**:
```
- Household transactions
- Product details (brand, category, price)
- Promotional mechanics (coupons, displays, sale tags)
- Time series data (156 weeks)
- Store and geography information
```

### Supplementary Datasets:
1. [**Kaggle Retail Sales Dataset**](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset) - Additional transaction patterns
2. **Generated Synthetic Data** - Custom promotional scenarios
3. **External APIs** - Real-time pricing/competitor data

---

## 🏗️ MCP Architecture

### Claude Desktop ↔ TPM Server Integration
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude        │    │   MCP Server    │    │   TPM FastAPI   │
│   Desktop       │◄──►│   (FastMCP)     │◄──►│   Application   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ Natural Language      │ MCP Protocol          │ API Calls
         │ Commands              │ Tools & Resources     │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ "Analyze Q4     │    │ - promotion_roi │    │   Analytics     │
│  promotions"    │    │ - predict_lift  │    │   Engine        │
│ "Optimize       │    │ - create_plan   │    │                 │
│  discounts"     │    │ - analyze_data  │    │   ML Models     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   PostgreSQL    │    │   Dunnhumby     │
                    │   (Results)     │    │   Dataset       │
                    └─────────────────┘    └─────────────────┘
```

---

## 🛠️ Technology Stack

### MCP Integration
**FastMCP**: High-level MCP server implementation
**Core Technologies**:
- **FastAPI**: High-performance API framework
- **FastMCP**: Zero-config MCP server creation  
- **Docker**: Containerized deployment
- **PostgreSQL**: Promotional transaction data
- **Pandas/NumPy**: Data processing and analysis
- **scikit-learn/XGBoost**: ML prediction models

### AI/ML Stack
```python
# Data Processing
pandas, numpy

# Machine Learning
scikit-learn (baseline models)
xgboost (promotion lift prediction)
pytorch (deep learning for complex patterns)

# Feature Engineering
Feature store for promotional attributes
Time series analysis for seasonality

# Model Serving
FastAPI endpoints for real-time predictions
MLflow for model versioning
```

---

## 📁 Project Structure

```
tpm-mcp/
├── docker-compose.yml
├── requirements.txt
├── README.md
├── claude_desktop_config.json  # MCP configuration
│
├── src/
│   ├── main.py                 # FastAPI + MCP server
│   ├── models/
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── entities.py
│   │
│   ├── services/
│   │   ├── promotion_analytics.py
│   │   ├── ml_predictor.py
│   │   ├── campaign_optimizer.py
│   │   └── data_processor.py
│   │
│   ├── api/
│   │   ├── promotions.py       # FastAPI routes
│   │   ├── analytics.py
│   │   └── predictions.py
│   │
│   └── mcp/
│       ├── tools.py            # Custom MCP tools
│       ├── resources.py        # MCP resources
│       └── prompts.py          # MCP prompts
│
├── data/
│   ├── dunnhumby/             # Raw Dunnhumby dataset
│   ├── processed/             # Cleaned data
│   └── models/                # Trained ML models
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── promotion_analysis.ipynb
│   └── model_development.ipynb
│
└── deployment/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🤖 AI/ML Components

### Model 1: Promotion Lift Predictor
```python
# Predict sales uplift from promotional campaigns
# Features: discount_pct, display_type, customer_segment, seasonality
# Target: lift_percentage
# Algorithm: XGBoost Regressor
```

### Model 2: ROI Optimizer
```python
# Optimize promotional spend for maximum ROI
# Multi-objective: maximize_volume & maximize_profit
# Constraints: budget_limit, inventory_capacity
# Algorithm: Genetic Algorithm or Bayesian Optimization
```

### Model 3: Customer Response Model
```python
# Predict customer response to promotional offers
# Features: purchase_history, demographics, promotion_type
# Target: probability_of_purchase
# Algorithm: Neural Network or Random Forest
```

### Model 4: Demand Forecasting
```python
# Forecast demand under different promotional scenarios
# Features: historical_sales, seasonality, promotions, external_factors
# Target: predicted_demand
# Algorithm: LSTM or Prophet
```
---

## 🐳 Docker & MCP Setup

### docker-compose.yml
```yaml
version: '3.8'
services:
  tpm-mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tpm
      - MCP_SERVER_NAME=Trade Promotion Assistant
    depends_on:
      - postgres
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=tpm
      - POSTGRES_USER=user  
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Claude Desktop Configuration
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "tpm-assistant": {
      "command": "mcp-proxy",
      "args": ["http://localhost:8000/mcp"]
    }
  }
}
```

## 💡 Claude Desktop Demo Scenarios

### Scenario 1: "Analyze Q4 Performance"
**User**: *"Hey Claude, analyze our frozen pizza promotions from Q4 and tell me which ones performed best"*

**Claude Response**: *Using the analyze_promotion_roi tool...*
```
Top Q4 Frozen Pizza Promotions:
1. DiGiorno BOGO (Week 45): 127% ROI, 34% lift
2. Red Baron 30% off (Week 48): 89% ROI, 22% lift  
3. Totino's Display + 25% (Week 52): 156% ROI, 41% lift

Key Insights:
- Display placements increased effectiveness by avg 28%
- BOGO performed better than straight discounts
- Week 52 had highest response due to holiday shopping
```

### Scenario 2: "Plan New Campaign"
**User**: *"I need to plan a promotion for cereal category with $50k budget, focusing on family brands"*

**Claude Response**: *Using optimize_promotion_budget and predict_promotion_lift tools...*
```
Recommended Campaign Strategy:
Budget Allocation:
- Cheerios (40%): $20k - BOGO, 4 weeks → Est. 156% ROI
- Lucky Charms (35%): $17.5k - 25% off + Display → Est. 142% ROI  
- Frosted Flakes (25%): $12.5k - 30% off, 2 weeks → Est. 134% ROI

Predicted Results:
- Total incremental revenue: $127k
- Overall campaign ROI: 154%
- Incremental households reached: 2,847
```

### Scenario 3: "Competitive Intelligence"
**User**: *"What promotional strategies are working for Kellogg's vs General Mills in the cereal category?"*

**Claude Response**: *Using competitive_analysis tool...*
```
Competitive Analysis: Cereal Category

Kellogg's Strategy:
- Prefers shorter, higher-discount promotions (avg 28%)
- Heavy use of in-store displays (73% of promotions)
- Focus on holiday/back-to-school timing

General Mills Strategy:  
- Longer duration, moderate discounts (avg 22%)
- BOGO promotions (45% of campaigns)
- Consistent monthly promotional cadence

Performance Comparison:
- Kellogg's: Higher lift per promotion (+31% avg)
- General Mills: Better customer retention (+18% repeat purchase)
```

### Scenario 4: "Real-time Optimization"
**User**: *"Our current Pepsi promotion is underperforming. What should we adjust?"*

**Claude Response**: *Using real-time analytics and recommendation tools...*
```
Current Pepsi Campaign Analysis:
- Week 2 performance: 67% of target
- Issue: Low visibility (only 34% shelf compliance)
- Competitor Coke launched competing promotion

Immediate Recommendations:
1. Add secondary display placement (+$3k investment)
2. Extend promotion by 1 week with deeper discount
3. Bundle with complementary snacks category

Expected Impact:
- Improved performance to 89% of original target
- Additional investment: $4.2k
- Revised ROI: 112% (down from 134% target but positive)
```

## 🚀 Quick Start - Claude Desktop Integration

> **Ready to get started?** Follow our comprehensive setup guide: **[📋 Claude Desktop Integration Guide](./CLAUDE_DESKTOP_GUIDE.md)**

### ⚡ One-Command Setup
```bash
# Clone and configure in 30 seconds
git clone <your-repo-url>
cd mcp-trade-position-mgmnt
./install_config.sh
```

### 🧪 Test with Natural Language
Once connected to Claude Desktop, try these example prompts:
- *"What can this TPM system do?"* 
- *"Predict lift for Cheerios with 25% discount for 2 weeks"*
- *"Optimize $50k budget across cereal, frozen foods, and beverages"*

➡️ **[Complete Setup Instructions →](./CLAUDE_DESKTOP_GUIDE.md)**

---

## 🚀 Development Quick Start

### 1. Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <your-repo>
cd mcp-trade-position-mgmnt
uv sync --extra dev --extra notebook
```

### 2. Configure Claude Desktop Integration
```bash
# One-command installation
./install_config.sh
```
➡️ **[Complete Claude Desktop Setup Guide →](./CLAUDE_DESKTOP_GUIDE.md)**

### 3. Start Development Server
```bash
# Run with all services
make docker

# Or run MCP server only  
make run
```

### 4. Generate Test Data
```bash
# Create sample dataset
make sample-data
make process-sample
```

### 5. Test Natural Language Interface
**Open Claude Desktop and try:**
- *"What can this TPM system do?"*
- *"Predict lift for Cheerios with 25% discount for 2 weeks"*  
- *"Optimize $50k budget across cereal, frozen foods, beverages"*
- *"Analyze competitive impact of Pepsi's summer campaign"*

📚 **[See Complete Test Examples →](./CLAUDE_DESKTOP_GUIDE.md#-how-to-interact)**

