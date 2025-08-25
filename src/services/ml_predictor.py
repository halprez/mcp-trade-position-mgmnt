"""
ML Predictor Service - AI-powered promotion prediction models
Provides real ML capabilities for the TPM MCP system
"""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models.database import get_db_session
from ..models.entities import Transaction, Promotion, Product, Household, Store


@dataclass
class PredictionResult:
    """Standard prediction result format"""

    prediction: float
    confidence_lower: float
    confidence_upper: float
    feature_importance: Dict[str, float]
    model_version: str


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    mae: float
    rmse: float
    r2: float
    cross_val_score: float


class FeatureEngineer:
    """Feature engineering pipeline for promotion prediction models"""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_db_session()
        self.scalers = {}
        self.encoders = {}

    def create_promotion_features(
        self,
        product_id: int,
        store_id: int = None,
        promotion_type: str = None,
        discount_pct: float = None,
        duration_days: int = None,
    ) -> pd.DataFrame:
        """Create ML features for promotion prediction"""

        features = {}

        # Product-level features
        product_stats = self._get_product_stats(product_id)
        features.update(product_stats)

        # Store-level features (if specified)
        if store_id:
            store_stats = self._get_store_stats(store_id)
            features.update(store_stats)

        # Promotion-specific features
        if promotion_type:
            promo_features = self._get_promotion_type_features(promotion_type)
            features.update(promo_features)

        # Discount and duration features
        if discount_pct is not None:
            features["discount_pct"] = discount_pct
            features["discount_bin"] = self._bin_discount(discount_pct)
            features["discount_squared"] = discount_pct ** 2

        if duration_days is not None:
            features["duration_days"] = duration_days
            features["duration_bin"] = self._bin_duration(duration_days)
            features["duration_log"] = np.log1p(duration_days)
            
        # Interaction features
        if discount_pct is not None and duration_days is not None:
            features["discount_duration_interaction"] = discount_pct * duration_days

        # Time-based features
        now = datetime.now()
        features["month"] = now.month
        features["quarter"] = (now.month - 1) // 3 + 1
        features["is_weekend"] = now.weekday() >= 5
        features["is_holiday_season"] = now.month in [11, 12, 1]
        features["start_day"] = now.timetuple().tm_yday  # Day of year as proxy

        return pd.DataFrame([features])

    def _get_product_stats(self, product_id: int) -> Dict:
        """Extract product-level statistical features"""

        # Basic product info
        product = (
            self.session.query(Product).filter(Product.product_id == product_id).first()
        )

        if not product:
            return {"product_id": product_id}

        # Sales statistics
        sales_stats = (
            self.session.query(
                func.count(Transaction.id).label("transaction_count"),
                func.sum(Transaction.sales_value).label("total_revenue"),
                func.avg(Transaction.sales_value).label("avg_basket_value"),
                func.sum(Transaction.quantity).label("total_quantity"),
                func.avg(Transaction.sales_value / Transaction.quantity).label(
                    "avg_price"
                ),
            )
            .filter(Transaction.product_id == product_id)
            .first()
        )

        # Historical promotion performance
        promo_stats = (
            self.session.query(
                func.count(Promotion.id).label("promo_count"),
                func.avg(Promotion.discount_percentage).label("avg_discount"),
                func.avg(Promotion.duration_days).label("avg_duration"),
            )
            .filter(Promotion.product_id == product_id)
            .first()
        )

        return {
            "product_id": product_id,
            "brand_encoded": self._encode_categorical(product.brand, "brand"),
            "commodity_encoded": self._encode_categorical(
                product.commodity_desc, "commodity"
            ),
            "transaction_count": sales_stats.transaction_count or 0,
            "total_revenue": float(sales_stats.total_revenue or 0),
            "avg_basket_value": float(sales_stats.avg_basket_value or 0),
            "total_quantity": sales_stats.total_quantity or 0,
            "avg_price": float(sales_stats.avg_price or 0),
            "historical_promo_count": promo_stats.promo_count or 0,
            "historical_avg_discount": float(promo_stats.avg_discount or 0),
            "historical_avg_duration": float(promo_stats.avg_duration or 0),
        }

    def _get_store_stats(self, store_id: int) -> Dict:
        """Extract store-level features"""

        store = self.session.query(Store).filter(Store.store_id == store_id).first()

        if not store:
            return {"store_id": store_id}

        # Store performance metrics
        store_stats = (
            self.session.query(
                func.count(Transaction.id).label("store_transactions"),
                func.sum(Transaction.sales_value).label("store_revenue"),
                func.count(func.distinct(Transaction.product_id)).label(
                    "unique_products"
                ),
            )
            .filter(Transaction.store_id == store_id)
            .first()
        )

        return {
            "store_id": store_id,
            "store_transactions": store_stats.store_transactions or 0,
            "store_revenue": float(store_stats.store_revenue or 0),
            "store_product_variety": store_stats.unique_products or 0,
        }

    def _get_promotion_type_features(self, promotion_type: str) -> Dict:
        """Create promotion type features"""

        # Historical performance by type
        type_performance = (
            self.session.query(
                func.count(Promotion.id).label("type_count"),
                func.avg(Promotion.discount_percentage).label("avg_type_discount"),
            )
            .filter(Promotion.promotion_type == promotion_type)
            .first()
        )

        # One-hot encoding for promotion types
        promo_types = ["BOGO", "DISCOUNT", "DISPLAY", "COUPON", "BUNDLE"]
        promo_features = {
            f"promo_type_{ptype.lower()}": 1 if ptype == promotion_type else 0
            for ptype in promo_types
        }

        promo_features.update(
            {
                "promo_type_historical_count": type_performance.type_count or 0,
                "promo_type_avg_discount": float(
                    type_performance.avg_type_discount or 0
                ),
            }
        )

        return promo_features

    def _encode_categorical(self, value: str, category: str) -> int:
        """Encode categorical variables with label encoder"""
        if category not in self.encoders:
            self.encoders[category] = LabelEncoder()

        if value is None:
            value = "Unknown"

        # Fit encoder if not already fitted
        try:
            return self.encoders[category].transform([value])[0]
        except:
            # If unseen category, fit with current value
            self.encoders[category].fit([value])
            return self.encoders[category].transform([value])[0]

    def _bin_discount(self, discount_pct: float) -> int:
        """Bin discount percentages into categories"""
        if discount_pct <= 10:
            return 0
        elif discount_pct <= 20:
            return 1
        elif discount_pct <= 30:
            return 2
        elif discount_pct <= 50:
            return 3
        else:
            return 4

    def _bin_duration(self, days: int) -> int:
        """Bin duration into categories"""
        if days <= 7:
            return 0
        elif days <= 14:
            return 1
        elif days <= 21:
            return 2
        else:
            return 3


class PromotionLiftPredictor:
    """ML model for predicting promotion sales lift"""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

        # Initialize model based on type
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif model_type == "gradient_boost":
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        else:
            self.model = LinearRegression()

    def prepare_training_data(
        self, session: Optional[Session] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from historical promotion performance"""

        if session is None:
            session = get_db_session()

        # Query historical promotions with their outcomes
        query = (
            session.query(
                Promotion.product_id,
                Promotion.store_id,
                Promotion.promotion_type,
                Promotion.discount_percentage,
                Promotion.duration_days,
                Promotion.start_day,
                Promotion.end_day,
                # Calculate actual lift from transaction data
                func.coalesce(
                    func.sum(Transaction.quantity).filter(
                        and_(
                            Transaction.day >= Promotion.start_day,
                            Transaction.day <= Promotion.end_day,
                            Transaction.product_id == Promotion.product_id,
                            Transaction.store_id == Promotion.store_id,
                        )
                    ),
                    0,
                ).label("promo_sales"),
            )
            .outerjoin(
                Transaction,
                and_(
                    Transaction.product_id == Promotion.product_id,
                    Transaction.store_id == Promotion.store_id,
                    Transaction.day >= Promotion.start_day,
                    Transaction.day <= Promotion.end_day,
                ),
            )
            .group_by(
                Promotion.id,
                Promotion.product_id,
                Promotion.store_id,
                Promotion.promotion_type,
                Promotion.discount_percentage,
                Promotion.duration_days,
                Promotion.start_day,
                Promotion.end_day,
            )
            .limit(10000)
        )  # Limit for performance

        training_data = []
        targets = []

        for row in query.all():
            # Create features
            features_df = self.feature_engineer.create_promotion_features(
                product_id=row.product_id,
                store_id=row.store_id,
                promotion_type=row.promotion_type,
                discount_pct=row.discount_percentage,
                duration_days=row.duration_days,
            )

            # Calculate baseline (pre-promotion sales)
            baseline_sales = (
                session.query(func.avg(Transaction.quantity))
                .filter(
                    and_(
                        Transaction.product_id == row.product_id,
                        Transaction.store_id == row.store_id,
                        Transaction.day >= max(1, row.start_day - 30),
                        Transaction.day < row.start_day,
                    )
                )
                .scalar()
                or 1
            )

            # Calculate lift percentage
            daily_baseline = baseline_sales
            total_baseline = daily_baseline * row.duration_days
            actual_lift_pct = (
                (row.promo_sales - total_baseline) / max(total_baseline, 1)
            ) * 100

            # Only include reasonable lifts (filter outliers)
            if -50 <= actual_lift_pct <= 200:
                training_data.append(features_df.iloc[0].to_dict())
                targets.append(actual_lift_pct)

        session.close()

        if not training_data:
            raise ValueError("No suitable training data found")

        X = pd.DataFrame(training_data)
        y = pd.Series(targets)

        # Store feature names
        self.feature_names = X.columns.tolist()

        return X, y

    def train(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> ModelMetrics:
        """Train the promotion lift prediction model"""

        # Prepare data if not provided
        if X is None or y is None:
            X, y = self.prepare_training_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.model_type == "xgboost":
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)

        metrics = ModelMetrics(
            mae=mean_absolute_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            r2=r2_score(y_test, y_pred),
            cross_val_score=np.mean(
                cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            ),
        )

        self.is_trained = True
        return metrics

    def predict(
        self,
        product_id: int,
        store_id: Optional[int] = None,
        promotion_type: str = "DISCOUNT",
        discount_pct: float = 20.0,
        duration_days: int = 14,
    ) -> PredictionResult:
        """Predict promotion lift percentage"""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create features
        features_df = self.feature_engineer.create_promotion_features(
            product_id=product_id,
            store_id=store_id,
            promotion_type=promotion_type,
            discount_pct=discount_pct,
            duration_days=duration_days,
        )

        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # Reorder columns to match training data
        features_df = features_df[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(features_df)

        # Make prediction
        prediction = self.model.predict(X_scaled)[0]

        # Calculate confidence intervals (simplified)
        if hasattr(self.model, "predict_proba"):
            # For models with uncertainty estimation
            confidence_range = 8.0
        else:
            # Simple confidence based on model type
            confidence_range = 10.0 if self.model_type == "linear" else 6.0

        confidence_lower = max(0, prediction - confidence_range)
        confidence_upper = min(100, prediction + confidence_range)

        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, importance in zip(
                self.feature_names, self.model.feature_importances_
            ):
                feature_importance[name] = float(importance)

        return PredictionResult(
            prediction=float(prediction),
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            feature_importance=feature_importance,
            model_version=f"{self.model_type}_v1.0",
        )

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.is_trained = model_data["is_trained"]


class ROIOptimizer:
    """Budget optimization engine for maximum ROI"""

    def __init__(self, lift_predictor: PromotionLiftPredictor):
        self.lift_predictor = lift_predictor

    def optimize_budget_allocation(
        self,
        total_budget: float,
        product_ids: List[int],
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """Optimize budget allocation across products"""

        session = get_db_session()

        # Default constraints
        if constraints is None:
            constraints = {
                "max_allocation_pct": 40,  # Max 40% to any single product
                "min_allocation_pct": 5,  # Min 5% to each product
                "max_discount_pct": 35,  # Max discount percentage
                "preferred_duration": 14,  # Preferred promotion duration
            }

        # Analyze each product's potential
        product_analysis = []

        for product_id in product_ids:
            # Get product info
            product = (
                session.query(Product).filter(Product.product_id == product_id).first()
            )

            if not product:
                continue

            # Test different discount levels to find optimal
            best_roi = 0
            best_config = None

            for discount in [15, 20, 25, 30, 35]:
                if discount > constraints["max_discount_pct"]:
                    continue

                # Predict lift
                lift_result = self.lift_predictor.predict(
                    product_id=product_id,
                    promotion_type="DISCOUNT",
                    discount_pct=discount,
                    duration_days=constraints["preferred_duration"],
                )

                # Calculate ROI
                avg_price = self._get_average_price(product_id, session)
                baseline_units = self._estimate_baseline_units(
                    product_id, constraints["preferred_duration"], session
                )

                # ROI calculation
                investment_per_unit = avg_price * (discount / 100)
                incremental_units = baseline_units * (lift_result.prediction / 100)
                total_investment = investment_per_unit * (
                    baseline_units + incremental_units
                )
                incremental_revenue = incremental_units * avg_price

                roi = (
                    (incremental_revenue / max(total_investment, 1)) * 100
                    if total_investment > 0
                    else 0
                )

                if roi > best_roi:
                    best_roi = roi
                    best_config = {
                        "discount_pct": discount,
                        "predicted_lift": lift_result.prediction,
                        "estimated_investment": total_investment,
                        "estimated_revenue": incremental_revenue,
                        "roi": roi,
                    }

            if best_config:
                product_analysis.append(
                    {
                        "product_id": product_id,
                        "product_name": f"{product.brand} - {product.commodity_desc}",
                        "roi_score": best_roi,
                        "config": best_config,
                    }
                )

        # Sort by ROI potential
        product_analysis.sort(key=lambda x: x["roi_score"], reverse=True)

        # Allocate budget
        allocations = []
        remaining_budget = total_budget
        min_budget = total_budget * (constraints["min_allocation_pct"] / 100)
        max_budget = total_budget * (constraints["max_allocation_pct"] / 100)

        for i, product_info in enumerate(product_analysis):
            if remaining_budget <= 0:
                break

            # Calculate allocation
            if i == len(product_analysis) - 1:  # Last product gets remainder
                allocation = min(remaining_budget, max_budget)
            else:
                # Allocate based on ROI score relative to others
                roi_weight = product_info["roi_score"] / sum(
                    p["roi_score"] for p in product_analysis
                )
                allocation = total_budget * roi_weight
                allocation = max(
                    min_budget, min(allocation, max_budget, remaining_budget)
                )

            allocations.append(
                {
                    "product_id": product_info["product_id"],
                    "product_name": product_info["product_name"],
                    "allocated_budget": round(allocation, 2),
                    "budget_percentage": round((allocation / total_budget) * 100, 1),
                    "recommended_discount": product_info["config"]["discount_pct"],
                    "expected_lift": product_info["config"]["predicted_lift"],
                    "expected_roi": product_info["config"]["roi"],
                    "promotion_type": "DISCOUNT",
                }
            )

            remaining_budget -= allocation

        session.close()

        # Calculate totals
        total_allocated = sum(a["allocated_budget"] for a in allocations)
        avg_roi = (
            sum(a["expected_roi"] for a in allocations) / len(allocations)
            if allocations
            else 0
        )
        avg_lift = (
            sum(a["expected_lift"] for a in allocations) / len(allocations)
            if allocations
            else 0
        )

        return {
            "total_budget": total_budget,
            "allocated_budget": total_allocated,
            "recommendations": allocations,
            "expected_roi": round(avg_roi, 1),
            "expected_lift": round(avg_lift, 1),
            "optimization_score": round(min(avg_roi / 120, 1.0), 2),  # Normalized score
        }

    def _get_average_price(self, product_id: int, session: Session) -> float:
        """Get average price for a product"""
        avg_price = (
            session.query(func.avg(Transaction.sales_value / Transaction.quantity))
            .filter(Transaction.product_id == product_id)
            .scalar()
        )

        return float(avg_price) if avg_price else 5.0

    def _estimate_baseline_units(
        self, product_id: int, duration_days: int, session: Session
    ) -> float:
        """Estimate baseline units sold during promotion period"""
        daily_avg = (
            session.query(func.avg(Transaction.quantity))
            .filter(Transaction.product_id == product_id)
            .scalar()
        )

        daily_units = float(daily_avg) if daily_avg else 10.0
        return daily_units * duration_days


class MLPredictorService:
    """Main service orchestrating all ML prediction capabilities"""

    def __init__(self, models_path: str = "data/models"):
        self.models_path = models_path
        self.lift_predictor = None
        self.roi_optimizer = None
        self.is_initialized = False

    def initialize(self, retrain: bool = False) -> None:
        """Initialize ML models"""

        os.makedirs(self.models_path, exist_ok=True)

        # Initialize lift predictor
        self.lift_predictor = PromotionLiftPredictor("xgboost")

        lift_model_path = os.path.join(self.models_path, "lift_predictor.pkl")

        if os.path.exists(lift_model_path) and not retrain:
            # Load existing model
            self.lift_predictor.load_model(lift_model_path)
        else:
            # Train new model
            print("Training promotion lift predictor...")
            try:
                metrics = self.lift_predictor.train()
                print(
                    f"Model trained successfully - R²: {metrics.r2:.3f}, RMSE: {metrics.rmse:.2f}"
                )
                self.lift_predictor.save_model(lift_model_path)
            except Exception as e:
                print(f"Warning: Could not train model - {e}")
                print("Using pre-trained fallback logic")

        # Initialize ROI optimizer
        self.roi_optimizer = ROIOptimizer(self.lift_predictor)

        self.is_initialized = True
        print("ML Predictor Service initialized successfully")

    def predict_lift(self, **kwargs) -> PredictionResult:
        """Predict promotion lift"""
        if not self.is_initialized:
            self.initialize()

        return self.lift_predictor.predict(**kwargs)

    def optimize_budget(self, **kwargs) -> Dict:
        """Optimize budget allocation"""
        if not self.is_initialized:
            self.initialize()

        return self.roi_optimizer.optimize_budget_allocation(**kwargs)

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "service_initialized": self.is_initialized,
            "lift_predictor_trained": (
                self.lift_predictor.is_trained if self.lift_predictor else False
            ),
            "models_path": self.models_path,
            "available_models": ["promotion_lift_predictor", "roi_optimizer"],
        }


# Global service instance
ml_service = MLPredictorService()


def get_ml_service() -> MLPredictorService:
    """Get the global ML predictor service instance"""
    if not ml_service.is_initialized:
        ml_service.initialize()
    return ml_service
