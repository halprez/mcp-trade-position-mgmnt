"""
Model Management Service for TPM System
Handles loading, caching, and serving trained ML models
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import logging

from .ml_predictor import PromotionLiftPredictor, PredictionResult

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages trained ML models for the TPM system"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model cache
        self._loaded_models: Dict[str, PromotionLiftPredictor] = {}
        self._model_metadata: Dict[str, dict] = {}
        
        # Load best model on initialization
        self._load_best_model()
    
    def _load_best_model(self) -> Optional[PromotionLiftPredictor]:
        """Load the best trained model"""
        best_model_path = self.models_dir / "best_promotion_predictor.pkl"
        
        if best_model_path.exists():
            try:
                predictor = self.load_model(str(best_model_path))
                self._loaded_models["best"] = predictor
                logger.info("Best model loaded successfully")
                return predictor
            except Exception as e:
                logger.warning(f"Failed to load best model: {e}")
        
        logger.warning("No best model found. Models need to be trained.")
        return None
    
    def load_model(self, filepath: Union[str, Path]) -> PromotionLiftPredictor:
        """Load a trained model from disk"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create predictor instance
            predictor = PromotionLiftPredictor(model_type=model_data['model_type'])
            
            # Restore model state
            predictor.model = model_data['model']
            predictor.scaler = model_data['scaler']
            predictor.feature_names = model_data['feature_names']
            predictor.is_trained = model_data['is_trained']
            
            # Store metadata
            self._model_metadata[str(filepath)] = {
                'model_type': model_data['model_type'],
                'loaded_at': datetime.now(),
                'file_size': filepath.stat().st_size,
                'is_trained': model_data['is_trained']
            }
            
            logger.info(f"Model loaded: {filepath.name}")
            return predictor
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise
    
    def get_available_models(self) -> Dict[str, dict]:
        """Get list of available trained models"""
        models = {}
        
        # Scan models directory
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                # Get basic info without loading full model
                stat_info = model_file.stat()
                models[model_file.stem] = {
                    'filename': model_file.name,
                    'path': str(model_file),
                    'size_mb': stat_info.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat_info.st_mtime),
                    'loaded': str(model_file) in self._loaded_models
                }
            except Exception as e:
                logger.warning(f"Could not read model info for {model_file}: {e}")
        
        return models
    
    def predict_promotion_lift(
        self,
        product_name: str,
        discount_percentage: float,
        duration_days: int = 14,
        promotion_type: str = "DISCOUNT",
        store_name: Optional[str] = None,
        model_name: str = "best"
    ) -> Union[PredictionResult, dict]:
        """Make promotion lift prediction using trained model"""
        
        # Check if we have a trained model
        if model_name not in self._loaded_models:
            # Try to load best model if requested
            if model_name == "best":
                self._load_best_model()
            
            if model_name not in self._loaded_models:
                # Fallback to rule-based prediction
                return self._fallback_prediction(
                    product_name, discount_percentage, duration_days, promotion_type
                )
        
        predictor = self._loaded_models[model_name]
        
        try:
            # Create simplified features to match training data
            import pandas as pd
            import numpy as np
            
            # Simple feature engineering that matches training
            features = {
                'discount_pct': discount_percentage,
                'duration_days': duration_days,
                'promotion_type_encoded': {'DISCOUNT': 0, 'BOGO': 1, 'DISPLAY': 2, 'COUPON': 3, 'BUNDLE': 4}.get(promotion_type, 0),
                'department_encoded': hash(product_name) % 10,  # Simple encoding
                'brand_encoded': hash(product_name.split()[0] if ' ' in product_name else product_name) % 20,
                'discount_squared': discount_percentage ** 2,
                'duration_log': np.log1p(duration_days),
                'discount_duration_interaction': discount_percentage * duration_days,
                'discount_bin': min(4, int(discount_percentage // 10)) if discount_percentage <= 50 else 4,
                'duration_bin': 0 if duration_days <= 7 else (1 if duration_days <= 14 else (2 if duration_days <= 21 else 3)),
                'start_day': 237  # Current day of year
            }
            
            # Create DataFrame
            X = pd.DataFrame([features])
            
            # Scale features
            X_scaled = predictor.scaler.transform(X)
            
            # Make prediction
            prediction = predictor.model.predict(X_scaled)[0]
            
            # Get feature importance
            feature_importance = {}
            if hasattr(predictor.model, 'feature_importances_'):
                for name, importance in zip(predictor.feature_names, predictor.model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            # Return PredictionResult-like object
            class SimpleResult:
                def __init__(self, pred):
                    self.prediction = float(pred)
                    self.confidence_lower = max(0, pred - 6)
                    self.confidence_upper = min(100, pred + 6)
                    self.feature_importance = feature_importance
                    self.model_version = f"{predictor.model_type}_trained"
            
            return SimpleResult(prediction)
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}. Using fallback.")
            return self._fallback_prediction(
                product_name, discount_percentage, duration_days, promotion_type
            )
    
    def _fallback_prediction(
        self,
        product_name: str,
        discount_percentage: float,
        duration_days: int,
        promotion_type: str
    ) -> dict:
        """Fallback rule-based prediction when ML model is not available"""
        
        # Simple rule-based prediction (same as current mcp_server.py logic)
        base_lift = 0.15  # 15% base lift
        
        # Adjust for discount depth
        discount_factor = min(discount_percentage / 100 * 2.5, 0.8)
        
        # Adjust for duration (diminishing returns)
        duration_factor = 1.0 + (duration_days - 14) * 0.02
        duration_factor = max(0.8, min(1.3, duration_factor))
        
        # Adjust for promotion type
        type_multipliers = {
            "DISCOUNT": 1.0,
            "BOGO": 1.4,
            "DISPLAY": 0.7,
            "COUPON": 1.1,
            "BUNDLE": 1.2
        }
        type_factor = type_multipliers.get(promotion_type.upper(), 1.0)
        
        # Calculate predicted lift
        predicted_lift = base_lift * (1 + discount_factor) * duration_factor * type_factor
        
        return {
            'prediction': float(predicted_lift * 100),  # Convert to percentage
            'confidence_lower': max(0, (predicted_lift - 0.08) * 100),
            'confidence_upper': min(100, (predicted_lift + 0.08) * 100),
            'feature_importance': {},
            'model_version': 'rule_based_fallback',
            'is_ml_prediction': False
        }
    
    def get_model_info(self, model_name: str = "best") -> dict:
        """Get information about a loaded model"""
        if model_name not in self._loaded_models:
            return {"error": f"Model '{model_name}' not loaded"}
        
        predictor = self._loaded_models[model_name]
        
        return {
            'model_type': predictor.model_type,
            'is_trained': predictor.is_trained,
            'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
            'feature_names': predictor.feature_names or [],
            'loaded_at': self._model_metadata.get(model_name, {}).get('loaded_at'),
        }
    
    def health_check(self) -> dict:
        """Check the health of the model management system"""
        available_models = self.get_available_models()
        loaded_models = list(self._loaded_models.keys())
        
        has_trained_model = any(
            name in self._loaded_models and self._loaded_models[name].is_trained
            for name in self._loaded_models.keys()
        )
        
        return {
            'status': 'healthy' if has_trained_model else 'no_trained_models',
            'available_models': len(available_models),
            'loaded_models': loaded_models,
            'has_best_model': 'best' in loaded_models,
            'models_directory': str(self.models_dir),
            'fallback_available': True  # Rule-based fallback always available
        }
    
    def retrain_recommended(self) -> bool:
        """Check if model retraining is recommended"""
        if 'best' not in self._loaded_models:
            return True
        
        # Check model age
        best_model_path = self.models_dir / "best_promotion_predictor.pkl"
        if best_model_path.exists():
            model_age_days = (datetime.now().timestamp() - best_model_path.stat().st_mtime) / (24 * 3600)
            if model_age_days > 30:  # Recommend retraining after 30 days
                return True
        
        return False


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(models_dir="data/models")
    return _model_manager


def predict_with_ml(
    product_name: str,
    discount_percentage: float,
    duration_days: int = 14,
    promotion_type: str = "DISCOUNT",
    store_name: Optional[str] = None
) -> Union[PredictionResult, dict]:
    """Convenience function for ML predictions"""
    manager = get_model_manager()
    return manager.predict_promotion_lift(
        product_name=product_name,
        discount_percentage=discount_percentage,
        duration_days=duration_days,
        promotion_type=promotion_type,
        store_name=store_name
    )


def get_model_status() -> dict:
    """Get current model status"""
    manager = get_model_manager()
    return manager.health_check()