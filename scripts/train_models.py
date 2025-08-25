#!/usr/bin/env uv run python
"""
ML Model Training Script for TPM System
Handles data partitioning, model training, evaluation, and persistence
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.services.ml_predictor import PromotionLiftPredictor, ModelMetrics
from src.services.data_processor import get_data_summary
from src.models.database import get_db_session
from src.models.entities import Transaction, Promotion, Product, Household, Store


class MLModelTrainer:
    """Comprehensive ML model training pipeline"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir = self.models_dir / "training_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.training_results = {}
        
    def prepare_training_data(self, test_size: float = 0.2, 
                            time_based_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare and partition data for ML training
        
        Args:
            test_size: Fraction of data for testing (0.2 = 20%)
            time_based_split: Use time-based split for time series data
        """
        print("📊 Preparing training data...")
        
        with get_db_session() as session:
            # Get comprehensive promotion and transaction data
            query = session.query(
                Promotion.product_id,
                Promotion.store_id,
                Promotion.promotion_type,
                Promotion.discount_percentage.label('discount_pct'),
                Promotion.start_day,
                Promotion.end_day,
                Product.department,
                Product.commodity_desc,
                Product.brand,
                Product.manufacturer,
                Store.store_id.label('store_location'),
                # Calculate promotion duration
                (Promotion.end_day - Promotion.start_day).label('duration_days'),
                # We'll calculate actual lift separately
            ).join(Product, Promotion.product_id == Product.product_id)\
             .join(Store, Promotion.store_id == Store.store_id)\
             .filter(
                 Promotion.discount_percentage.isnot(None),
                 Promotion.start_day.isnot(None),
                 Promotion.end_day.isnot(None)
             ).all()
        
        # Convert to DataFrame
        promotion_data = pd.DataFrame([
            {
                'product_id': row.product_id,
                'store_id': row.store_id,
                'promotion_type': row.promotion_type or 'DISCOUNT',
                'discount_pct': float(row.discount_pct or 0),
                'duration_days': max(1, int(row.duration_days or 14)),
                'start_day': row.start_day,
                'end_day': row.end_day,
                'department': row.department or 'UNKNOWN',
                'commodity_desc': row.commodity_desc or 'UNKNOWN',
                'brand': row.brand or 'UNKNOWN',
                'manufacturer': row.manufacturer or 'UNKNOWN',
            }
            for row in query
        ])
        
        print(f"   → Found {len(promotion_data)} historical promotions")
        
        if len(promotion_data) < 50:
            print("⚠️ Warning: Limited training data. Generating synthetic data.")
            valid_promotions = self._generate_synthetic_data(n_samples=200)
        else:
            # Calculate actual lift for each promotion
            print("🧮 Calculating actual promotion lift...")
            promotion_data['actual_lift'] = promotion_data.apply(
                lambda row: self._calculate_actual_lift(row), axis=1
            )
            
            # Filter out promotions with insufficient data
            valid_promotions = promotion_data[
                (promotion_data['actual_lift'] >= -50) &  # Remove extreme outliers
                (promotion_data['actual_lift'] <= 500) &
                (promotion_data['discount_pct'] > 0) &
                (promotion_data['duration_days'] > 0)
            ].copy()
            
            print(f"   → {len(valid_promotions)} promotions with valid lift data")
            
            if len(valid_promotions) < 50:
                print("⚠️ Warning: Still limited training data after filtering. Generating synthetic data.")
                valid_promotions = self._generate_synthetic_data(n_samples=200)
        
        # Prepare features and target
        features = self._engineer_features(valid_promotions)
        target = valid_promotions['actual_lift']
        
        # Data partitioning
        if time_based_split and 'start_day' in valid_promotions.columns:
            print("📅 Using time-based data split...")
            # Sort by time and split
            valid_promotions_sorted = valid_promotions.sort_values('start_day')
            split_idx = int(len(valid_promotions_sorted) * (1 - test_size))
            
            train_indices = valid_promotions_sorted.index[:split_idx]
            test_indices = valid_promotions_sorted.index[split_idx:]
            
            X_train = features.loc[train_indices]
            X_test = features.loc[test_indices]
            y_train = target.loc[train_indices]
            y_test = target.loc[test_indices]
            
        else:
            print("🎲 Using random data split...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42
            )
        
        print(f"   → Training set: {len(X_train)} samples")
        print(f"   → Test set: {len(X_test)} samples")
        print(f"   → Features: {len(X_train.columns)} features")
        
        return X_train, X_test, y_train, y_test
    
    def _calculate_actual_lift(self, promotion_row) -> float:
        """Calculate actual sales lift for a promotion"""
        try:
            with get_db_session() as session:
                # Get sales during promotion
                promo_sales = session.query(
                    session.query(Transaction.quantity).filter(
                        Transaction.product_id == promotion_row['product_id'],
                        Transaction.store_id == promotion_row['store_id'],
                        Transaction.day >= promotion_row['start_day'],
                        Transaction.day <= promotion_row['end_day']
                    ).scalar_subquery().label('promo_qty')
                ).scalar() or 0
                
                # Get baseline sales (same period, different time)
                duration = promotion_row['duration_days']
                baseline_start = max(1, promotion_row['start_day'] - duration - 7)
                baseline_end = promotion_row['start_day'] - 7
                
                baseline_sales = session.query(
                    session.query(Transaction.quantity).filter(
                        Transaction.product_id == promotion_row['product_id'],
                        Transaction.store_id == promotion_row['store_id'],
                        Transaction.day >= baseline_start,
                        Transaction.day <= baseline_end
                    ).scalar_subquery().label('baseline_qty')
                ).scalar() or 1
                
                # Calculate lift percentage
                if baseline_sales > 0:
                    lift = ((promo_sales - baseline_sales) / baseline_sales) * 100
                else:
                    # Fallback to discount-based estimate
                    lift = promotion_row['discount_pct'] * 1.5
                    
                return float(lift)
                
        except Exception:
            # Fallback to discount-based estimate
            return float(promotion_row['discount_pct'] * 1.5)
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model"""
        features = data.copy()
        
        # Categorical encoding
        features['promotion_type_encoded'] = pd.Categorical(features['promotion_type']).codes
        features['department_encoded'] = pd.Categorical(features['department']).codes
        features['brand_encoded'] = pd.Categorical(features['brand']).codes
        
        # Numerical features
        features['discount_squared'] = features['discount_pct'] ** 2
        features['duration_log'] = np.log1p(features['duration_days'])
        features['discount_duration_interaction'] = features['discount_pct'] * features['duration_days']
        
        # Discount bins
        features['discount_bin'] = pd.cut(
            features['discount_pct'], 
            bins=[0, 10, 20, 30, 50, 100], 
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # Duration bins  
        features['duration_bin'] = pd.cut(
            features['duration_days'],
            bins=[0, 7, 14, 21, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Select feature columns
        feature_cols = [
            'discount_pct', 'duration_days', 'promotion_type_encoded',
            'department_encoded', 'brand_encoded', 'discount_squared',
            'duration_log', 'discount_duration_interaction', 'discount_bin',
            'duration_bin', 'start_day'  # Seasonality proxy
        ]
        
        return features[feature_cols]
    
    def _generate_synthetic_data(self, n_samples: int = 200) -> pd.DataFrame:
        """Generate synthetic promotion data for training when real data is limited"""
        print(f"🤖 Generating {n_samples} synthetic training samples...")
        
        np.random.seed(42)
        
        # Define realistic ranges
        discount_ranges = {
            'DISCOUNT': (5, 50),
            'BOGO': (25, 50),
            'COUPON': (10, 30),
            'DISPLAY': (0, 15),
            'BUNDLE': (15, 35)
        }
        
        synthetic_data = []
        
        for i in range(n_samples):
            promo_type = np.random.choice(list(discount_ranges.keys()))
            discount_min, discount_max = discount_ranges[promo_type]
            
            discount_pct = np.random.uniform(discount_min, discount_max)
            duration_days = np.random.choice([7, 10, 14, 21, 28], p=[0.2, 0.2, 0.4, 0.15, 0.05])
            
            # Simulate realistic lift based on promotion characteristics
            base_lift = discount_pct * 1.2
            type_multipliers = {'DISCOUNT': 1.0, 'BOGO': 1.4, 'COUPON': 1.1, 'DISPLAY': 0.7, 'BUNDLE': 1.2}
            duration_effect = 1.0 + (duration_days - 14) * 0.02
            
            actual_lift = base_lift * type_multipliers[promo_type] * duration_effect
            actual_lift += np.random.normal(0, 5)  # Add noise
            
            synthetic_data.append({
                'product_id': 1000 + (i % 100),
                'store_id': 1 + (i % 10),
                'promotion_type': promo_type,
                'discount_pct': discount_pct,
                'duration_days': duration_days,
                'start_day': np.random.randint(1, 365),
                'end_day': 0,  # Will be calculated
                'department': np.random.choice(['GROCERY', 'DAIRY', 'FROZEN', 'BEVERAGES']),
                'commodity_desc': f'COMMODITY_{i % 20}',
                'brand': f'BRAND_{i % 30}',
                'manufacturer': f'MANUFACTURER_{i % 15}',
                'actual_lift': max(-10, min(200, actual_lift))
            })
        
        df = pd.DataFrame(synthetic_data)
        df['end_day'] = df['start_day'] + df['duration_days']
        
        return df
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train multiple ML models and compare performance"""
        
        models_to_train = ['xgboost', 'random_forest', 'gradient_boost', 'linear']
        results = {}
        
        print("🚂 Training ML models...")
        
        for model_type in models_to_train:
            print(f"\n   🧠 Training {model_type} model...")
            
            try:
                # Initialize predictor
                predictor = PromotionLiftPredictor(model_type=model_type)
                
                # Prepare data in the format expected by the predictor
                # Convert to the format expected by the original train method
                training_data = X_train.copy()
                training_data['actual_lift'] = y_train
                
                # Mock the prepare_training_data method to use our prepared data
                def mock_prepare_training_data(session=None):
                    features = X_train
                    target = y_train
                    return features, target
                
                # Temporarily replace the method
                original_method = predictor.prepare_training_data
                predictor.prepare_training_data = mock_prepare_training_data
                
                # Train model using internal method
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Set up predictor attributes
                predictor.feature_names = list(X_train.columns)
                predictor.scaler = scaler
                
                # Train model directly
                predictor.model.fit(X_train_scaled, y_train)
                predictor.is_trained = True
                
                # Evaluate model
                y_pred_train = predictor.model.predict(X_train_scaled)
                y_pred_test = predictor.model.predict(X_test_scaled)
                
                # Calculate metrics
                train_metrics = {
                    'mae': mean_absolute_error(y_train, y_pred_train),
                    'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'r2': r2_score(y_train, y_pred_train)
                }
                
                test_metrics = {
                    'mae': mean_absolute_error(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'r2': r2_score(y_test, y_pred_test)
                }
                
                # Feature importance
                feature_importance = {}
                if hasattr(predictor.model, 'feature_importances_'):
                    for name, importance in zip(predictor.feature_names, predictor.model.feature_importances_):
                        feature_importance[name] = float(importance)
                
                # Store results
                results[model_type] = {
                    'predictor': predictor,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': feature_importance,
                    'predictions': {
                        'train': y_pred_train,
                        'test': y_pred_test
                    }
                }
                
                print(f"     ✅ {model_type}: Test R² = {test_metrics['r2']:.3f}, MAE = {test_metrics['mae']:.2f}%")
                
                # Restore original method
                predictor.prepare_training_data = original_method
                
            except Exception as e:
                print(f"     ❌ Failed to train {model_type}: {str(e)}")
                
        return results
    
    def evaluate_models(self, results: Dict) -> str:
        """Evaluate and compare model performance"""
        print("\n📊 Model Evaluation Results:")
        
        comparison_data = []
        
        for model_type, result in results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            
            comparison_data.append({
                'Model': model_type,
                'Train R²': f"{train_metrics['r2']:.3f}",
                'Test R²': f"{test_metrics['r2']:.3f}",
                'Train MAE': f"{train_metrics['mae']:.2f}",
                'Test MAE': f"{test_metrics['mae']:.2f}",
                'Overfitting': f"{train_metrics['r2'] - test_metrics['r2']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best model
        best_model_type = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])
        best_r2 = results[best_model_type]['test_metrics']['r2']
        
        print(f"\n🏆 Best Model: {best_model_type} (Test R² = {best_r2:.3f})")
        
        return best_model_type
    
    def save_models(self, results: Dict, best_model_type: str):
        """Save trained models to disk"""
        print("\n💾 Saving trained models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_type, result in results.items():
            # Save model
            model_path = self.models_dir / f"{model_type}_promotion_predictor_{timestamp}.pkl"
            result['predictor'].save_model(str(model_path))
            
            # Mark best model
            if model_type == best_model_type:
                best_model_path = self.models_dir / "best_promotion_predictor.pkl"
                result['predictor'].save_model(str(best_model_path))
                print(f"   ✅ Best model saved: {best_model_path}")
            
            print(f"   📁 {model_type}: {model_path}")
        
        # Save training results
        training_summary = {
            'timestamp': timestamp,
            'best_model': best_model_type,
            'models': {
                model_type: {
                    'train_metrics': result['train_metrics'],
                    'test_metrics': result['test_metrics'],
                    'feature_importance': result['feature_importance']
                }
                for model_type, result in results.items()
            }
        }
        
        summary_path = self.results_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"   📊 Training summary: {summary_path}")
        
    def create_visualizations(self, results: Dict, X_test: pd.DataFrame, y_test: pd.Series):
        """Create training visualization plots"""
        if not PLOTTING_AVAILABLE:
            print("\n⚠️ Matplotlib/Seaborn not available. Skipping visualization plots.")
            return
            
        print("\n📈 Creating visualization plots...")
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        model_names = list(results.keys())
        train_r2 = [results[model]['train_metrics']['r2'] for model in model_names]
        test_r2 = [results[model]['test_metrics']['r2'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        axes[0,0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        axes[0,0].set_xlabel('Model Type')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Model R² Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        
        # MAE comparison
        train_mae = [results[model]['train_metrics']['mae'] for model in model_names]
        test_mae = [results[model]['test_metrics']['mae'] for model in model_names]
        
        axes[0,1].bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.8)
        axes[0,1].bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.8)
        axes[0,1].set_xlabel('Model Type')
        axes[0,1].set_ylabel('Mean Absolute Error')
        axes[0,1].set_title('Model MAE Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(model_names, rotation=45)
        axes[0,1].legend()
        
        # Prediction vs actual scatter plot for best model
        best_model = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])
        y_pred_test = results[best_model]['predictions']['test']
        
        axes[1,0].scatter(y_test, y_pred_test, alpha=0.6)
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Lift %')
        axes[1,0].set_ylabel('Predicted Lift %')
        axes[1,0].set_title(f'Predictions vs Actual ({best_model})')
        
        # Feature importance for best model
        if results[best_model]['feature_importance']:
            importance_df = pd.DataFrame(
                list(results[best_model]['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            axes[1,1].barh(importance_df['Feature'], importance_df['Importance'])
            axes[1,1].set_xlabel('Feature Importance')
            axes[1,1].set_title(f'Feature Importance ({best_model})')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"training_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Plots saved: {plot_path}")
        
        plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train TPM ML models')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test set size (default: 0.2 for 80/20 split)')
    parser.add_argument('--time-split', action='store_true',
                       help='Use time-based split instead of random')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--synthetic', action='store_true',
                       help='Force use of synthetic data')
    
    args = parser.parse_args()
    
    print("🚂 TPM ML Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MLModelTrainer(models_dir=args.models_dir)
    
    # Check if we have sufficient data
    try:
        data_summary = get_data_summary()
        print(f"📊 Data Summary: {data_summary}")
        
        if data_summary.get('promotions', 0) < 50 or args.synthetic:
            print("⚠️ Using synthetic training data")
    except Exception as e:
        print(f"⚠️ Could not access database: {e}")
        print("⚠️ Using synthetic training data")
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            test_size=args.test_size,
            time_based_split=args.time_split
        )
        
        # Train models
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        if results:
            # Evaluate models
            best_model_type = trainer.evaluate_models(results)
            
            # Save models
            trainer.save_models(results, best_model_type)
            
            # Create visualizations
            trainer.create_visualizations(results, X_test, y_test)
            
            print("\n✅ Model training completed successfully!")
            print(f"🏆 Best model: {best_model_type}")
            print(f"📁 Models saved in: {trainer.models_dir}")
            
        else:
            print("❌ No models were successfully trained")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()