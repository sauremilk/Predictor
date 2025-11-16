"""Feature engineering and importance analysis for best_call prediction.

Implements:
1. Feature interactions (ratios, squares, interactions)
2. Statistical transforms (z-score, log)
3. Domain-specific engineering (health ratios, position clusters)
4. Permutation importance analysis
5. Sklearn-compatible transformer wrapper

Usage:
    from feature_engineering import FeatureEngineer, FeatureEngineeringTransformer
    
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df_train)
    df_test_engineered = engineer.transform(df_test)
    
    # For use in sklearn pipelines:
    from sklearn.pipeline import Pipeline
    transformer = FeatureEngineeringTransformer(engineer)
    transformer.fit(df_train)
    pipeline = Pipeline([('fe', transformer), ('clf', model)])
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngineer:
    """Engineering and scaling of features for best_call prediction."""
    
    BASE_NUMERIC_FEATURES = [
        'zone_index', 'alive_players', 'teammates_alive',
        'storm_edge_dist', 'mats_total', 'surge_above',
        'outcome_placement', 'outcome_alive_time'
    ]
    
    BASE_CATEGORICAL_FEATURES = [
        'zone_phase', 'height_status', 'position_type'
    ]
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit feature engineering pipeline (scaler)."""
        numeric_df = self._create_engineered_features(df)
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_df[self._get_numeric_feature_names(numeric_df)])
        self.feature_names = numeric_df.columns.tolist()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (requires fit first)."""
        numeric_df = self._create_engineered_features(df)
        
        feature_cols = self._get_numeric_feature_names(numeric_df)
        numeric_df[feature_cols] = self.scaler.transform(numeric_df[feature_cols])
        
        return numeric_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and derived features."""
        df_eng = df[self.BASE_NUMERIC_FEATURES + self.BASE_CATEGORICAL_FEATURES].copy()
        
        # ====== Ratio & Interaction Features ======
        # Health ratios
        df_eng['health_ratio'] = df_eng['alive_players'] / (df_eng['teammates_alive'] + 1e-6)
        
        # Storm threat (closer to edge = more threat)
        df_eng['storm_threat'] = 1.0 / (df_eng['storm_edge_dist'] + 1.0)
        
        # Resource efficiency
        df_eng['mats_per_player'] = df_eng['mats_total'] / (df_eng['alive_players'] + 1.0)
        
        # Elevation advantage
        df_eng['surge_above_normalized'] = df_eng['surge_above'] / (df_eng['mats_total'] + 1.0)
        
        # Zone progression
        df_eng['zone_index_squared'] = df_eng['zone_index'] ** 2
        
        # Survival pressure (placement vs alive time)
        df_eng['placement_time_ratio'] = df_eng['outcome_placement'] / (df_eng['outcome_alive_time'] + 1e-6)
        
        # Interaction: zone index × storm threat
        df_eng['zone_storm_interaction'] = df_eng['zone_index'] * df_eng['storm_threat']
        
        # Interaction: alive players × mats
        df_eng['players_mats_interaction'] = df_eng['alive_players'] * (df_eng['mats_total'] / 1000.0)
        
        # ====== Log Transforms ======
        df_eng['log_mats'] = np.log1p(df_eng['mats_total'])
        df_eng['log_outcome_placement'] = np.log1p(df_eng['outcome_placement'])
        df_eng['log_outcome_alive_time'] = np.log1p(df_eng['outcome_alive_time'])
        
        # ====== Remove Categorical Columns (will be handled upstream or dropped) ======
        df_eng = df_eng.drop(columns=self.BASE_CATEGORICAL_FEATURES, errors='ignore')
        
        return df_eng
    
    def _get_numeric_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get numeric feature column names (all except categorical)."""
        return [col for col in df.columns if col not in self.BASE_CATEGORICAL_FEATURES]
    
    def get_feature_names(self) -> List[str]:
        """Return list of all engineered feature names."""
        if self.feature_names is None:
            raise RuntimeError("Fit engineer first with fit() or fit_transform()")
        return self.feature_names

# ============================================================================
# Feature Importance Analysis
# ============================================================================

def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, float]:
    """Compute permutation importance for model predictions.
    
    Args:
        model: Trained sklearn classifier with predict_proba()
        X: Feature dataframe
        y: Target labels
        n_repeats: Permutation repeats
        random_state: Random seed
    
    Returns:
        Dict of feature_name -> importance score
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    importances = {
        name: float(result.importances_mean[i])
        for i, name in enumerate(X.columns)
    }
    
    return importances

def analyze_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str = 'models/feature_importance.json'
) -> Dict[str, float]:
    """Analyze feature importance and save results.
    
    Args:
        model: Trained classifier
        X: Feature dataframe
        y: Target labels
        output_path: Path to save JSON results
    
    Returns:
        Dict of feature_name -> importance score
    """
    import json
    
    logger.info("Computing permutation importance...")
    importances = compute_permutation_importance(model, X, y)
    
    # Sort by importance
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    # Log results
    logger.info("Feature Importance Ranking:")
    for feature, importance in sorted_importances.items():
        logger.info(f"  {feature:35s}: {importance:+.6f}")
    
    # Save
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sorted_importances, f, indent=2)
    
    logger.info(f"Saved to {output_path}")
    
    return sorted_importances

# ============================================================================
# Sklearn Transformer Wrapper
# ============================================================================

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer wrapping FeatureEngineer for use in pipelines.
    
    Allows FeatureEngineer to be used as a step in sklearn Pipeline objects.
    
    Example:
        from sklearn.pipeline import Pipeline
        from feature_engineering import FeatureEngineeringTransformer
        
        transformer = FeatureEngineeringTransformer()
        transformer.fit(X_train)
        X_transformed = transformer.transform(X_test)
    """

    def __init__(self, engineer=None):
        """Initialize with optional pre-created FeatureEngineer instance."""
        self.engineer = engineer if engineer is not None else FeatureEngineer()
        self.fitted = False

    def fit(self, X, y=None):
        """Fit the underlying FeatureEngineer on training data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.engineer.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform raw features into engineered features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not self.fitted:
            raise RuntimeError('Transformer not fitted. Call fit() first.')
        return self.engineer.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return engineered feature names (for sklearn 1.0+)."""
        if not self.fitted:
            raise RuntimeError('Transformer not fitted.')
        return np.array(self.engineer.get_feature_names())

# ============================================================================
# Data Augmentation (optional)
# ============================================================================

def augment_with_synthetic_minority(
    df: pd.DataFrame,
    target_col: str = 'best_call',
    random_state: int = 42
) -> pd.DataFrame:
    """Simple oversampling of minority classes.
    
    Note: For advanced SMOTE, use imblearn.over_sampling.SMOTE
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        random_state: Random seed
    
    Returns:
        DataFrame with oversampled rows
    """
    np.random.seed(random_state)
    
    class_counts = df[target_col].value_counts()
    max_count = class_counts.max()
    
    frames = [df]
    for class_label in class_counts.index:
        class_df = df[df[target_col] == class_label]
        n_missing = max_count - len(class_df)
        
        if n_missing > 0:
            indices = np.random.choice(class_df.index, size=n_missing, replace=True)
            augmented = df.loc[indices].copy()
            frames.append(augmented)
    
    df_augmented = pd.concat(frames, ignore_index=True)
    logger.info(f"Augmented dataset: {len(df)} → {len(df_augmented)} rows")
    
    return df_augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)
