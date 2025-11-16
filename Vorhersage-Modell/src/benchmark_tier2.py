"""End-to-end benchmarking: compare models before/after improvements.

Benchmarks:
1. Baseline RF without feature engineering (900 samples)
2. Baseline RF with feature engineering (900 samples)
3. Baseline RF on larger dataset (5000 samples)
4. Multimodal model (900 samples)
5. Multimodal model on larger dataset (5000 samples)

Usage:
    python3 src/benchmark_tier2.py --quick  # 2-fold CV, 5 epochs
    python3 src/benchmark_tier2.py --full   # 5-fold CV, 20 epochs (longer)
"""
import argparse
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
from feature_engineering import FeatureEngineer
from multimodal_model import GameStateImageDataset, train_epoch, evaluate

# ============================================================================
# Benchmark Configuration
# ============================================================================

SMALL_DATASET_PATH = 'data/processed/call_states_synth_large.csv'
LARGE_DATASET_PATH = 'data/processed/call_states_large.csv'
IMAGE_DIR = 'data/dummy_screenshots'
IMAGE_META = os.path.join(IMAGE_DIR, 'metadata.jsonl')

# ============================================================================
# Baseline Benchmark (RF)
# ============================================================================

def benchmark_baseline(
    data_path: str,
    feature_engineering: bool = False,
    cv_folds: int = 5,
    name: str = 'baseline'
) -> dict:
    """Benchmark baseline Random Forest model.
    
    Args:
        data_path: Path to CSV
        feature_engineering: Whether to use feature engineering
        cv_folds: Number of CV folds
        name: Benchmark name for logging
    
    Returns:
        Dict with benchmark results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK: {name} (FE={feature_engineering})")
    logger.info(f"{'='*80}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    target = 'best_call'
    X = df.drop(columns=[target])
    y = df[target]
    
    # ====== Feature Engineering ======
    if feature_engineering:
        engineer = FeatureEngineer()
        X = engineer.fit_transform(X)
        pipeline = Pipeline([('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))])
    else:
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, num_cols),
            ('cat', categorical_pipeline, cat_cols)
        ])
        
        pipeline = Pipeline([
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
        ])
    
    # ====== Cross-Validation ======
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': [],
        'macro_f1': [],
        'macro_precision': [],
        'macro_recall': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['macro_f1'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['macro_precision'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['macro_recall'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        
        logger.info(f"  Fold {fold}: Accuracy={metrics['accuracy'][-1]:.4f}, F1={metrics['macro_f1'][-1]:.4f}")
    
    # ====== Summary ======
    result = {
        'name': name,
        'type': 'baseline_rf',
        'feature_engineering': feature_engineering,
        'dataset': os.path.basename(data_path),
        'n_samples': len(df),
        'cv_folds': cv_folds,
        'accuracy': {
            'mean': float(np.mean(metrics['accuracy'])),
            'std': float(np.std(metrics['accuracy']))
        },
        'macro_f1': {
            'mean': float(np.mean(metrics['macro_f1'])),
            'std': float(np.std(metrics['macro_f1']))
        },
        'macro_precision': {
            'mean': float(np.mean(metrics['macro_precision'])),
            'std': float(np.std(metrics['macro_precision']))
        },
        'macro_recall': {
            'mean': float(np.mean(metrics['macro_recall'])),
            'std': float(np.std(metrics['macro_recall']))
        }
    }
    
    logger.info(f"✓ {name}:")
    logger.info(f"  Accuracy:  {result['accuracy']['mean']:.4f} ± {result['accuracy']['std']:.4f}")
    logger.info(f"  Macro F1:  {result['macro_f1']['mean']:.4f} ± {result['macro_f1']['std']:.4f}")
    
    return result

# ============================================================================
# Multimodal Benchmark
# ============================================================================

def benchmark_multimodal(
    data_path: str,
    image_dir: str,
    image_meta_path: str,
    cv_folds: int = 5,
    epochs: int = 20,
    name: str = 'multimodal'
) -> dict:
    """Benchmark multimodal PyTorch model.
    
    Args:
        data_path: Path to CSV
        image_dir: Directory with images
        image_meta_path: Path to image metadata.jsonl
        cv_folds: Number of CV folds
        epochs: Max epochs per fold
        name: Benchmark name
    
    Returns:
        Dict with benchmark results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK: {name}")
    logger.info(f"{'='*80}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    target = 'best_call'
    X = df.drop(columns=[target])
    y = df[target]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ====== Cross-Validation ======
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': [],
        'macro_f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Import here to avoid GPU issues
        from multimodal_model import MultimodalCNN
        
        train_dataset = GameStateImageDataset(df_train, image_dir, image_meta_path, scaler=None)
        val_dataset = GameStateImageDataset(df_val, image_dir, image_meta_path, scaler=train_dataset.scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        model = MultimodalCNN(
            num_classes=len(train_dataset.ALLOWED_CLASSES),
            num_tab_features=len(train_dataset.TABULAR_FEATURES)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_f1 = -1.0
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % 2 == 0:
                val_acc, val_f1, val_loss, _, _ = evaluate(model, val_loader, device)
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"    Early stopping at epoch {epoch+1}")
                        break
        
        metrics['accuracy'].append(val_acc)
        metrics['macro_f1'].append(best_f1)
        logger.info(f"  Fold {fold}: Accuracy={val_acc:.4f}, F1={best_f1:.4f}")
        
        del model, train_loader, val_loader
    
    # ====== Summary ======
    result = {
        'name': name,
        'type': 'multimodal_pytorch',
        'feature_engineering': False,
        'dataset': os.path.basename(data_path),
        'n_samples': len(df),
        'cv_folds': cv_folds,
        'epochs': epochs,
        'accuracy': {
            'mean': float(np.mean(metrics['accuracy'])),
            'std': float(np.std(metrics['accuracy']))
        },
        'macro_f1': {
            'mean': float(np.mean(metrics['macro_f1'])),
            'std': float(np.std(metrics['macro_f1']))
        }
    }
    
    logger.info(f"✓ {name}:")
    logger.info(f"  Accuracy:  {result['accuracy']['mean']:.4f} ± {result['accuracy']['std']:.4f}")
    logger.info(f"  Macro F1:  {result['macro_f1']['mean']:.4f} ± {result['macro_f1']['std']:.4f}")
    
    return result

# ============================================================================
# Main Benchmarking
# ============================================================================

def main(args):
    """Run full benchmarking suite."""
    
    # Generate large dataset if not exists
    if not os.path.exists(LARGE_DATASET_PATH):
        logger.info(f"Large dataset not found. Generating {args.large_n} samples...")
        from generate_large_dataset import generate_large
        df_large = generate_large(n=args.large_n, seed=42)
        os.makedirs(os.path.dirname(LARGE_DATASET_PATH), exist_ok=True)
        df_large.to_csv(LARGE_DATASET_PATH, index=False)
        logger.info(f"Saved to {LARGE_DATASET_PATH}")
    
    cv_folds = 2 if args.quick else 5
    epochs = 5 if args.quick else 20
    
    results = []
    
    # 1. Baseline without FE on small dataset
    result = benchmark_baseline(
        SMALL_DATASET_PATH,
        feature_engineering=False,
        cv_folds=cv_folds,
        name='Baseline RF (900 samples, no FE)'
    )
    if result:
        results.append(result)
    
    # 2. Baseline with FE on small dataset
    result = benchmark_baseline(
        SMALL_DATASET_PATH,
        feature_engineering=True,
        cv_folds=cv_folds,
        name='Baseline RF (900 samples, +FE)'
    )
    if result:
        results.append(result)
    
    # 3. Baseline on large dataset
    result = benchmark_baseline(
        LARGE_DATASET_PATH,
        feature_engineering=True,
        cv_folds=cv_folds,
        name='Baseline RF (5000+ samples, +FE)'
    )
    if result:
        results.append(result)
    
    # 4. Multimodal on small dataset
    try:
        result = benchmark_multimodal(
            SMALL_DATASET_PATH,
            IMAGE_DIR,
            IMAGE_META,
            cv_folds=cv_folds,
            epochs=epochs,
            name='Multimodal PyTorch (900 samples)'
        )
        if result:
            results.append(result)
    except Exception as e:
        logger.warning(f"Multimodal benchmark failed: {e}")
    
    # 5. Multimodal on large dataset
    try:
        result = benchmark_multimodal(
            LARGE_DATASET_PATH,
            IMAGE_DIR,
            IMAGE_META,
            cv_folds=cv_folds,
            epochs=epochs,
            name='Multimodal PyTorch (5000+ samples)'
        )
        if result:
            results.append(result)
    except Exception as e:
        logger.warning(f"Multimodal benchmark on large dataset failed: {e}")
    
    # ========== Summary Report ==========
    logger.info(f"\n\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}\n")
    
    summary_df = pd.DataFrame([
        {
            'Model': r['name'],
            'Type': r['type'],
            'Samples': r['n_samples'],
            'Accuracy': f"{r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f}",
            'Macro F1': f"{r['macro_f1']['mean']:.4f}±{r['macro_f1']['std']:.4f}"
        }
        for r in results
    ])
    
    print(summary_df.to_string(index=False))
    
    # Save detailed results
    output_file = f"models/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('models', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tier 2 benchmark suite')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (2-fold CV, 5 epochs)')
    parser.add_argument('--full', action='store_true', help='Full benchmark (5-fold CV, 20 epochs)')
    parser.add_argument('--large-n', type=int, default=5000, help='Size of large dataset to generate')
    args = parser.parse_args()
    
    if not args.quick and not args.full:
        args.quick = True  # Default to quick
    
    main(args)
