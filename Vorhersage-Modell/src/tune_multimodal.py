"""Hyperparameter tuning for multimodal model using Optuna.

Optimizes architecture, learning rate, patience, batch size, dropout using Bayesian search.

Usage:
    python3 src/tune_multimodal.py --data data/call_states_real_labeled.csv \
                                   --images data/dummy_screenshots \
                                   --n-trials 20 --cv-folds 3 --epochs 10
"""
import argparse
import os
import json
import logging
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Import multimodal components
import sys
sys.path.insert(0, os.path.dirname(__file__))
from multimodal_model import GameStateImageDataset, train_epoch, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Tunable Architecture
# ============================================================================

class TunableMultimodalCNN(nn.Module):
    """Multimodal model with tunable hyperparameters."""
    
    def __init__(
        self,
        num_classes: int,
        num_tab_features: int,
        conv_filters: int,
        dense_hidden: int,
        dropout_rate: float
    ):
        super().__init__()
        
        # Vision branch: tunable conv filters
        self.vision = nn.Sequential(
            nn.Conv2d(3, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.vision_fc = nn.Linear(conv_filters * 4, 128)
        
        # Tabular branch: tunable hidden size
        self.tabular = nn.Sequential(
            nn.Linear(num_tab_features, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden, dense_hidden // 2),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(128 + dense_hidden // 2, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden, num_classes)
        )
    
    def forward(self, images, tabular):
        x_vis = self.vision(images)
        x_vis = x_vis.view(x_vis.size(0), -1)
        x_vis = self.vision_fc(x_vis)
        x_tab = self.tabular(tabular)
        x_fused = torch.cat([x_vis, x_tab], dim=1)
        logits = self.fusion(x_fused)
        return logits

# ============================================================================
# Objective Function for Optuna
# ============================================================================

def objective(trial: Trial, args) -> float:
    """Objective function for Optuna: maximize validation macro F1."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    conv_filters = trial.suggest_categorical('conv_filters', [16, 32, 64])
    dense_hidden = trial.suggest_categorical('dense_hidden', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    patience = trial.suggest_int('patience', 2, 5)
    eval_interval = trial.suggest_int('eval_interval', 1, 3)
    
    logger.info(f"Trial {trial.number}: LR={lr:.4f}, BS={batch_size}, Conv={conv_filters}, Dense={dense_hidden}, Dropout={dropout_rate:.2f}, Patience={patience}")
    
    # Load data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    df = pd.read_csv(args.data)
    image_meta_path = os.path.join(args.images, 'metadata.jsonl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # StratifiedKFold CV
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    y = df['best_call'].values
    
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        logger.info(f"  Fold {fold+1}/{args.cv_folds}")
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Create datasets
        train_dataset = GameStateImageDataset(df_train, args.images, image_meta_path, scaler=None)
        val_dataset = GameStateImageDataset(df_val, args.images, image_meta_path, scaler=train_dataset.scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Model
        model = TunableMultimodalCNN(
            num_classes=len(train_dataset.ALLOWED_CLASSES),
            num_tab_features=len(train_dataset.TABULAR_FEATURES),
            conv_filters=conv_filters,
            dense_hidden=dense_hidden,
            dropout_rate=dropout_rate
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = -1.0
        patience_counter = 0
        
        # Training
        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % eval_interval == 0:
                val_acc, val_f1, val_loss, preds, labels = evaluate(model, val_loader, device)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"    Early stopping at epoch {epoch+1}")
                        break
            
            # Pruning (intermediate report to Optuna)
            if (epoch + 1) % eval_interval == 0:
                trial.report(best_val_f1, epoch)
                if trial.should_prune():
                    logger.info(f"    Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
        
        fold_f1s.append(best_val_f1)
    
    # Return mean CV F1
    mean_f1 = np.mean(fold_f1s)
    logger.info(f"Trial {trial.number} result: Mean CV F1 = {mean_f1:.4f}")
    
    return mean_f1

# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def run_tuning(args):
    """Run hyperparameter tuning with Optuna."""
    
    logger.info(f"Starting hyperparameter tuning ({args.n_trials} trials, {args.cv_folds}-fold CV)")
    
    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Objective with args bound
    def objective_with_args(trial: Trial) -> float:
        return objective(trial, args)
    
    # Optimize
    study.optimize(
        objective_with_args,
        n_trials=args.n_trials,
        catch=(RuntimeError, ValueError)
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("TUNING RESULTS")
    logger.info("="*80)
    
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value (Mean CV F1): {best_trial.value:.4f}")
    logger.info(f"Best params:")
    for key, val in best_trial.params.items():
        logger.info(f"  {key}: {val}")
    
    # Save results
    results_path = f"models/tuning_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        'best_trial': best_trial.number,
        'best_value': float(best_trial.value),
        'best_params': best_trial.params,
        'all_trials': []
    }
    
    for trial in study.trials:
        results['all_trials'].append({
            'number': trial.number,
            'value': float(trial.value) if trial.value is not None else None,
            'params': trial.params,
            'state': trial.state.name
        })
    
    os.makedirs('models', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    return best_trial.params

# ============================================================================
# Main
# ============================================================================

def main(args):
    best_params = run_tuning(args)
    
    print("\n" + "="*80)
    print("RECOMMENDED HYPERPARAMETERS FOR TRAINING:")
    print("="*80)
    print("\nUse these parameters in your training command:")
    print("\npython3 src/multimodal_model.py \\")
    print(f"  --lr {best_params['learning_rate']:.6f} \\")
    print(f"  --batch-size {best_params['batch_size']} \\")
    print(f"  --patience {best_params['patience']} \\")
    print(f"  --eval-interval {best_params['eval_interval']} \\")
    print("  --epochs 50 \\")
    print("  --tensorboard")
    print("\nNote: Architecture parameters (conv_filters, dense_hidden, dropout_rate)")
    print("      require code modification in src/multimodal_model.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for multimodal model')
    parser.add_argument('--data', type=str, default='data/call_states_real_labeled.csv',
                       help='Path to labeled data CSV')
    parser.add_argument('--images', type=str, default='data/dummy_screenshots',
                       help='Directory with screenshot images')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--cv-folds', type=int, default=3, help='CV folds per trial')
    parser.add_argument('--epochs', type=int, default=15, help='Max epochs per fold')
    args = parser.parse_args()
    main(args)
