"""Multimodal model combining CNN (images) + Tabular features using PyTorch.

Architecture:
  - Vision branch: CNN (ResNet18-like) on screenshots → visual features
  - Tabular branch: Dense network on game-state features → tabular features
  - Fusion: Concatenate both → final classifier

Uses StratifiedKFold CV for evaluation.

Usage:
    python3 src/multimodal_model.py --data data/call_states_real_labeled.csv \
                                    --images data/dummy_screenshots \
                                    --epochs 20 --n-splits 5
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from torchvision import transforms
except Exception:
    # Minimal substitutes if torchvision is not available in the environment.
    from PIL import Image
    import numpy as _np

    class _ToTensor:
        def __call__(self, img: Image.Image):
            arr = _np.array(img).astype(_np.float32) / 255.0
            # HWC -> CHW
            arr = arr.transpose(2, 0, 1)
            return torch.tensor(arr)

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = _np.array(mean).reshape(3, 1, 1)
            self.std = _np.array(std).reshape(3, 1, 1)
        def __call__(self, tensor):
            arr = tensor.numpy()
            arr = (arr - self.mean) / self.std
            return torch.tensor(arr)

    class _Resize:
        def __init__(self, size):
            self.size = size
        def __call__(self, img: Image.Image):
            return img.resize(self.size)

    class transforms:
        @staticmethod
        def Compose(lst):
            def fn(x):
                for t in lst:
                    x = t(x)
                return x
            return fn

        @staticmethod
        def Resize(size):
            return _Resize(size)

        @staticmethod
        def ToTensor():
            return _ToTensor()

        @staticmethod
        def Normalize(mean, std):
            return _Normalize(mean, std)
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class GameStateImageDataset(Dataset):
    """Dataset combining images and tabular features."""
    
    ALLOWED_CLASSES = [
        'stick_deadside', 'play_frontside', 'take_height',
        'stabilize_box', 'look_for_refresh', 'drop_low'
    ]
    
    # Only numeric features; categorical ones handled separately
    TABULAR_FEATURES = [
        'zone_index', 'alive_players', 'teammates_alive',
        'storm_edge_dist', 'mats_total', 'surge_above',
        'outcome_placement', 'outcome_alive_time'
    ]
    
    def __init__(self, df: pd.DataFrame, image_dir: str, image_metadata_path: str, transform=None, scaler=None):
        """
        Args:
            df: DataFrame with best_call labels and tabular features
            image_dir: Directory containing images
            image_metadata_path: Path to metadata.jsonl with image paths
            transform: Transforms to apply to images
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load image metadata
        self.image_paths = {}
        if os.path.exists(image_metadata_path):
            with open(image_metadata_path, 'r') as f:
                for line in f:
                    meta = json.loads(line)
                    key = (meta['match_id'], meta['frame_id'])
                    self.image_paths[key] = meta['image_path']
        
        # Class to index mapping
        self.class_to_idx = {c: i for i, c in enumerate(self.ALLOWED_CLASSES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
        # Fit scaler on tabular features if not provided. For validation datasets
        # pass the training scaler so that val uses train statistics.
        if scaler is None:
            self.scaler = StandardScaler()
            tab_data = self.df[self.TABULAR_FEATURES].fillna(0).values
            self.scaler.fit(tab_data)
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image loading
        match_id = str(row.get('match_id', ''))
        frame_id = int(row.get('frame_id', 0))
        key = (match_id, frame_id)
        
        if key in self.image_paths:
            img_path = self.image_paths[key]
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
            except Exception as e:
                print(f'Warning: Failed to load {img_path}: {e}. Using zero tensor.')
                img = torch.zeros(3, 224, 224)
        else:
            # No image found, use zero tensor
            img = torch.zeros(3, 224, 224)
        
        # Tabular features
        tab_features = row[self.TABULAR_FEATURES].fillna(0).values.astype(np.float32)
        tab_features = self.scaler.transform(tab_features.reshape(1, -1))[0]
        tab_tensor = torch.tensor(tab_features, dtype=torch.float32)
        
        # Label
        label = row['best_call']
        label_idx = self.class_to_idx.get(label, 0)
        
        return img, tab_tensor, label_idx


class MultimodalCNN(nn.Module):
    """CNN + Tabular fusion model."""
    
    def __init__(self, num_classes=6, num_tab_features=8):
        super().__init__()
        
        # Vision branch: Simple CNN (ResNet-like)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.vision_fc = nn.Linear(128, 128)
        
        # Tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(num_tab_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, images, tabular):
        # Vision branch
        x_vis = self.vision(images)
        x_vis = x_vis.view(x_vis.size(0), -1)
        x_vis = self.vision_fc(x_vis)
        
        # Tabular branch
        x_tab = self.tabular(tabular)
        
        # Fusion
        x_fused = torch.cat([x_vis, x_tab], dim=1)
        logits = self.fusion(x_fused)
        
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, tab_features, labels in dataloader:
        images = images.to(device)
        tab_features = tab_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images, tab_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, tab_features, labels in dataloader:
            images = images.to(device)
            tab_features = tab_features.to(device)
            labels = labels.to(device)
            
            logits = model(images, tab_features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(dataloader.dataset)
    
    return acc, f1, avg_loss, all_preds, all_labels


def main(args):
    # Load data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f'Data file not found: {args.data}')
    
    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} samples from {args.data}')
    
    # Check for image metadata
    image_meta_path = os.path.join(args.images, 'metadata.jsonl')
    if not os.path.exists(image_meta_path):
        print(f'Warning: Image metadata not found at {image_meta_path}')
        print('Images will be treated as missing (zero tensors will be used).')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Seed for reproducibility
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # TensorBoard writer (optional)
    writer = None
    if args.tensorboard and SummaryWriter is not None:
        tb_dir = f'runs/multimodal_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(tb_dir)
        print(f'TensorBoard logging to {tb_dir}')
    
    # StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    y = df['best_call'].values
    
    cv_results = []
    fold_accuracies = []
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f'\n=== Fold {fold}/{args.n_splits} ===')
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Create datasets and dataloaders. Fit scaler on train and pass to val.
        train_dataset = GameStateImageDataset(
            df_train, args.images, image_meta_path, scaler=None
        )
        val_dataset = GameStateImageDataset(
            df_val, args.images, image_meta_path, scaler=train_dataset.scaler
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Model, optimizer, loss
        model = MultimodalCNN(num_classes=len(train_dataset.ALLOWED_CLASSES), num_tab_features=len(train_dataset.TABULAR_FEATURES)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping state
        best_val_f1 = -1.0
        patience = args.patience
        patience_counter = 0
        checkpoint_dir = f'models/checkpoints/fold_{fold}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop with early stopping and checkpointing
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate every few epochs or at the end
            if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
                val_acc, val_f1, val_loss, preds, labels = evaluate(model, val_loader, device)
                print(f'  Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}')
                
                # TensorBoard logging
                if writer is not None:
                    global_step = fold * args.epochs + epoch
                    writer.add_scalar(f'fold_{fold}/train_loss', train_loss, global_step)
                    writer.add_scalar(f'fold_{fold}/val_loss', val_loss, global_step)
                    writer.add_scalar(f'fold_{fold}/val_accuracy', val_acc, global_step)
                    writer.add_scalar(f'fold_{fold}/val_macro_f1', val_f1, global_step)
                
                # Checkpointing and early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save best model
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    if args.verbose:
                        print(f'    Checkpoint saved: {checkpoint_path} (F1={val_f1:.4f})')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'    Early stopping at epoch {epoch+1} (no improvement for {patience} evals)')
                        break
            else:
                if (epoch + 1) % max(1, args.epochs // 5) == 0:
                    print(f'  Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}')
        
        # Load best model for final evaluation
        best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_checkpoint):
            model.load_state_dict(torch.load(best_checkpoint, map_location=device))
            print(f'  Loaded best model from {best_checkpoint}')
        
        # Final evaluation
        val_acc, val_f1, val_loss, preds, labels = evaluate(model, val_loader, device)
        fold_accuracies.append(val_acc)
        fold_f1s.append(val_f1)
        
        print(f'  Final Val Accuracy: {val_acc:.4f}')
        print(f'  Final Val Macro F1: {val_f1:.4f}')
        
        cv_results.append({
            'fold': fold,
            'accuracy': val_acc,
            'macro_f1': val_f1
        })
    
    # Print summary
    print(f'\n=== Summary ===')
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    
    print(f'CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}')
    
    # Save results
    os.makedirs('models', exist_ok=True)
    cv_df = pd.DataFrame(cv_results)
    cv_path = 'models/multimodal_cv_results.csv'
    cv_df.to_csv(cv_path, index=False)
    print(f'\nResults saved to {cv_path}')
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print('TensorBoard logs saved')

    # Optionally train final model on all data and save
    if args.save_final:
        print('\nTraining final model on full dataset...')
        full_dataset = GameStateImageDataset(df, args.images, image_meta_path)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        final_model = MultimodalCNN(num_classes=len(full_dataset.ALLOWED_CLASSES), num_tab_features=len(full_dataset.TABULAR_FEATURES)).to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(args.epochs):
            loss = train_epoch(final_model, full_loader, optimizer, criterion, device)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f'  Final Epoch {epoch+1}/{args.epochs}: Loss={loss:.4f}')
        os.makedirs('models', exist_ok=True)
        final_path = 'models/multimodal_final.pth'
        torch.save(final_model.state_dict(), final_path)
        print(f'Final model saved to {final_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/call_states_real_labeled.csv',
                       help='Path to labeled data CSV')
    parser.add_argument('--images', type=str, default='data/dummy_screenshots',
                       help='Directory with screenshot images')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--save-final', action='store_true', help='Train final model on full data and save')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (number of eval intervals without improvement)')
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--verbose', action='store_true', help='Verbose checkpoint logging')
    args = parser.parse_args()
    main(args)
