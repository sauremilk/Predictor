"""Train a simple baseline classifier for `best_call`.

Usage:
    python3 src/train_best_call_baseline.py --data data/call_states_demo.csv
    python3 src/train_best_call_baseline.py --data data/call_states_real_labeled.csv --feature-engineering --importance

If the CSV is missing, the script can generate a synthetic dataset.
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from feature_engineering import FeatureEngineer, analyze_feature_importance


def create_classifier(model_type='rf'):
    """Create a classifier based on model_type.
    Args:
        model_type: 'rf' for RandomForest (default), 'hgb' for HistGradientBoosting.
    """
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    elif model_type == 'hgb':
        return HistGradientBoostingClassifier(random_state=42, max_iter=100, learning_rate=0.1)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


def generate_synthetic(n=500, seed=42):
    rng = np.random.RandomState(seed)
    zone_phase = rng.choice(['early', 'mid', 'late'], size=n, p=[0.3,0.5,0.2])
    alive_players = rng.randint(1,5,size=n)
    mats_total = rng.randint(0,301,size=n)
    surge_above = rng.choice([True, False], size=n, p=[0.25, 0.75])
    height_status = rng.choice(['low','mid','high'], size=n, p=[0.2,0.5,0.3])
    position_type = rng.choice(['edge','center','cover'], size=n)
    teammates_alive = rng.randint(0,4,size=n)
    outcome_placement = rng.randint(1,101,size=n)
    outcome_alive_time = rng.randint(1,401,size=n)
    # heuristic-ish synthetic label
    best_call = []
    for zp, hp, at, mats in zip(zone_phase, height_status, alive_players, mats_total):
        if hp == 'high' and at >= 3:
            best_call.append('take_height')
        elif at <= 1:
            best_call.append('stick_deadside')
        else:
            best_call.append('play_frontside')
    df = pd.DataFrame({
        'zone_phase': zone_phase,
        'alive_players': alive_players,
        'mats_total': mats_total,
        'surge_above': surge_above,
        'height_status': height_status,
        'position_type': position_type,
        'teammates_alive': teammates_alive,
        'outcome_placement': outcome_placement,
        'outcome_alive_time': outcome_alive_time,
        'best_call': best_call
    })
    return df


def load_data(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
    else:
        print("Data file not found — generating synthetic dataset (n=1000)")
        df = generate_synthetic(n=1000)
    return df


def compare_models(data_path, n_splits=5):
    """Compare rf and hgb models using same CV splits."""
    df = load_data(data_path)
    target = 'best_call'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
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

    comparison_results = []
    os.makedirs('models', exist_ok=True)

    # Test both models
    for model_type in ['rf', 'hgb']:
        print(f'\nComparing {model_type} model with {n_splits}-fold CV...')
        clf = create_classifier(model_type)
        pipeline = Pipeline([
            ('pre', preprocessor),
            ('clf', clf)
        ])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        f1s = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='macro')
            accs.append(acc)
            f1s.append(f1)
            print(f'  Fold {fold}: Accuracy={acc:.4f}, MacroF1={f1:.4f}')

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        print(f'  {model_type} CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
        print(f'  {model_type} CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}')

        comparison_results.append({
            'model': model_type,
            'cv_accuracy_mean': mean_acc,
            'cv_accuracy_std': std_acc,
            'cv_macro_f1_mean': mean_f1,
            'cv_macro_f1_std': std_f1
        })

    # Write comparison CSV
    cmp_df = pd.DataFrame(comparison_results)
    cmp_path = 'models/cv_model_comparison.csv'
    cmp_df.to_csv(cmp_path, index=False)
    print(f'\nModel comparison saved to {cmp_path}')
    print(cmp_df.to_string())

    # Append to report
    report_path = 'models/report.txt'
    with open(report_path, 'a') as f:
        f.write('\n\nModel Comparison\n')
        f.write('================\n')
        for _, row in cmp_df.iterrows():
            f.write(f"{row['model'].upper()}: Accuracy={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f}, "
                   f"MacroF1={row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f}\n")


def main(args):
    df = load_data(args.data)
    target = 'best_call'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # ====== Feature Engineering ======
    if args.feature_engineering:
        print('Applying feature engineering...')
        engineer = FeatureEngineer()
        X = engineer.fit_transform(df.drop(columns=[target]))
        y = df[target]
        cat_cols = []  # Already handled by engineer
        num_cols = X.columns.tolist()
    else:
        X = df.drop(columns=[target])
        y = df[target]
        cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        engineer = None

    # Build preprocessing (skip if feature engineering was used)
    if args.feature_engineering:
        # Already scaled and engineered; just use as-is
        preprocessor = None
        clf = create_classifier(args.model)
        if args.feature_engineering:
            pipeline = Pipeline([('clf', clf)])
        else:
            pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
    else:
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

        clf = create_classifier(args.model)
        pipeline = Pipeline([
            ('pre', preprocessor),
            ('clf', clf)
        ])

    # Cross-Validation or single holdout
    os.makedirs('models', exist_ok=True)
    # placeholders for selecting final estimator
    best_pipeline = None
    grid = None

    if args.cv:
        print(f'Running StratifiedKFold CV with {args.n_splits} splits...')
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        best_f1 = -1.0
        best_pipeline = None
        labels = np.unique(y)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            print(f' Fold {fold}: train={len(X_train)}, val={len(X_val)}')
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='macro')
            fold_metrics.append({'fold': fold, 'accuracy': acc, 'macro_f1': f1})
            all_y_true.append(y_val.values)
            all_y_pred.append(y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = copy.deepcopy(pipeline)

        # aggregate
        accs = [m['accuracy'] for m in fold_metrics]
        f1s = [m['macro_f1'] for m in fold_metrics]
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
        print(f'CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
        print(f'CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}')

        # save cv report
        cv_df = pd.DataFrame(fold_metrics)
        cv_report_path = os.path.join('models', 'cv_report.csv')
        cv_df.to_csv(cv_report_path, index=False)
        print(f'Saved CV per-fold metrics to {cv_report_path}')

        # overall classification report (concatenate all folds)
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        overall_report = classification_report(y_true_all, y_pred_all, digits=4)
        report_path = os.path.join('models', 'report.txt')
        with open(report_path, 'w') as f:
            f.write(f'CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n')
            f.write(f'CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}\n\n')
            f.write(overall_report)
        print(f'Saved aggregated report to {report_path}')

        # confusion matrix aggregated
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Aggregated Confusion Matrix (CV)')
        cm_path = os.path.join('models', 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        print(f'Saved confusion matrix to {cm_path}')

        # save best pipeline
        if best_pipeline is not None:
            model_path = os.path.join('models', 'baseline_pipeline.joblib')
            joblib.dump(best_pipeline, model_path)
            print(f'Saved best pipeline (by fold macro_f1={best_f1:.4f}) to {model_path}')

        # ====== Feature Importance Analysis ======
        if args.importance and best_pipeline is not None:
            print('Computing feature importance on validation set...')
            # Recompute last fold for importance
            train_idx, val_idx = list(skf.split(X, y))[-1]
            X_val_last = X.iloc[val_idx]
            y_val_last = y.iloc[val_idx]
            try:
                importances = analyze_feature_importance(
                    best_pipeline.named_steps['clf'],
                    X_val_last,
                    y_val_last,
                    output_path='models/feature_importance.json'
                )
                print('Top 10 important features:')
                for i, (feat, imp) in enumerate(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                    print(f"  {i}. {feat:35s}: {imp:+.6f}")
            except Exception as e:
                print(f'Feature importance computation failed: {e}')

    # Optionally run GridSearchCV on the full dataset
    if args.grid_search:
        print('Running GridSearchCV for RandomForest...')
        param_grid = {
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [None, 8, 16],
            'clf__min_samples_leaf': [1, 3, 5],
            'clf__max_features': ['sqrt', 'log2']
        }
        gs_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=gs_cv, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid.fit(X, y)
        best_params = grid.best_params_
        best_score = grid.best_score_
        # estimate std from cv_results for best index
        best_idx = grid.best_index_
        std_score = grid.cv_results_['std_test_score'][best_idx]
        print(f'Best params: {best_params}')
        print(f'Best CV macro_f1: {best_score:.4f} ± {std_score:.4f}')

        # append to report
        report_path = os.path.join('models', 'report.txt')
        with open(report_path, 'a') as f:
            f.write('\nGridSearchCV Best Params:\n')
            f.write(str(best_params) + '\n')
            f.write(f'Best CV macro_f1: {best_score:.4f} ± {std_score:.4f}\n')
        print(f'Appended GridSearch results to {report_path}')

        # append summary row to cv_report.csv
        cv_report_path = os.path.join('models', 'cv_report.csv')
        if os.path.exists(cv_report_path):
            cv_df = pd.read_csv(cv_report_path)
            extra = {'fold': 'best_model_mean', 'accuracy': best_score, 'macro_f1': std_score}
            cv_df = pd.concat([cv_df, pd.DataFrame([extra])], ignore_index=True)
            cv_df.to_csv(cv_report_path, index=False)
            print(f'Appended best_model summary to {cv_report_path}')
        else:
            pd.DataFrame([{'fold': 'best_model_mean', 'accuracy': best_score, 'macro_f1': std_score}]).to_csv(cv_report_path, index=False)
            print(f'Created {cv_report_path} with best_model summary')

    else:
        # single holdout fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # train
        print('Training baseline RandomForest (single holdout)...')
        pipeline.fit(X_train, y_train)

        # predict
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f'Accuracy: {acc:.4f}')
        print(f'Macro F1: {f1:.4f}')
        print('\nClassification report:\n')
        print(classification_report(y_test, y_pred, digits=4))

        # save model
        model_path = os.path.join('models', 'baseline_pipeline.joblib')
        joblib.dump(pipeline, model_path)
        print(f'Saved pipeline to {model_path}')

        # save report
        report_path = os.path.join('models', 'report.txt')
        with open(report_path, 'w') as f:
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'Macro F1: {f1:.4f}\n\n')
            f.write(classification_report(y_test, y_pred, digits=4))
        print(f'Saved report to {report_path}')

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = os.path.join('models', 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        print(f'Saved confusion matrix to {cm_path}')

    # After CV / Grid / Holdout, train a final pipeline on the full dataset
    # Prefer GridSearch best estimator, then best_pipeline from CV, else the original pipeline
    final_estimator = None
    if grid is not None and hasattr(grid, 'best_estimator_'):
        final_estimator = grid.best_estimator_
        print('Using GridSearch best estimator for final fit')
    elif best_pipeline is not None:
        final_estimator = best_pipeline
        print('Using best pipeline from CV for final fit')
    else:
        final_estimator = pipeline
        print('Using base pipeline for final fit')

    # Fit final estimator on the entire dataset
    try:
        final_estimator.fit(X, y)
        final_path = os.path.join('models', 'baseline_pipeline_final.joblib')
        joblib.dump(final_estimator, final_path)
        print(f'Saved final pipeline trained on full data to {final_path}')

        # append final model summary to report
        report_path = os.path.join('models', 'report.txt')
        with open(report_path, 'a') as f:
            f.write('\nFinal model trained on full dataset and saved to `models/baseline_pipeline_final.joblib`.\n')
            try:
                classes = list(final_estimator.named_steps['clf'].classes_)
            except Exception:
                # fallback if pipeline is not a Pipeline object
                try:
                    classes = list(final_estimator.classes_)
                except Exception:
                    classes = []
            f.write('Classes: ' + str(classes) + '\n')
        print(f'Appended final model summary to {report_path}')
    except Exception as e:
        print('Failed to fit final estimator on full data:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/call_states_demo.csv', help='Path to CSV with the call states')
    parser.add_argument('--cv', dest='cv', action='store_true', help='Enable StratifiedKFold CV (default: enabled)')
    parser.add_argument('--no-cv', dest='cv', action='store_false', help='Disable CV and use single holdout')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of splits for StratifiedKFold')
    parser.set_defaults(cv=True)
    parser.add_argument('--grid-search', dest='grid_search', action='store_true', help='Run GridSearchCV after CV')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--model', type=str, choices=['rf', 'hgb'], default='rf', help='Model type: rf (RandomForest) or hgb (HistGradientBoosting)')
    parser.add_argument('--compare', dest='compare', action='store_true', help='Run comparison between both rf and hgb models')
    parser.set_defaults(compare=False)
    parser.add_argument('--feature-engineering', dest='feature_engineering', action='store_true', help='Enable feature engineering (ratios, interactions, transforms)')
    parser.set_defaults(feature_engineering=False)
    parser.add_argument('--importance', dest='importance', action='store_true', help='Compute permutation feature importance')
    parser.set_defaults(importance=False)
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.data, args.n_splits)
    else:
        main(args)
