"""Wrap baseline classifier with FeatureEngineer transformer.

Creates a complete pipeline (FeatureEngineer → Classifier) that accepts raw game-state inputs
and produces predictions. This allows notebooks and APIs to use predict() with raw feature names.

Usage:
    python3 src/wrap_baseline_pipeline.py --model models/baseline_pipeline_final.joblib \
        --training-data data/processed/call_states_synth_large.csv \
        --output models/baseline_pipeline_wrapped.joblib

Saves wrapped pipeline to models/baseline_pipeline_wrapped.joblib.
"""
import argparse
import os
import sys
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from feature_engineering import FeatureEngineer, FeatureEngineeringTransformer

COMMON_DATA_PATHS = [
    'data/call_states_real_labeled.csv',
    'data/processed/call_states_large.csv',
    'data/processed/call_states_synth_large.csv',
    'data/call_states_demo.csv'
]


def find_training_data(provided_path=None):
    """Find training data to fit FeatureEngineer."""
    if provided_path and os.path.exists(provided_path):
        return provided_path
    for p in COMMON_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Wrap baseline classifier with FeatureEngineer')
    parser.add_argument('--model', type=str, default='models/baseline_pipeline_final.joblib',
                       help='Path to saved classifier')
    parser.add_argument('--training-data', type=str, default=None,
                       help='Path to training data for fitting FeatureEngineer')
    parser.add_argument('--output', type=str, default='models/baseline_pipeline_wrapped.joblib',
                       help='Output path for wrapped pipeline')
    args = parser.parse_args()

    # ====== Load classifier ======
    if not os.path.exists(args.model):
        print(f'ERROR: Model file not found: {args.model}')
        sys.exit(1)

    print(f'Loading model from {args.model}...')
    try:
        clf_obj = joblib.load(args.model)
        print(f'  Loaded: {type(clf_obj)}')
    except Exception as e:
        print(f'ERROR loading model: {e}')
        sys.exit(1)

    # Check if already wrapped
    if isinstance(clf_obj, Pipeline):
        steps = [name for name, _ in clf_obj.steps]
        if any(s in ['pre', 'preprocessor', 'fe', 'feature_engineering', 'feature_engineer'] for s in steps):
            print('Model is already a wrapped pipeline. Copying to output...')
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            joblib.dump(clf_obj, args.output)
            print(f'Saved to {args.output}')
            return

    # ====== Find training data ======
    train_path = find_training_data(args.training_data)
    if train_path is None:
        print('ERROR: No training data found. Provide --training-data or place data in common paths.')
        sys.exit(1)

    print(f'Using training data: {train_path}')
    try:
        df = pd.read_csv(train_path)
        print(f'  Loaded {len(df)} rows')
    except Exception as e:
        print(f'ERROR loading training data: {e}')
        sys.exit(1)

    # ====== Prepare features ======
    if 'best_call' in df.columns:
        X = df.drop(columns=['best_call'])
    else:
        X = df

    # ====== Fit FeatureEngineer ======
    print('Fitting FeatureEngineer...')
    try:
        fe = FeatureEngineer()
        fe.fit(X)
        print(f'  Features: {fe.get_feature_names()}')
    except Exception as e:
        print(f'ERROR fitting FeatureEngineer: {e}')
        sys.exit(1)

    # ====== Build wrapped pipeline ======
    print('Building wrapped pipeline (FeatureEngineeringTransformer → Classifier)...')
    try:
        fe_transformer = FeatureEngineeringTransformer(engineer=fe)
        fe_transformer.fit(X)  # Fit the transformer on training data
        wrapped_pipeline = Pipeline([
            ('fe', fe_transformer),
            ('clf', clf_obj)
        ])
        print('  Pipeline created and transformer fitted')
    except Exception as e:
        print(f'ERROR building pipeline: {e}')
        sys.exit(1)

    # ====== Test on sample ======
    print('Testing wrapped pipeline on sample...')
    try:
        sample = X.iloc[:3]
        preds = wrapped_pipeline.predict(sample)
        print(f'  Sample predictions: {preds}')
        try:
            proba = wrapped_pipeline.predict_proba(sample)
            print(f'  Probabilities available (shape={proba.shape})')
        except Exception:
            print('  predict_proba not available')
    except Exception as e:
        print(f'WARNING: Test prediction failed: {e}')

    # ====== Save ======
    print(f'Saving wrapped pipeline to {args.output}...')
    try:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        joblib.dump(wrapped_pipeline, args.output)
        print('✓ Wrapped pipeline saved successfully')
    except Exception as e:
        print(f'ERROR saving pipeline: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
