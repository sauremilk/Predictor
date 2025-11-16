"""Wrap an existing saved classifier with the FeatureEngineer so inference accepts raw features.

Usage:
    python3 src/wrap_baseline_with_fe.py --model models/baseline_pipeline_final.joblib

The script will look for a training CSV to fit the FeatureEngineer if needed.
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from feature_engineering import FeatureEngineer

CANDIDATE_DATA_PATHS = [
    'data/call_states_real_labeled.csv',
    'data/processed/call_states_synth_large.csv',
    'data/processed/call_states_large.csv',
    'data/call_states_demo.csv',
    'data/call_states_labeled.csv'
]

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, engineer: FeatureEngineer = None):
        self.engineer = engineer or FeatureEngineer()

    def fit(self, X, y=None):
        # X is expected to be a DataFrame with raw features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.engineer.fit(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Xt = self.engineer.transform(X)
        # return numpy array for sklearn compatibility
        return Xt.values


def find_training_csv():
    for p in CANDIDATE_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def main(args):
    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f'Loaded model from {model_path} (type={type(model)})')

    # Heuristic: if model is a Pipeline and contains a preprocessor step, nothing to do
    transformer_needed = True
    if isinstance(model, Pipeline):
        steps = dict(model.named_steps)
        # common names: 'pre', 'preprocessor', 'preproc'
        for key in ['pre', 'preprocessor', 'preproc']:
            if key in steps:
                transformer_needed = False
                print('Model pipeline already contains a preprocessor step; no wrapping needed.')
                break
        # If pipeline has only 'clf' step, we need wrapping
        if len(steps) > 0 and 'clf' in steps and len(steps) == 1:
            transformer_needed = True
    else:
        # Single estimator (not a pipeline) -> need wrapping
        transformer_needed = True

    if not transformer_needed:
        print('No action required.')
        return

    # Find training CSV to fit the FeatureEngineer
    csv_path = find_training_csv()
    if csv_path is None:
        raise FileNotFoundError('No candidate training CSV found to fit FeatureEngineer. Please provide training data.')

    print(f'Using training CSV {csv_path} to fit FeatureEngineer')
    df = pd.read_csv(csv_path)
    if 'best_call' in df.columns:
        X_train = df.drop(columns=['best_call'])
    else:
        X_train = df

    fe = FeatureEngineeringTransformer()
    fe.fit(X_train)
    print('Fitted FeatureEngineer on training data')

    # Extract classifier
    if isinstance(model, Pipeline):
        # assume final step is classifier
        clf = model.steps[-1][1]
    else:
        clf = model

    wrapped_pipeline = Pipeline([
        ('fe', fe),
        ('clf', clf)
    ])

    out_path = args.out or 'models/baseline_pipeline_wrapped.joblib'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(wrapped_pipeline, out_path)
    print(f'Saved wrapped pipeline to {out_path}')

    # Quick smoke test with a sample row from X_train
    sample = X_train.iloc[[0]]
    try:
        preds = wrapped_pipeline.predict(sample)
        print('Smoke test prediction OK:', preds[0])
    except Exception as e:
        print('Smoke test prediction FAILED:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/baseline_pipeline_final.joblib')
    parser.add_argument('--out', type=str, default='models/baseline_pipeline_wrapped.joblib')
    args = parser.parse_args()
    main(args)
