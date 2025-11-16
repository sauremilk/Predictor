#!/usr/bin/env python3
"""Test script to validate the wrapped pipeline can be loaded and used with raw input features."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd

# Import the transformer class so pickle can find it during unpickling
from src.feature_engineering import FeatureEngineeringTransformer

def main():
    # Load wrapped pipeline
    try:
        pipeline = joblib.load('models/baseline_pipeline_wrapped.joblib')
        print("✅ Wrapped pipeline loaded successfully")
        print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    except Exception as e:
        print(f"❌ Failed to load wrapped pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create example with raw features (as would come from API or notebook)
    example_raw = pd.DataFrame({
        'zone_phase': ['offensive'],
        'alive_players': [50],
        'zone_index': [2],
        'storm_edge_dist': [250.5],
        'mats_total': [1200],
        'surge_above': [30],
        'height_status': ['mid'],
        'position_type': ['open'],
        'teammates_alive': [3],
        'outcome_placement': [12],
        'outcome_alive_time': [320]
    })

    print(f"\nInput features: {list(example_raw.columns)}")
    print(f"Input shape: {example_raw.shape}")

    # Test prediction
    try:
        pred = pipeline.predict(example_raw)
        print(f"\n✅ Prediction successful: {pred[0]}")
        
        proba = pipeline.predict_proba(example_raw)
        print(f"✅ Predict-proba successful: shape {proba.shape}")
        class_names = ['stick_deadside', 'play_frontside', 'take_height', 'stabilize_box', 'look_for_refresh', 'drop_low']
        print(f"   Class probabilities: {dict(zip(class_names, proba[0].round(4)))}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ All validation tests passed - wrapped pipeline is fully functional!")

if __name__ == '__main__':
    main()
