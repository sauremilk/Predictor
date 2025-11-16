"""Export final scikit-learn pipeline to ONNX.

Loads `models/baseline_pipeline_final.joblib`, inspects feature columns from
`data/processed/call_states_synth_large.csv` and exports to
`models/best_call_baseline.onnx`.

Usage:
    python3 src/export_to_onnx.py
"""
import os
import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

MODEL_IN = 'models/baseline_pipeline_final.joblib'
OUT_ONNX = 'models/best_call_baseline.onnx'
DATA_SAMPLE = 'data/processed/call_states_synth_large.csv'

if __name__ == '__main__':
    if not os.path.exists(MODEL_IN):
        raise FileNotFoundError(f"Final pipeline not found: {MODEL_IN}. Train final model first.")
    if not os.path.exists(DATA_SAMPLE):
        raise FileNotFoundError(f"Sample data not found: {DATA_SAMPLE}")

    pipeline = joblib.load(MODEL_IN)
    df = pd.read_csv(DATA_SAMPLE, nrows=1)
    feature_cols = df.drop(columns=['best_call']).columns.tolist()

    # For ONNX export we will export the *classifier* as ONNX and document
    # that the ONNX model expects the preprocessed feature vector (output of the
    # pipeline.named_steps['pre'].transform).
    pre = pipeline.named_steps.get('pre')
    clf = pipeline.named_steps.get('clf')

    # build a sample transformed feature vector to determine dimensionality
    sample_X = pd.read_csv(DATA_SAMPLE).drop(columns=['best_call'])
    X_trans = pre.transform(sample_X.iloc[:1])
    if hasattr(X_trans, 'toarray'):
        X_tr_arr = X_trans.toarray()
    else:
        X_tr_arr = np.asarray(X_trans)
    n_features_trans = X_tr_arr.shape[1]

    initial_types = [('input', FloatTensorType([None, n_features_trans]))]

    # convert only the classifier (expects preprocessed float vector)
    # disable ZipMap so probabilities are returned as a numeric array
    options = {id(clf): {'zipmap': False}}
    onnx_model = convert_sklearn(clf, initial_types=initial_types, target_opset=14, options=options)
    os.makedirs('models', exist_ok=True)
    with open(OUT_ONNX, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f'Exported classifier ONNX model to {OUT_ONNX}')

    # Determine preprocessor output feature names when possible
    feature_names_out = None
    try:
        feature_names_out = pre.get_feature_names_out()
    except Exception:
        # fallback: create generic names
        feature_names_out = [f'col_{i}' for i in range(n_features_trans)]

    print('ONNX expects a preprocessed float vector with length', n_features_trans)

    # write short note for report (features / order)
    report_path = 'models/report.txt'
    with open(report_path, 'a') as f:
        f.write('\nONNX export:\n')
        f.write(f'ONNX classifier model saved to: {OUT_ONNX}\n')
        f.write('ONNX input: preprocessed float vector (output of pipeline.named_steps["pre"].transform)\n')
        f.write(f'Length: {n_features_trans}\n')
        f.write('Preprocessor output feature names / order:\n')
        for i, n in enumerate(feature_names_out, start=1):
            f.write(f' {i}. {n}\n')
        f.write('\nNote: The ONNX model contains only the classifier. For inference the\n')
        f.write('Python preprocessor (`models/baseline_pipeline_final.joblib` -> `pre`) must be\n')
        f.write('applied to raw features to obtain the input vector for ONNX.\n')
    print(f'Appended ONNX input feature mapping to {report_path}')
