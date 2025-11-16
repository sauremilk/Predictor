"""Inference helper: apply Python preprocessor then run ONNX classifier.

Usage:
  python3 -m src.predict_best_call --input example.json
  python3 -m src.predict_best_call --json '{"alive_players":30, ...}'

The script:
- loads `models/baseline_pipeline_final.joblib` and its `pre` (preprocessor)
- loads `models/best_call_baseline.onnx` (classifier) via onnxruntime
- applies `pre.transform` to the input Game-State and feeds the resulting float
  vector to the ONNX classifier
- prints predicted class and class probabilities (labels from Python classifier)

This keeps preprocessing in Python (using the saved pipeline) and uses ONNX for the
classifier core (portable runtime).
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

ONNX_PATH = 'models/best_call_baseline.onnx'
JOBLIB_PIPELINE = 'models/baseline_pipeline_final.joblib'


def load_session(path=ONNX_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}. Run export_to_onnx.py first.")
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])


def predict(example: dict, pipeline=None, sess=None):
    # load pipeline and sess (allow passing preloaded objects to avoid repeated loads)
    if pipeline is None:
        if not os.path.exists(JOBLIB_PIPELINE):
            raise FileNotFoundError(f"Joblib pipeline not found: {JOBLIB_PIPELINE}. Train final model first.")
        pipeline = joblib.load(JOBLIB_PIPELINE)
    pre = pipeline.named_steps.get('pre')
    clf = pipeline.named_steps.get('clf')
    if sess is None:
        sess = load_session()

    # build dataframe and apply preprocessing
    example_df = pd.DataFrame([example])
    X_trans = pre.transform(example_df)
    if hasattr(X_trans, 'toarray'):
        X_arr = X_trans.toarray().astype(np.float32)
    else:
        X_arr = np.asarray(X_trans).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {input_name: X_arr})

    # Map outputs by name
    outputs = {name: np.array(arr) for name, arr in zip(out_names, res)}

    probs = None
    pred_label = None

    # Preferred: explicit probability output
    if 'output_probability' in outputs:
        probs = outputs['output_probability']
    # If skl2onnx produced a label output as string, capture it
    if 'output_label' in outputs:
        # output_label may be ndarray of bytes/strings
        lab = outputs['output_label']
        try:
            pred_label = lab[0].tobytes().decode() if hasattr(lab[0], 'tobytes') else str(lab[0])
        except Exception:
            pred_label = str(lab[0])

    # If no probability array from ONNX, fallback to Python predict_proba
    if probs is None:
        try:
            probs = clf.predict_proba(X_trans)
        except Exception:
            # as final fallback, set uniform small vector
            n_classes = len(clf.classes_)
            probs = np.ones((1, n_classes)) / n_classes

    # Ensure probs is 2D array
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    labels = list(clf.classes_)
    if pred_label is None:
        pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

    return pred_label, probs[0], labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to JSON file with a single example (dict)')
    parser.add_argument('--json', type=str, help='Inline JSON string for a single example')
    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r') as f:
            example = json.load(f)
    elif args.json:
        example = json.loads(args.json)
    else:
        raise SystemExit('Provide --input sample.json or --json "{...}"')

    pred_label, probs_arr, labels = predict(example)
    print('Predicted class:', pred_label)
    print('Class labels order:', labels)
    if isinstance(probs_arr, (list, tuple)) or (hasattr(probs_arr, 'tolist')):
        try:
            print('Class probabilities:', probs_arr.tolist())
        except Exception:
            print('Class probabilities:', list(probs_arr))
    else:
        print('Class probabilities:', [float(probs_arr)])
