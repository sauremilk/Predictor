"""Run batch predictions from a JSONL file and write CSV output.

Each line in the input JSONL must be a JSON object representing one Game-State.
Optional keys supported per input: `match_id`, `frame_id`. Other keys must match
what `src/predict_best_call.predict` expects (same as `--json` example).

Output CSV columns:
  match_id, frame_id, predicted_call, p_predicted, p_second, second_call, all_probs_json

Usage example:
  python3 src/run_batch_predictions.py --input data/examples/call_states_batch_example.jsonl \
      --output models/batch_predictions_example.csv
"""
import argparse
import json
import csv
import os
from typing import Dict

import sys
import numpy as np
# ensure `src` module dir is on path so we can import `predict_best_call` when running
sys.path.insert(0, os.path.dirname(__file__))
import predict_best_call
import joblib


def load_pipeline_and_session():
    # load pipeline and onnx session once
    pipeline = None
    sess = None
    if os.path.exists(predict_best_call.JOBLIB_PIPELINE):
        pipeline = joblib.load(predict_best_call.JOBLIB_PIPELINE)
    else:
        raise FileNotFoundError(f"Joblib pipeline not found: {predict_best_call.JOBLIB_PIPELINE}")
    sess = predict_best_call.load_session()
    return pipeline, sess


def row_from_prediction(example: Dict, pred_label: str, probs: "np.ndarray", labels: list):
    # Build probability mapping label->prob
    prob_map = {lab: float(p) for lab, p in zip(labels, probs)}
    # sort by probability desc
    sorted_items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    primary_label, primary_p = sorted_items[0]
    if len(sorted_items) > 1:
        second_label, second_p = sorted_items[1]
    else:
        second_label, second_p = '', 0.0

    match_id = example.get('match_id', '')
    frame_id = example.get('frame_id', '')

    return {
        'match_id': match_id,
        'frame_id': frame_id,
        'predicted_call': primary_label,
        'p_predicted': primary_p,
        'p_second': second_p,
        'second_call': second_label,
        'all_probs_json': json.dumps(prob_map, ensure_ascii=False)
    }


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    pipeline, sess = load_pipeline_and_session()

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if args.output else None

    # prepare CSV writer
    fieldnames = ['match_id', 'frame_id', 'predicted_call', 'p_predicted', 'p_second', 'second_call', 'all_probs_json']
    out_rows = []
    with open(args.input, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: skipping invalid JSON at line {i}')
                continue
            pred_label, probs, labels = predict_best_call.predict(example, pipeline=pipeline, sess=sess)
            row = row_from_prediction(example, pred_label, probs, labels)
            out_rows.append(row)

    # write CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f'Wrote {len(out_rows)} predictions to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to JSONL input file')
    parser.add_argument('--output', type=str, required=True, help='Path to CSV output file')
    args = parser.parse_args()
    main(args)
