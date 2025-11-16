"""Label helper utilities:

Two main commands:

1) prepare --input <csv|jsonl> --output data/call_states_to_label.csv
   - Reads input Game-States (CSV or JSONL) using the canonical schema
   - Adds `row_id` (if match_id+frame_id available uses that) and an empty
     `best_call` column for manual labeling

2) finalize --input data/call_states_labeled.csv --output data/call_states_real_labeled.csv
   - Reads the manually labeled CSV, validates that `best_call` values are in
     the allowed 6-class set, and writes the final training CSV.

Usage examples:
  python3 src/label_helper.py prepare --input data/examples/call_states_batch_example.jsonl
  python3 src/label_helper.py finalize --input data/call_states_labeled.csv

The script is intentionally permissive in `prepare` (keeps all columns) and
strict in `finalize` (rejects invalid/missing labels).
"""
import argparse
import csv
import json
import os
import pandas as pd
from typing import List

ALLOWED_CLASSES = [
    'stick_deadside',
    'play_frontside',
    'take_height',
    'stabilize_box',
    'look_for_refresh',
    'drop_low'
]

CANONICAL_FIELDS = [
    'match_id', 'frame_id',
    'zone_phase', 'zone_index',
    'alive_players', 'teammates_alive',
    'storm_edge_dist', 'mats_total', 'surge_above',
    'height_status', 'position_type',
    'outcome_placement', 'outcome_alive_time'
]


def read_input(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext in ('.jsonl', '.ndjson'):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(rows)
    else:
        # try reading as CSV
        return pd.read_csv(path)


def make_row_id(row) -> str:
    # prefer match_id+frame_id when available and not null
    mid = row.get('match_id')
    fid = row.get('frame_id')
    if pd.notna(mid) and pd.notna(fid):
        return f"{mid}_{fid}"
    # fallback: use index-based id
    return None


def prepare(input_path: str, output_path: str):
    df = read_input(input_path)
    # ensure canonical fields exist (keep other fields too)
    for c in CANONICAL_FIELDS:
        if c not in df.columns:
            df[c] = None

    # compute row_id column
    row_ids = []
    for idx, r in df.iterrows():
        rid = make_row_id(r)
        if rid is None:
            rid = f'row_{idx}'
        row_ids.append(rid)
    df['row_id'] = row_ids

    # add empty best_call column for manual labeling if not present
    if 'best_call' not in df.columns:
        df['best_call'] = ''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # write to CSV with all columns, row_id and empty best_call
    df.to_csv(output_path, index=False)
    print(f'Wrote {len(df)} rows to {output_path} (ready for manual labeling)')


def finalize(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    if 'best_call' not in df.columns:
        raise SystemExit('Input file must contain a `best_call` column with labels')

    # drop rows with empty labels
    missing_mask = df['best_call'].isnull() | (df['best_call'].astype(str).str.strip() == '')
    if missing_mask.any():
        cnt = missing_mask.sum()
        raise SystemExit(f'Found {cnt} rows with empty `best_call`. Please fill all labels before finalizing.')

    # validate labels
    invalid = df[~df['best_call'].isin(ALLOWED_CLASSES)]
    if len(invalid) > 0:
        # list unique invalid labels
        bad = sorted(invalid['best_call'].unique())
        raise SystemExit(f'Invalid labels found: {bad}. Allowed: {ALLOWED_CLASSES}')

    # select canonical fields + best_call + row_id (keep other fields too)
    # for training, we keep all columns but ensure 'best_call' exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Validated and wrote {len(df)} labeled rows to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_prep = sub.add_parser('prepare')
    p_prep.add_argument('--input', '-i', required=True, help='Input CSV or JSONL of Game-States')
    p_prep.add_argument('--output', '-o', default='data/call_states_to_label.csv', help='Output CSV for manual labeling')

    p_fin = sub.add_parser('finalize')
    p_fin.add_argument('--input', '-i', required=True, help='Manually labeled CSV with best_call column')
    p_fin.add_argument('--output', '-o', default='data/call_states_real_labeled.csv', help='Validated training CSV output')

    args = parser.parse_args()
    if args.cmd == 'prepare':
        prepare(args.input, args.output)
    elif args.cmd == 'finalize':
        finalize(args.input, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
