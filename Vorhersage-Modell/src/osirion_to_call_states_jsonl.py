"""Translate Osirion / API / replay exports into call_states JSONL compatible with
`src/run_batch_predictions.py` and training pipeline.

This is a flexible converter skeleton — adapt `parse_osirion_record()` to your
actual export format. The script will output one JSON line per Game-State with
exactly the canonical fields used across the project.

Canonical Game-State schema (each JSON line must contain these keys):
  - match_id, frame_id
  - zone_phase, zone_index
  - alive_players, teammates_alive
  - storm_edge_dist, mats_total, surge_above
  - height_status, position_type
  - outcome_placement, outcome_alive_time

Usage example:
  python3 src/osirion_to_call_states_jsonl.py --in osirion_dump.json --out-dir data/processed

If you pass `--out-dir`, the script writes a timestamped file named
`call_states_YYYYMMDD.jsonl` into that dir.

Note: This file is a converter skeleton — update `parse_osirion_record()` to
extract fields from your actual Osirion / replay export schema.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, Iterable

CANONICAL_FIELDS = [
    'match_id', 'frame_id',
    'zone_phase', 'zone_index',
    'alive_players', 'teammates_alive',
    'storm_edge_dist', 'mats_total', 'surge_above',
    'height_status', 'position_type',
    'outcome_placement', 'outcome_alive_time'
]


def parse_osirion_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Osirion/export record to canonical Game-State dict.

    Customize this function to match the structure of your Osirion/API/Replay
    export. The goal is to return a dict with at least the keys in
    `CANONICAL_FIELDS`. Missing values are allowed (set to None), but it's
    better to fill sensible defaults where possible.
    """
    # Example mapping for a hypothetical Osirion JSON structure. Replace with
    # the real mapping logic for your exports.
    gs = {}

    # match/frame identifiers
    gs['match_id'] = record.get('match_id') or record.get('game_id') or record.get('match')
    gs['frame_id'] = record.get('frame') or record.get('tick') or record.get('time')

    # zone info
    gs['zone_phase'] = record.get('zone', {}).get('phase') if isinstance(record.get('zone'), dict) else record.get('zone_phase')
    gs['zone_index'] = record.get('zone_index') or (record.get('zone', {}).get('index') if isinstance(record.get('zone'), dict) else None)

    # players
    gs['alive_players'] = record.get('alive_players') or record.get('players_alive')
    gs['teammates_alive'] = record.get('teammates_alive') or record.get('num_teammates')

    # distances / mats / surge
    gs['storm_edge_dist'] = record.get('storm_edge_distance') or record.get('storm_edge_dist')
    gs['mats_total'] = record.get('materials_total') or record.get('mats_total')
    gs['surge_above'] = record.get('surge_above') if 'surge_above' in record else record.get('elevation_delta')

    # height + position
    gs['height_status'] = record.get('height') or record.get('height_status')
    gs['position_type'] = record.get('position') or record.get('position_type')

    # outcomes (if known at that time)
    gs['outcome_placement'] = record.get('placement') or record.get('outcome_placement')
    gs['outcome_alive_time'] = record.get('alive_time') or record.get('outcome_alive_time')

    # Ensure all canonical keys exist (fill missing with None)
    for k in CANONICAL_FIELDS:
        if k not in gs:
            gs[k] = None

    return gs


def iter_records_from_input(path: str) -> Iterable[Dict[str, Any]]:
    """Read the input dump and yield raw records.

    This helper supports JSONL and JSON arrays as common export formats. For
    other formats (CSV, binary replays) implement a custom reader here.
    """
    _, ext = os.path.splitext(path.lower())
    if ext in ('.jsonl', '.ndjson'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # if top-level is an array, iterate; if it's an object with 'events', iterate that
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                # common keys to try
                for k in ('events', 'frames', 'records'):
                    if k in data and isinstance(data[k], list):
                        for item in data[k]:
                            yield item
                        break
                else:
                    # fallback to iterating values if they're list-like
                    for v in data.values():
                        if isinstance(v, list):
                            for item in v:
                                yield item
    else:
        raise ValueError('Unsupported input file type. Provide .json or .jsonl')


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input file not found: {args.input}')

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d')
    out_name = args.out_name or f'call_states_{ts}.jsonl'
    out_path = os.path.join(args.out_dir, out_name)

    n_written = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for raw in iter_records_from_input(args.input):
            gs = parse_osirion_record(raw)
            # optionally filter or validate minimal required fields
            # we keep everything and let downstream pipeline handle missing data
            fout.write(json.dumps(gs, ensure_ascii=False) + '\n')
            n_written += 1

    print(f'Wrote {n_written} Game-State lines to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to Osirion / export JSON or JSONL file')
    parser.add_argument('--out-dir', '-d', default='data/processed', help='Directory to write call_states JSONL')
    parser.add_argument('--out-name', '-o', default=None, help='Optional output filename (default: call_states_YYYYMMDD.jsonl)')
    args = parser.parse_args()
    main(args)
