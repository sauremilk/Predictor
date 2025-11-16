"""Generate a larger synthetic dataset for `best_call` and save to `data/processed/call_states_synth_large.csv`.

Usage:
    python3 src/generate_synth_large.py --n 800
"""
import os
import argparse
import numpy as np
import pandas as pd


def generate_large(n=800, seed=42):
    rng = np.random.RandomState(seed)
    # classes (6 total) with a reasonable imbalance
    class_names = [
        'stick_deadside',
        'play_frontside',
        'take_height',
        'stabilize_box',
        'look_for_refresh',
        'drop_low'
    ]
    # distribution (sum to 1): stick 30%, play 25%, take 15%, stabilize 12%, look 10%, drop 8%
    p = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]
    classes = rng.choice(class_names, size=n, p=p)

    # features per spec
    alive_players = rng.randint(20, 81, size=n)  # 20-80
    zone_index = rng.randint(3, 10, size=n)      # 3-9 inclusive
    storm_edge_dist = rng.randint(0, 61, size=n) # 0-60
    mats_total = rng.randint(0, 3001, size=n)   # 0-3000
    surge_above = rng.randint(-150, 401, size=n) # -150 to +400

    # other example features to keep compatibility with pipeline
    zone_phase = rng.choice(['early','mid','late'], size=n, p=[0.3,0.5,0.2])
    height_status = rng.choice(['low','mid','high'], size=n, p=[0.2,0.5,0.3])
    position_type = rng.choice(['center','edge','corner'], size=n)
    teammates_alive = rng.randint(0,4,size=n)
    outcome_placement = rng.randint(1,101,size=n)
    outcome_alive_time = rng.randint(1,401,size=n)

    df = pd.DataFrame({
        'zone_phase': zone_phase,
        'alive_players': alive_players,
        'zone_index': zone_index,
        'storm_edge_dist': storm_edge_dist,
        'mats_total': mats_total,
        'surge_above': surge_above,
        'height_status': height_status,
        'position_type': position_type,
        'teammates_alive': teammates_alive,
        'outcome_placement': outcome_placement,
        'outcome_alive_time': outcome_alive_time,
        'best_call': classes
    })
    return df


def main(args):
    os.makedirs('data/processed', exist_ok=True)
    n = args.n
    df = generate_large(n=n, seed=args.seed)
    out = 'data/processed/call_states_synth_large.csv'
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=800, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args)
