"""Generate larger synthetic dataset for improved training (5000+ samples).

Usage:
    python3 src/generate_large_dataset.py --n 5000 --output data/processed/call_states_large.csv --seed 123
"""
import argparse
import os
import numpy as np
import pandas as pd

ALLOWED_CLASSES = [
    'stick_deadside', 'play_frontside', 'take_height',
    'stabilize_box', 'look_for_refresh', 'drop_low'
]

def generate_large(n=5000, seed=42):
    """Generate large synthetic dataset with realistic feature correlations.
    
    Args:
        n: Number of samples
        seed: Random seed
    
    Returns:
        DataFrame with call_states
    """
    rng = np.random.RandomState(seed)
    
    # ====== Zone Phase (early/mid/late) ======
    zone_phase = rng.choice(['early', 'mid', 'late'], size=n, p=[0.25, 0.50, 0.25])
    
    # ====== Height Status (low/mid/high) ======
    height_status = rng.choice(['low', 'mid', 'high'], size=n, p=[0.35, 0.40, 0.25])
    
    # ====== Position Type (corner/center/edge) ======
    position_type = rng.choice(['corner', 'center', 'edge'], size=n, p=[0.35, 0.35, 0.30])
    
    # ====== Zone Index (1-15: early=1-5, mid=6-10, late=11-15) ======
    zone_index = np.where(
        zone_phase == 'early', rng.randint(1, 6, size=n),
        np.where(
            zone_phase == 'mid', rng.randint(6, 11, size=n),
            rng.randint(11, 16, size=n)
        )
    )
    
    # ====== Player Counts (correlate with zone phase) ======
    # Early: higher player count; late: lower
    alive_players = np.where(
        zone_phase == 'early', rng.randint(2, 6, size=n),
        np.where(
            zone_phase == 'mid', rng.randint(1, 5, size=n),
            rng.randint(1, 4, size=n)
        )
    )
    teammates_alive = np.maximum(0, alive_players - rng.randint(0, 3, size=n))
    
    # ====== Materials (correlate with position_type) ======
    mats_total = np.where(
        position_type == 'corner', rng.randint(50, 300, size=n),
        np.where(
            position_type == 'center', rng.randint(100, 400, size=n),
            rng.randint(20, 200, size=n)
        )
    )
    
    # ====== Storm Edge Distance ======
    storm_edge_dist = rng.randint(10, 500, size=n)
    
    # ====== Surge Above (binary with height correlation) ======
    surge_prob = np.where(height_status == 'high', 0.6, np.where(height_status == 'mid', 0.3, 0.1))
    surge_above = rng.binomial(1, surge_prob)
    
    # ====== Outcome Features ======
    outcome_placement = rng.randint(1, 101, size=n)
    outcome_alive_time = rng.randint(1, 600, size=n)
    
    # ====== Label Generation (heuristic-based) ======
    best_call = []
    for i in range(n):
        hp = height_status[i]
        ap = alive_players[i]
        zp = zone_phase[i]
        mat = mats_total[i]
        te = teammates_alive[i]
        pt = position_type[i]
        sd = storm_edge_dist[i]
        
        # Decision tree-like labeling
        if hp == 'high' and ap >= 2 and te >= 1:
            call = 'take_height'
        elif ap <= 1 and pt == 'corner':
            call = 'stick_deadside'
        elif zp == 'early' and ap >= 3:
            call = 'play_frontside'
        elif mat >= 200 and hp == 'mid':
            call = 'stabilize_box'
        elif sd < 100 and hp != 'high':
            call = 'look_for_refresh'
        else:
            call = 'drop_low'
        
        best_call.append(call)
    
    df = pd.DataFrame({
        'zone_phase': zone_phase,
        'zone_index': zone_index,
        'alive_players': alive_players,
        'teammates_alive': teammates_alive,
        'height_status': height_status,
        'position_type': position_type,
        'storm_edge_dist': storm_edge_dist,
        'mats_total': mats_total,
        'surge_above': surge_above,
        'outcome_placement': outcome_placement,
        'outcome_alive_time': outcome_alive_time,
        'best_call': best_call
    })
    
    return df

def main(args):
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    print(f'Generating {args.n} samples with seed={args.seed}...')
    df = generate_large(n=args.n, seed=args.seed)
    
    # Statistics
    print(f'\nDataset generated: {len(df)} rows × {len(df.columns)} columns')
    print('\nClass distribution:')
    print(df['best_call'].value_counts().to_string())
    
    print('\nFeature summary:')
    print(df.describe().to_string())
    
    # Save
    df.to_csv(args.output, index=False)
    print(f'\nWrote {len(df)} rows to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate large synthetic dataset')
    parser.add_argument('--n', type=int, default=5000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/processed/call_states_large.csv', help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args)
