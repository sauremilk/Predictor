"""Data Quality and Validation Pipeline.

Checks input data for consistency with training data distribution,
missing values, outliers, and feature validity.

Usage:
    python3 src/data_validation.py --input data/call_states_real_labeled.csv [--reference data/processed/call_states_synth_large.csv]
"""
import argparse
import os
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

ALLOWED_CLASSES = [
    'stick_deadside', 'play_frontside', 'take_height',
    'stabilize_box', 'look_for_refresh', 'drop_low'
]

NUMERIC_FEATURES = [
    'zone_index', 'alive_players', 'teammates_alive',
    'storm_edge_dist', 'mats_total', 'surge_above',
    'outcome_placement', 'outcome_alive_time'
]

CATEGORICAL_FEATURES = [
    'zone_phase', 'height_status', 'position_type'
]

FEATURE_BOUNDS = {
    'zone_index': (0, 10),
    'alive_players': (1, 100),
    'teammates_alive': (0, 4),
    'storm_edge_dist': (-1000, 5000),
    'mats_total': (0, 3000),
    'surge_above': (-1000, 5000),
    'outcome_placement': (1, 100),
    'outcome_alive_time': (0, 3600)
}

# ============================================================================
# Validation Functions
# ============================================================================

def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """Check percentage of missing values per column."""
    missing_pct = (df.isnull().sum() / len(df)) * 100
    return missing_pct[missing_pct > 0].to_dict() if (missing_pct > 0).any() else {}

def check_class_distribution(df: pd.DataFrame, reference_df: pd.DataFrame = None) -> Dict[str, any]:
    """Check class distribution and compare with reference if provided."""
    if 'best_call' not in df.columns:
        return {'status': 'skipped', 'reason': 'best_call column not found'}
    
    input_dist = df['best_call'].value_counts(normalize=True).to_dict()
    
    result = {
        'distribution': input_dist,
        'invalid_classes': list(set(df['best_call'].unique()) - set(ALLOWED_CLASSES)),
        'class_count': len(df['best_call'].unique())
    }
    
    if reference_df is not None and 'best_call' in reference_df.columns:
        ref_dist = reference_df['best_call'].value_counts(normalize=True).to_dict()
        # Chi-square test
        chi2, pval = stats.chisquare(
            [input_dist.get(c, 0) * len(df) for c in ALLOWED_CLASSES],
            [ref_dist.get(c, 0) * len(reference_df) for c in ALLOWED_CLASSES]
        )
        result['chi_square_test'] = {
            'statistic': float(chi2),
            'p_value': float(pval),
            'distribution_shift': 'yes' if pval < 0.05 else 'no'
        }
    
    return result

def check_numeric_features(df: pd.DataFrame, reference_df: pd.DataFrame = None) -> Dict[str, Dict[str, float]]:
    """Check numeric feature ranges and distributions."""
    result = {}
    
    for feat in NUMERIC_FEATURES:
        if feat not in df.columns:
            continue
        
        col_data = df[feat].dropna()
        
        feat_result = {
            'count': int(col_data.count()),
            'missing': int(df[feat].isnull().sum()),
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'median': float(col_data.median())
        }
        
        # Check bounds
        bounds = FEATURE_BOUNDS.get(feat)
        if bounds:
            out_of_bounds = ((col_data < bounds[0]) | (col_data > bounds[1])).sum()
            feat_result['out_of_bounds'] = {
                'count': int(out_of_bounds),
                'percentage': float((out_of_bounds / len(col_data)) * 100) if len(col_data) > 0 else 0,
                'expected_range': bounds
            }
        
        # Compare with reference if provided
        if reference_df is not None and feat in reference_df.columns:
            ref_data = reference_df[feat].dropna()
            # KS test for distribution comparison
            ks_stat, ks_pval = stats.ks_2samp(col_data, ref_data)
            feat_result['distribution_test'] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'distribution_shift': 'yes' if ks_pval < 0.05 else 'no',
                'reference_mean': float(ref_data.mean()),
                'reference_std': float(ref_data.std())
            }
        
        result[feat] = feat_result
    
    return result

def check_categorical_features(df: pd.DataFrame) -> Dict[str, Dict]:
    """Check categorical feature values."""
    result = {}
    
    valid_values = {
        'zone_phase': ['early', 'mid', 'late', 'endgame'],
        'height_status': ['low', 'mid', 'high'],
        'position_type': ['center', 'edge', 'corner']
    }
    
    for feat, valid in valid_values.items():
        if feat not in df.columns:
            continue
        
        col_data = df[feat].dropna()
        unique_vals = col_data.unique()
        invalid = set(unique_vals) - set(valid)
        
        result[feat] = {
            'unique_values': list(unique_vals),
            'value_counts': col_data.value_counts().to_dict(),
            'invalid_values': list(invalid),
            'missing': int(df[feat].isnull().sum())
        }
    
    return result

def check_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """Check for duplicate rows."""
    total_dupes = df.duplicated().sum()
    
    result = {
        'total_duplicates': int(total_dupes),
        'total_rows': len(df),
        'duplicate_percentage': float((total_dupes / len(df)) * 100)
    }
    
    # Check duplicates by match_id + frame_id
    if 'match_id' in df.columns and 'frame_id' in df.columns:
        dupes_by_key = df.duplicated(subset=['match_id', 'frame_id']).sum()
        result['duplicates_by_key'] = int(dupes_by_key)
    
    return result

def generate_validation_report(
    input_df: pd.DataFrame,
    reference_df: pd.DataFrame = None,
    output_path: str = None
) -> Dict:
    """Generate comprehensive validation report."""
    
    logger.info("Starting data validation...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'input_rows': len(input_df),
        'checks': {}
    }
    
    # Run all checks
    logger.info("Checking for missing values...")
    report['checks']['missing_values'] = check_missing_values(input_df)
    
    logger.info("Checking class distribution...")
    report['checks']['class_distribution'] = check_class_distribution(input_df, reference_df)
    
    logger.info("Checking numeric features...")
    report['checks']['numeric_features'] = check_numeric_features(input_df, reference_df)
    
    logger.info("Checking categorical features...")
    report['checks']['categorical_features'] = check_categorical_features(input_df)
    
    logger.info("Checking for duplicates...")
    report['checks']['duplicates'] = check_duplicates(input_df)
    
    # Summary
    issues = []
    if report['checks']['missing_values']:
        issues.append(f"Found missing values: {report['checks']['missing_values']}")
    
    invalid_classes = report['checks']['class_distribution'].get('invalid_classes', [])
    if invalid_classes:
        issues.append(f"Invalid classes found: {invalid_classes}")
    
    for feat, info in report['checks']['numeric_features'].items():
        if info.get('out_of_bounds', {}).get('percentage', 0) > 5:
            issues.append(f"Feature {feat}: {info['out_of_bounds']['percentage']:.1f}% out of bounds")
    
    for feat, info in report['checks']['categorical_features'].items():
        if info.get('invalid_values'):
            issues.append(f"Feature {feat}: Invalid values {info['invalid_values']}")
    
    report['summary'] = {
        'total_issues': len(issues),
        'issues': issues,
        'validation_status': 'PASS' if len(issues) == 0 else 'WARNING' if len(issues) < 3 else 'FAIL'
    }
    
    # Save report
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    return report

def print_validation_report(report: Dict):
    """Print validation report in readable format."""
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Input rows: {report['input_rows']}")
    print(f"Status: {report['summary']['validation_status']}\n")
    
    if report['summary']['issues']:
        print("ISSUES FOUND:")
        for issue in report['summary']['issues']:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ No issues found!")
    
    print("\n" + "="*80)

# ============================================================================
# Main
# ============================================================================

def main(args):
    # Load input data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    logger.info(f"Loading input data from {args.input}")
    input_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(input_df)} rows")
    
    # Load reference data if provided
    reference_df = None
    if args.reference:
        if not os.path.exists(args.reference):
            logger.warning(f"Reference file not found: {args.reference}")
        else:
            logger.info(f"Loading reference data from {args.reference}")
            reference_df = pd.read_csv(args.reference)
            logger.info(f"Loaded {len(reference_df)} reference rows")
    
    # Generate report
    output_path = args.output or f"models/validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = generate_validation_report(input_df, reference_df, output_path)
    
    # Print summary
    print_validation_report(report)
    
    # Exit with appropriate code
    if report['summary']['validation_status'] == 'FAIL':
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data validation and quality checks')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file to validate')
    parser.add_argument('--reference', type=str, default=None, help='Reference CSV for distribution comparison')
    parser.add_argument('--output', type=str, default=None, help='Output JSON report path')
    args = parser.parse_args()
    main(args)
