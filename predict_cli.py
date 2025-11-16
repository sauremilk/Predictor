#!/usr/bin/env python3
"""
CLI-Tool für Predictions - nutzbar in Scripts/Automation
Beispiele:
  ./predict_cli.py --zone late --players 15 --team 2 --height low
  ./predict_cli.py --json '{"zone_phase": "mid", "alive_players": 30, ...}'
  ./predict_cli.py --csv input.csv --output results.csv
"""
import argparse
import joblib
import pandas as pd
import json
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "Vorhersage-Modell" / "models" / "baseline_pipeline_final.joblib"

def load_model():
    """Lade Modell (mit Caching)"""
    if not hasattr(load_model, 'pipeline'):
        load_model.pipeline = joblib.load(MODEL_PATH)
    return load_model.pipeline

def predict_single(zone_phase, zone_index, alive_players, teammates_alive, 
                   storm_edge_dist, mats_total, surge_above, height_status, position_type):
    """Single Prediction"""
    pipeline = load_model()
    
    game_state = {
        "zone_phase": zone_phase,
        "zone_index": zone_index,
        "alive_players": alive_players,
        "teammates_alive": teammates_alive,
        "storm_edge_dist": storm_edge_dist,
        "mats_total": mats_total,
        "surge_above": surge_above,
        "height_status": height_status,
        "position_type": position_type,
        "outcome_placement": 0,
        "outcome_alive_time": 0
    }
    
    df = pd.DataFrame([game_state])
    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]
    
    result = {
        "predicted_call": prediction,
        "confidence": float(probabilities.max()),
        "probabilities": {name: float(prob) for name, prob in zip(pipeline.classes_, probabilities)}
    }
    
    return result

def predict_from_json(json_str):
    """Prediction von JSON-String"""
    data = json.loads(json_str)
    return predict_single(
        data.get("zone_phase", "mid"),
        data.get("zone_index", 5),
        data.get("alive_players", 30),
        data.get("teammates_alive", 3),
        data.get("storm_edge_dist", 100.0),
        data.get("mats_total", 300),
        data.get("surge_above", 10),
        data.get("height_status", "mid"),
        data.get("position_type", "edge")
    )

def predict_from_csv(input_csv, output_csv):
    """Batch-Predictions aus CSV"""
    pipeline = load_model()
    df = pd.read_csv(input_csv)
    
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)
    
    df['predicted_call'] = predictions
    df['confidence'] = probabilities.max(axis=1)
    
    df.to_csv(output_csv, index=False)
    return len(df)

def main():
    parser = argparse.ArgumentParser(
        description="Predictor CLI - Robustes Command-Line Tool für Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Single Prediction
  %(prog)s --zone late --players 15 --team 2 --height low --position corner
  
  # Von JSON
  %(prog)s --json '{"zone_phase": "mid", "alive_players": 30, "teammates_alive": 3, "height_status": "mid", "position_type": "edge", "storm_edge_dist": 100, "mats_total": 300, "surge_above": 10, "zone_index": 5}'
  
  # Batch von CSV
  %(prog)s --csv input.csv --output predictions.csv
  
  # Nur Call ausgeben (für Scripts)
  %(prog)s --zone mid --players 20 --team 3 --height mid --position edge --quiet
        """
    )
    
    # Single Prediction Argumente
    parser.add_argument("--zone", choices=["early", "mid", "late"], help="Zone Phase")
    parser.add_argument("--zone-index", type=int, default=5, help="Zone Index (1-9)")
    parser.add_argument("--players", type=int, help="Alive players total")
    parser.add_argument("--team", type=int, help="Teammates alive")
    parser.add_argument("--storm", type=float, default=100.0, help="Storm edge distance")
    parser.add_argument("--mats", type=int, default=300, help="Total materials")
    parser.add_argument("--surge", type=int, default=10, help="Surge above")
    parser.add_argument("--height", choices=["low", "mid", "high"], help="Height status")
    parser.add_argument("--position", choices=["edge", "corner", "center"], help="Position type")
    
    # JSON Input
    parser.add_argument("--json", help="JSON string with game state")
    
    # CSV Batch
    parser.add_argument("--csv", help="Input CSV file for batch predictions")
    parser.add_argument("--output", help="Output CSV file (required with --csv)")
    
    # Output Format
    parser.add_argument("--quiet", action="store_true", help="Only output predicted call (for scripts)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Batch Mode
        if args.csv:
            if not args.output:
                print("Error: --output required with --csv", file=sys.stderr)
                sys.exit(1)
            count = predict_from_csv(args.csv, args.output)
            if not args.quiet:
                print(f"✅ Processed {count} predictions → {args.output}")
            sys.exit(0)
        
        # JSON Mode
        if args.json:
            result = predict_from_json(args.json)
        
        # Single Prediction Mode
        elif args.zone and args.players and args.team and args.height and args.position:
            result = predict_single(
                args.zone, args.zone_index, args.players, args.team,
                args.storm, args.mats, args.surge, args.height, args.position
            )
        else:
            parser.print_help()
            sys.exit(1)
        
        # Output
        if args.quiet:
            print(result['predicted_call'])
        elif args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Predicted Call: {result['predicted_call']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nProbabilities:")
            for call, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0.01:
                    print(f"  {call:20s}: {prob:6.2%}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
