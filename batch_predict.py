#!/usr/bin/env python3
"""
Batch-Processing: Verarbeite viele Game States auf einmal aus CSV/JSONL
"""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "Vorhersage-Modell" / "models" / "baseline_pipeline_final.joblib"

def predict_batch_from_csv(input_csv: str, output_csv: str):
    """Batch-Predictions aus CSV-Datei"""
    print(f"📦 Lade Modell...")
    pipeline = joblib.load(MODEL_PATH)
    
    print(f"📂 Lese Input: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   → {len(df)} Game States gefunden")
    
    # Predictions
    print(f"🔮 Führe Predictions aus...")
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)
    
    # Confidence (max probability)
    confidences = probabilities.max(axis=1)
    
    # Ergebnisse hinzufügen
    df['predicted_call'] = predictions
    df['confidence'] = confidences
    
    # Optional: Top-3 Wahrscheinlichkeiten
    class_names = pipeline.classes_
    for i in range(min(3, len(class_names))):
        top_idx = probabilities.argsort(axis=1)[:, -(i+1)]
        df[f'top_{i+1}_call'] = [class_names[idx] for idx in top_idx]
        df[f'top_{i+1}_prob'] = [probabilities[j, idx] for j, idx in enumerate(top_idx)]
    
    # Speichern
    df.to_csv(output_csv, index=False)
    print(f"✅ Ergebnisse gespeichert: {output_csv}")
    print(f"   → {len(df)} Predictions geschrieben")
    
    # Statistik
    print(f"\n📊 Verteilung der Predictions:")
    print(df['predicted_call'].value_counts())
    
    return df

if __name__ == "__main__":
    # Beispiel: Nutze Demo-Daten
    input_file = "Vorhersage-Modell/data/call_states_demo.csv"
    output_file = "batch_predictions_output.csv"
    
    results = predict_batch_from_csv(input_file, output_file)
    
    print(f"\n📋 Beispiel-Ergebnisse (erste 5 Zeilen):")
    print(results[['zone_phase', 'alive_players', 'predicted_call', 'confidence']].head())
