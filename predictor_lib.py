"""
Predictor Library - Direkte Python-Integration
==============================================

Einfache Bibliothek für die Integration des Prediction-Modells in deine Python-Apps.

Installation:
    Kopiere diese Datei in dein Projekt oder installiere als Modul

Beispiele siehe unten in __main__
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class PredictorModel:
    """
    Wrapper für das Best-Call Prediction Modell
    
    Beispiel:
        >>> predictor = PredictorModel()
        >>> result = predictor.predict(
        ...     zone_phase="late",
        ...     alive_players=15,
        ...     teammates_alive=2,
        ...     height_status="low",
        ...     position_type="corner"
        ... )
        >>> print(result['predicted_call'])
        'take_height'
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialisiere das Modell
        
        Args:
            model_path: Pfad zur Joblib-Pipeline (optional)
        """
        if model_path is None:
            # Auto-detect: Suche Modell relativ zu dieser Datei
            base_path = Path(__file__).parent
            model_path = base_path / "Vorhersage-Modell" / "models" / "baseline_pipeline_final.joblib"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")
        
        print(f"📦 Lade Modell von: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.classes_ = self.pipeline.classes_
        print(f"✅ Modell geladen. Verfügbare Klassen: {list(self.classes_)}")
    
    def predict(
        self,
        zone_phase: str,
        alive_players: int,
        teammates_alive: int,
        height_status: str,
        position_type: str,
        zone_index: int = 5,
        storm_edge_dist: float = 100.0,
        mats_total: int = 300,
        surge_above: int = 10,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Führe eine einzelne Prediction aus
        
        Args:
            zone_phase: "early", "mid", oder "late"
            alive_players: Anzahl lebender Spieler gesamt
            teammates_alive: Anzahl Teammitglieder
            height_status: "low", "mid", oder "high"
            position_type: "edge", "corner", oder "center"
            zone_index: Zone-Index (1-9), default=5
            storm_edge_dist: Distanz zum Storm, default=100.0
            mats_total: Verfügbare Materialien, default=300
            surge_above: Surge-Vorteil, default=10
            return_probabilities: Wenn True, gibt Wahrscheinlichkeiten zurück
        
        Returns:
            Dict mit predicted_call, confidence, und optional probabilities
        """
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
            "outcome_placement": 0,  # Dummy-Werte (nicht für Prediction genutzt)
            "outcome_alive_time": 0
        }
        
        df = pd.DataFrame([game_state])
        prediction = self.pipeline.predict(df)[0]
        probabilities = self.pipeline.predict_proba(df)[0]
        
        result = {
            "predicted_call": prediction,
            "confidence": float(probabilities.max())
        }
        
        if return_probabilities:
            result["probabilities"] = {
                name: float(prob) 
                for name, prob in zip(self.classes_, probabilities)
            }
        
        return result
    
    def predict_batch(
        self, 
        game_states: List[Dict],
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Führe Batch-Predictions aus (schneller für mehrere Predictions)
        
        Args:
            game_states: Liste von Dicts mit Game-State-Features
            return_probabilities: Wenn True, gibt Wahrscheinlichkeiten zurück
        
        Returns:
            Liste von Prediction-Dicts
        """
        # Ergänze fehlende Features mit Defaults
        for state in game_states:
            state.setdefault("zone_index", 5)
            state.setdefault("storm_edge_dist", 100.0)
            state.setdefault("mats_total", 300)
            state.setdefault("surge_above", 10)
            state.setdefault("outcome_placement", 0)
            state.setdefault("outcome_alive_time", 0)
        
        df = pd.DataFrame(game_states)
        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                "predicted_call": pred,
                "confidence": float(probs.max())
            }
            
            if return_probabilities:
                result["probabilities"] = {
                    name: float(prob) 
                    for name, prob in zip(self.classes_, probs)
                }
            
            results.append(result)
        
        return results
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führe Predictions auf einem DataFrame aus und füge Ergebnisse hinzu
        
        Args:
            df: DataFrame mit Game-State-Features
        
        Returns:
            DataFrame mit zusätzlichen Spalten: predicted_call, confidence
        """
        # Kopie erstellen um Original nicht zu ändern
        df_result = df.copy()
        
        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)
        
        df_result['predicted_call'] = predictions
        df_result['confidence'] = probabilities.max(axis=1)
        
        return df_result
    
    def get_top_n_predictions(
        self,
        zone_phase: str,
        alive_players: int,
        teammates_alive: int,
        height_status: str,
        position_type: str,
        n: int = 3,
        **kwargs
    ) -> List[Dict]:
        """
        Gibt die Top-N wahrscheinlichsten Calls zurück
        
        Returns:
            Liste von Dicts mit call und probability, sortiert nach Wahrscheinlichkeit
        """
        result = self.predict(
            zone_phase=zone_phase,
            alive_players=alive_players,
            teammates_alive=teammates_alive,
            height_status=height_status,
            position_type=position_type,
            **kwargs
        )
        
        # Sortiere nach Wahrscheinlichkeit
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"call": call, "probability": prob}
            for call, prob in sorted_probs[:n]
        ]


# ============================================================================
# BEISPIELE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTOR - DIREKTE PYTHON-INTEGRATION")
    print("=" * 70)
    print()
    
    # Modell initialisieren
    predictor = PredictorModel()
    print()
    
    # ========================================================================
    # Beispiel 1: Einfache Single Prediction
    # ========================================================================
    print("📍 BEISPIEL 1: Single Prediction")
    print("-" * 70)
    
    result = predictor.predict(
        zone_phase="late",
        alive_players=15,
        teammates_alive=2,
        height_status="low",
        position_type="corner",
        storm_edge_dist=50.0,
        mats_total=150
    )
    
    print(f"Situation: Late Game, 15 alive, 2 teammates, low height, corner")
    print(f"✅ Empfehlung: {result['predicted_call']}")
    print(f"🎯 Confidence: {result['confidence']:.1%}")
    print()
    
    # ========================================================================
    # Beispiel 2: Top-N Predictions
    # ========================================================================
    print("📍 BEISPIEL 2: Top-3 Empfehlungen")
    print("-" * 70)
    
    top_calls = predictor.get_top_n_predictions(
        zone_phase="mid",
        alive_players=30,
        teammates_alive=3,
        height_status="mid",
        position_type="edge",
        n=3
    )
    
    print("Situation: Mid Game, 30 alive, 3 teammates, mid height, edge")
    for i, call_info in enumerate(top_calls, 1):
        bar = "█" * int(call_info['probability'] * 50)
        print(f"{i}. {call_info['call']:20s}: {call_info['probability']:5.1%} {bar}")
    print()
    
    # ========================================================================
    # Beispiel 3: Batch Predictions
    # ========================================================================
    print("📍 BEISPIEL 3: Batch Processing (mehrere Situationen)")
    print("-" * 70)
    
    situations = [
        {
            "zone_phase": "early",
            "alive_players": 80,
            "teammates_alive": 4,
            "height_status": "mid",
            "position_type": "center",
            "mats_total": 500
        },
        {
            "zone_phase": "mid",
            "alive_players": 30,
            "teammates_alive": 3,
            "height_status": "high",
            "position_type": "edge",
            "mats_total": 300
        },
        {
            "zone_phase": "late",
            "alive_players": 10,
            "teammates_alive": 2,
            "height_status": "low",
            "position_type": "corner",
            "mats_total": 100
        }
    ]
    
    results = predictor.predict_batch(situations, return_probabilities=False)
    
    for i, (situation, result) in enumerate(zip(situations, results), 1):
        print(f"{i}. {situation['zone_phase']:5s} | {situation['alive_players']:2d} alive | "
              f"{situation['height_status']:4s} → {result['predicted_call']:20s} "
              f"({result['confidence']:.0%})")
    print()
    
    # ========================================================================
    # Beispiel 4: DataFrame Integration
    # ========================================================================
    print("📍 BEISPIEL 4: DataFrame-Integration (CSV-Workflow)")
    print("-" * 70)
    
    # Simuliere CSV-Daten
    import pandas as pd
    
    csv_data = pd.DataFrame([
        {"zone_phase": "early", "zone_index": 1, "alive_players": 90, "teammates_alive": 4,
         "storm_edge_dist": 500, "mats_total": 600, "surge_above": 20, 
         "height_status": "mid", "position_type": "center",
         "outcome_placement": 0, "outcome_alive_time": 0},
        {"zone_phase": "mid", "zone_index": 5, "alive_players": 40, "teammates_alive": 3,
         "storm_edge_dist": 200, "mats_total": 400, "surge_above": 15, 
         "height_status": "high", "position_type": "edge",
         "outcome_placement": 0, "outcome_alive_time": 0},
        {"zone_phase": "late", "zone_index": 8, "alive_players": 12, "teammates_alive": 2,
         "storm_edge_dist": 50, "mats_total": 150, "surge_above": 5, 
         "height_status": "low", "position_type": "corner",
         "outcome_placement": 0, "outcome_alive_time": 0},
    ])
    
    # Predictions hinzufügen
    result_df = predictor.predict_from_dataframe(csv_data)
    
    print("Input CSV mit Predictions:")
    print(result_df[['zone_phase', 'alive_players', 'height_status', 
                      'predicted_call', 'confidence']].to_string(index=False))
    print()
    
    # ========================================================================
    # Beispiel 5: Integration in eigene Anwendung
    # ========================================================================
    print("📍 BEISPIEL 5: Integration-Template")
    print("-" * 70)
    print("""
# So integrierst du es in deinen Code:

from predictor_lib import PredictorModel

# Einmal initialisieren (z.B. beim App-Start)
predictor = PredictorModel()

# In deiner Game-Loop:
def analyze_game_state(game_data):
    result = predictor.predict(
        zone_phase=game_data['zone'],
        alive_players=game_data['players'],
        teammates_alive=game_data['team_size'],
        height_status=game_data['height'],
        position_type=game_data['position']
    )
    
    return result['predicted_call'], result['confidence']

# Beispiel-Nutzung:
# call, confidence = analyze_game_state(current_game_state)
# if confidence > 0.7:
#     print(f"Starke Empfehlung: {call}")
    """)
    
    print()
    print("=" * 70)
    print("✅ Alle Beispiele erfolgreich ausgeführt!")
    print("=" * 70)
