"""
QUICKSTART - Direkte Python-Integration
========================================

Die einfachste Art, das Modell zu nutzen.
"""

from predictor_lib import PredictorModel

# 1. Modell laden (einmalig beim Start)
predictor = PredictorModel()

# 2. Deine Game-Daten
my_game_state = {
    "zone_phase": "mid",        # early/mid/late
    "alive_players": 25,
    "teammates_alive": 3,
    "height_status": "high",    # low/mid/high
    "position_type": "edge",    # edge/corner/center
    "mats_total": 400,
    "storm_edge_dist": 120.0
}

# 3. Prediction abrufen
result = predictor.predict(**my_game_state)

# 4. Ergebnis nutzen
print(f"\n🎮 Situation: {my_game_state['zone_phase']} game, "
      f"{my_game_state['alive_players']} players alive")
print(f"📊 AI Empfehlung: {result['predicted_call']}")
print(f"🎯 Confidence: {result['confidence']:.0%}")

if result['confidence'] > 0.6:
    print("✅ Starke Empfehlung - hohe Confidence!")
else:
    print("⚠️  Unsichere Situation - prüfe Alternativen:")
    top_3 = predictor.get_top_n_predictions(**my_game_state, n=3)
    for i, call in enumerate(top_3[1:], 2):  # Skip first (already shown)
        print(f"   {i}. {call['call']}: {call['probability']:.0%}")

print("\n" + "="*60)
print("💡 Tipp: Ändere my_game_state oben und führe erneut aus!")
print("="*60)
