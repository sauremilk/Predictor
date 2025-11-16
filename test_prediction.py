#!/usr/bin/env python3
"""
Einfaches Test-Skript um das Prediction-Modell zu nutzen
"""
import requests
import json

# API URL
API_URL = "http://localhost:8000/predict"

# Beispiel Game State
game_state = {
    "zone_phase": "late",      # early/mid/late
    "zone_index": 7,
    "alive_players": 15,
    "teammates_alive": 2,
    "storm_edge_dist": 50.0,
    "mats_total": 150,
    "surge_above": 5,
    "height_status": "low",    # low/mid/high
    "position_type": "corner", # edge/corner/center
    "match_id": "my_match_001",
    "frame_id": "0042"
}

print("🎮 Sende Game State an Modell...")
print(f"Situation: {game_state['zone_phase']} game, Zone {game_state['zone_index']}")
print(f"Players alive: {game_state['alive_players']}, Team: {game_state['teammates_alive']}")
print(f"Position: {game_state['position_type']}, Height: {game_state['height_status']}")
print()

# API Request
response = requests.post(API_URL, json=game_state)

if response.status_code == 200:
    result = response.json()
    
    print("✅ Prediction erfolgreich!")
    print(f"📊 Empfohlener Call: {result['predicted_call']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print()
    print("Wahrscheinlichkeiten für alle Optionen:")
    for call, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        if prob > 0:
            bar = "█" * int(prob * 50)
            print(f"  {call:20s}: {prob:5.1%} {bar}")
else:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
