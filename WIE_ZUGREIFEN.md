# 🌐 So greifst du auf das Web-Interface zu

## Schritt 1: Port 8000 öffentlich machen

1. **Klicke unten in VS Code auf den Tab "PORTS"** (neben PROBLEMS, OUTPUT, TERMINAL)
2. **Finde die Zeile mit Port 8000** in der Liste
3. **Rechtsklick auf Port 8000** → **Port Visibility** → **Public** auswählen

## Schritt 2: URL öffnen

Nach dem Port-Freigabe:

1. **In der PORTS-Liste**: Klicke auf das **Globus-Symbol 🌐** neben Port 8000
2. Es öffnet sich eine URL wie: `https://obscure-space-cod-r4xjxq6gx6qp3pjw6-8000.app.github.dev`

## Wichtige URLs (nach Freigabe):

Ersetze `<BASE-URL>` mit der URL aus dem PORTS-Tab:

### Web-Interface (Haupt-UI):
```
<BASE-URL>/static/
```
Beispiel: `https://obscure-space-cod-r4xjxq6gx6qp3pjw6-8000.app.github.dev/static/`

### API-Dokumentation (Swagger):
```
<BASE-URL>/docs
```

### Health Check:
```
<BASE-URL>/health
```

## Alternativ: Im Codespace direkt

Wenn du im VS Code Browser innerhalb des Codespaces bist, nutze einfach:
- http://localhost:8000/static/
- http://localhost:8000/docs

---

## Problembehandlung

**Web-Interface lädt nicht?**
1. Prüfe ob Server läuft: `./manage_api.sh status`
2. Prüfe Port-Visibility im PORTS-Tab
3. Stelle sicher, dass du `/static/` am Ende der URL hast
4. Versuche die URL in einem **Inkognito-Fenster**

**Port 8000 erscheint nicht im PORTS-Tab?**
1. Warte 5-10 Sekunden (Auto-Detection)
2. Oder füge manuell hinzu: Klicke auf "Add Port" → Gib `8000` ein

**Server antwortet nicht?**
```bash
cd /workspaces/Predictor/Vorhersage-Modell
./manage_api.sh restart
```
