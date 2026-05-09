from flask import Flask, request
import threading
import time
import logging
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

lock       = threading.Lock()
stop_event = threading.Event()
sphero_api = None

latest_data = {"gx": 0.0, "gy": 0.0, "gz": 0.0}

# ── Konfiguration ─────────────────────────────────────────────────────────────
MAX_SPEED = 50   # Geschwindigkeit 0-255. Erhöhen = schneller, verringern = langsamer
STOP_TIME = 1.5  # Sekunden bis Auto-Stopp. Verringern = schneller stoppen

# ── Schwellenwerte (aus deinen echten Messdaten) ──────────────────────────────
# RECHTS: gY sinkt auf ca. -0.99
# → Erhöhe -0.80 auf z.B. -0.70 wenn Rechts zu schwer auszulösen
# → Verringere auf -0.90 wenn Rechts versehentlich ausgelöst wird
GY_RIGHT_THRESHOLD = -0.80

# LINKS: gY steigt auf ca. +0.97
# → Verringere +0.80 auf z.B. +0.70 wenn Links zu schwer auszulösen
# → Erhöhe auf +0.90 wenn Links versehentlich ausgelöst wird
GY_LEFT_THRESHOLD = +0.80

# VORWÄRTS: gZ sinkt auf ca. -0.96, gX wird negativ ca. -0.17
# → gZ: Verringere -0.75 auf -0.85 wenn Vorwärts zu leicht auslöst
# → gZ: Erhöhe auf -0.65 wenn Vorwärts zu schwer auszulösen ist
GZ_FORWARD_THRESHOLD = -0.9
# → gX: Verringere -0.05 weiter wenn Normal fälschlich als Vorwärts erkannt
GX_FORWARD_MAX = +0.1

# NORMAL/UNTEN: gX steigt auf ca. +0.95
# → Verringere +0.70 wenn Normal zu früh erkannt wird
# → Erhöhe auf +0.85 wenn Normal zu schwer zu erkennen ist
GX_NEUTRAL_THRESHOLD = +0.12


# ── Zustandserkennung – rein aus Messdaten ────────────────────────────────────
def get_state(gx, gy, gz):
    """
    Reihenfolge ist wichtig: eindeutigste Zustände zuerst prüfen.

    RECHTS:   gY ≈ -0.99  → einzigartig, zuerst prüfen
    LINKS:    gY ≈ +0.97  → einzigartig, zuerst prüfen
    NORMAL:   gX ≈ +0.95  → Hand hängt nach unten → neutral
    VORWÄRTS: gZ ≈ -0.96, gX negativ → Hand flach nach vorne
    """

    # RECHTS: gY stark negativ (Uhr zeigt nach rechts)
    # Deine Werte: gY ≈ -0.99
    if gy < GY_RIGHT_THRESHOLD:
        return "right"

    # LINKS: gY stark positiv (Uhr zeigt nach links)
    # Deine Werte: gY ≈ +0.97
    if gy > GY_LEFT_THRESHOLD:
        return "left"

    # NORMAL: gX stark positiv = Hand hängt nach unten → kein Fahren
    # Deine Werte: gX ≈ +0.95
    if gx > GX_NEUTRAL_THRESHOLD:
        return "neutral"

    # VORWÄRTS: gZ stark negativ UND gX negativ = Hand flach nach vorne
    # Deine Werte: gZ ≈ -0.96, gX ≈ -0.17
    if gz < GZ_FORWARD_THRESHOLD and gx < GX_FORWARD_MAX:
        return "forward"

    # Alles andere = neutral (Übergangsbewegungen)
    return "neutral"


# ── Flask Route ───────────────────────────────────────────────────────────────
@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json
    if data is None:
        return "No data", 400

    gx = float(data.get("gravityX", 0))
    gy = float(data.get("gravityY", 0))
    gz = float(data.get("gravityZ", 0))

    state = get_state(gx, gy, gz)
    print(f"gX={gx:+.2f} | gY={gy:+.2f} | gZ={gz:+.2f} → {state}")

    with lock:
        latest_data["gx"] = gx
        latest_data["gy"] = gy
        latest_data["gz"] = gz

    return "OK", 200


def run_server():
    app.run(host='0.0.0.0', port=56671, debug=False)


# ── Sphero Steuerung ──────────────────────────────────────────────────────────
def control_sphero():
    global sphero_api
    print("🔍 Suche Sphero BOLT...")
    toy = scanner.find_toy()
    if not toy:
        print("❌ Kein Sphero gefunden!")
        return

    print("✅ Sphero verbunden!")

    with SpheroEduAPI(toy) as sphero:
        sphero_api     = sphero
        sphero_heading = 0
        last_move_time = time.time()
        is_stopped     = True

        sphero.set_heading(0)
        sphero.set_main_led(Color(r=255, g=255, b=255))  # Weiß = bereit
        print("\n✅ Bereit! Steuerung:")
        print("   Hand nach unten  = Neutral/Stopp")
        print("   Hand flach vorne = Vorwärts  🟢")
        print("   Uhr nach rechts  = Rechts    🟠")
        print("   Uhr nach links   = Links     🔵\n")

        while not stop_event.is_set():
            with lock:
                gx = latest_data["gx"]
                gy = latest_data["gy"]
                gz = latest_data["gz"]

            state = get_state(gx, gy, gz)

            if state == "right":
                # Heading +3° pro Schritt nach rechts drehen und fahren
                # Erhöhe 3 auf 5 wenn Kurve zu langsam
                # Verringere auf 2 wenn Kurve zu abrupt
                sphero_heading = (sphero_heading + 10) % 360
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=255, g=100, b=0))  # Orange
                last_move_time = time.time()
                is_stopped     = False

            elif state == "left":
                # Heading -3° pro Schritt nach links drehen und fahren
                sphero_heading = (sphero_heading - 10) % 360
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=0, g=200, b=255))  # Cyan
                last_move_time = time.time()
                is_stopped     = False

            elif state == "forward":
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=0, g=255, b=0))    # Grün
                last_move_time = time.time()
                is_stopped     = False

            elif state == "neutral":
                # Auto-Stopp nach STOP_TIME Sekunden Inaktivität
                if not is_stopped and time.time() - last_move_time > STOP_TIME:
                    sphero.stop_roll(int(sphero_heading))
                    sphero.set_main_led(Color(r=255, g=0, b=0))  # Rot
                    is_stopped = True
                    print("⏸️  Auto-Stopp")

            time.sleep(0.05)

        # Sauber beenden
        print("\n🔌 Trenne Sphero...")
        try:
            sphero.stop_roll(0)
            sphero.set_main_led(Color(r=0, g=0, b=0))
            time.sleep(0.5)
        except Exception:
            pass
        print("✅ Getrennt.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    try:
        control_sphero()
    except KeyboardInterrupt:
        print("\n⛔ Beende...")
        stop_event.set()