from flask import Flask, request
import threading
import time
import math
import logging
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# Flask-Logging deaktivieren
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ── Thread-Lock & Sensordaten ─────────────────────────────────────────────────
lock       = threading.Lock()
stop_event = threading.Event()
sphero_api = None

latest_data = {
    "gravity_x": 0.0,
    "gravity_y": 0.0,
    "gravity_z": -1.0,
    "last_update": 0.0
}

# ── Kalibrierung ──────────────────────────────────────────────────────────────
calibration = {
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "done": False
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
MAX_SPEED = 60
MIN_SPEED = 40
STOP_TIME = 1.0   # Sekunden bis Stopp nach Inaktivität

def wait_for_sensor_data():
    """Wartet, bis echte Apple-Watch-Daten angekommen sind."""
    print("⏳ Warte auf Apple-Watch-Daten...")

    while True:
        with lock:
            last_update = latest_data["last_update"]

        if last_update > 0:
            print("✅ Apple-Watch-Daten empfangen.")
            return

        time.sleep(0.1)
def calibrate(sphero):
    """2 Sekunden Ruheposition messen und als Nullpunkt speichern."""

    wait_for_sensor_data()

    print("⏳ Kalibrierung: Hand flach / neutral halten (2 Sekunden)...")
    print("   Diese Position wird später als VORWÄRTS verwendet.")

    samples_gx, samples_gy, samples_gz = [], [], []

    end_time = time.time() + 2.0

    while time.time() < end_time:
        with lock:
            samples_gx.append(latest_data["gravity_x"])
            samples_gy.append(latest_data["gravity_y"])
            samples_gz.append(latest_data["gravity_z"])

        time.sleep(0.05)

    calibration["gx"] = sum(samples_gx) / len(samples_gx)
    calibration["gy"] = sum(samples_gy) / len(samples_gy)
    calibration["gz"] = sum(samples_gz) / len(samples_gz)
    calibration["done"] = True

    print(
        f"✅ Kalibriert: "
        f"gX={calibration['gx']:+.3f} | "
        f"gY={calibration['gy']:+.3f} | "
        f"gZ={calibration['gz']:+.3f}"
    )

    sphero.set_main_led(Color(r=0, g=255, b=0))

# ── Zustand erkennen aus Gravity-Vektor ───────────────────────────────────────
# ── Zustand erkennen ──────────────────────────────────────────────────────────
# ── Zustand aus Differenz zur Ruheposition ────────────────────────────────────
def get_state(gx, gy, gz):
    """
    Erkennt den Steuerzustand aus dem Gravity-Vektor.

    Ziel:
    - Kalibrierte/flache Hand        -> forward
    - Handgelenk nach rechts drehen  -> right
    - Handgelenk nach links drehen   -> left
    - Hand nach oben/unten bewegen   -> stop

    Wichtig:
    Stop wird hier bewusst hauptsächlich über die Y-Achse erkannt,
    weil sich Z auch beim Links/Rechts-Drehen stark verändert.
    """

    if not calibration["done"]:
        return "neutral"

    # Differenz zur kalibrierten Vorwärtsposition
    dx = gx - calibration["gx"]
    dy = gy - calibration["gy"]
    dz = gz - calibration["gz"]

    # -----------------------------
    # Schwellwerte
    # -----------------------------
    TURN_THRESHOLD = 0.18       # kleiner = empfindlicher für links/rechts
    STOP_Y_THRESHOLD = 0.6     # kleiner = Stop wird früher erkannt

    # -----------------------------
    # 1. STOP über Y-Achse
    # -----------------------------
    # In deinem Log:
    # Hand hoch/runter: gY ungefähr +0.99 oder -0.96
    # Handgelenk drehen: gY meistens deutlich kleiner
    if abs(dy) > STOP_Y_THRESHOLD:
        return "stop"

    # -----------------------------
    # 2. LINKS / RECHTS über X-Achse
    # -----------------------------
    if dx > TURN_THRESHOLD:
        return "right"

    if dx < -TURN_THRESHOLD:
        return "left"

    # -----------------------------
    # 3. Standard: flache Hand = vorwärts
    # -----------------------------
    return "forward"


# ── Flask Route ───────────────────────────────────────────────────────────────
@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json

    if data is None:
        return "No data", 400

    try:
        gx = float(data.get("gravityX", 0.0))
        gy = float(data.get("gravityY", 0.0))
        gz = float(data.get("gravityZ", -1.0))
    except ValueError:
        return "Invalid gravity data", 400

    with lock:
        latest_data["gravity_x"] = gx
        latest_data["gravity_y"] = gy
        latest_data["gravity_z"] = gz
        latest_data["last_update"] = time.time()

    return "OK", 200


# ── Flask Server ──────────────────────────────────────────────────────────────
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
        sphero_api = sphero

        sphero_heading = 0
        sphero.set_heading(0)
        sphero.set_main_led(Color(r=255, g=165, b=0))  # Orange = Kalibrierung

        calibrate(sphero)

        print("✅ Bereit! Bewege deine Hand.")
        print("   Flache Hand     = Vorwärts")
        print("   Hand rechts     = Rechts fahren")
        print("   Hand links      = Links fahren")
        print("   Hand nach oben  = Stopp")

        sphero.set_heading(0)
        sphero.set_main_led(Color(r=255, g=255, b=255))

        last_move_time = time.time()
        last_print_time = 0
        is_stopped = True

        while not stop_event.is_set():
            with lock:
                gx = latest_data["gravity_x"]
                gy = latest_data["gravity_y"]
                gz = latest_data["gravity_z"]
                last_update = latest_data["last_update"]

            now = time.time()

            # Sicherheitsstopp, falls keine Apple-Watch-Daten mehr kommen
            if now - last_update > 1.0:
                if not is_stopped:
                    sphero.stop_roll(int(sphero_heading))
                    sphero.set_main_led(Color(r=255, g=0, b=0))
                    is_stopped = True
                    print("⚠️ Keine neuen Watch-Daten → Stopp")

                time.sleep(0.05)
                continue

            state = get_state(gx, gy, gz)

            # Debug nur alle 0.3 Sekunden
            if now - last_print_time > 0.3:
                dx = gx - calibration["gx"]
                dy = gy - calibration["gy"]
                dz = gz - calibration["gz"]

                print(
                    f"gX={gx:+.2f} | gY={gy:+.2f} | gZ={gz:+.2f} || "
                    f"dx={dx:+.2f} | dy={dy:+.2f} | dz={dz:+.2f} → {state}"
                )

                last_print_time = now

            if state == "right":
                sphero_heading = (sphero_heading + 3) % 360
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=255, g=100, b=0))
                last_move_time = now
                is_stopped = False

            elif state == "left":
                sphero_heading = (sphero_heading - 3) % 360
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=0, g=200, b=255))
                last_move_time = now
                is_stopped = False

            elif state == "forward":
                sphero.roll(int(sphero_heading), MAX_SPEED, 0.1)
                sphero.set_main_led(Color(r=0, g=255, b=0))
                last_move_time = now
                is_stopped = False

            elif state == "stop":
                if not is_stopped:
                    sphero.stop_roll(int(sphero_heading))
                    sphero.set_main_led(Color(r=255, g=0, b=0))
                    is_stopped = True
                    print("🛑 Stopp!")

            else:
                if not is_stopped and now - last_move_time > STOP_TIME:
                    sphero.stop_roll(int(sphero_heading))
                    sphero.set_main_led(Color(r=255, g=0, b=0))
                    is_stopped = True
                    print("⏸️ Auto-Stopp")

            time.sleep(0.05)

        print("🔌 Sphero wird getrennt...")

        try:
            sphero.stop_roll(0)
            sphero.set_main_led(Color(r=0, g=0, b=0))
            time.sleep(0.5)
        except Exception:
            pass

        print("✅ Sphero getrennt.")


#Main 
if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()

    try:
        control_sphero()
    except KeyboardInterrupt:
        print("\n⛔ Ctrl+C erkannt – beende...")
        stop_event.set()