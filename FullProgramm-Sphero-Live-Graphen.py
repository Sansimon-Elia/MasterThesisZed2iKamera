from flask import Flask, request
import threading
import time
import math
import matplotlib.pyplot as plt
from collections import deque
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
import logging
import os

# Flask-Logging deaktivieren
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ── Thread-Lock & Datenspeicher ───────────────────────────────────────────────
lock = threading.Lock()
MAX_POINTS = 300

intensity_values  = deque(maxlen=MAX_POINTS)
heart_rate_values = deque(maxlen=MAX_POINTS)
time_values       = deque(maxlen=MAX_POINTS)
start_time        = time.time()

# ── Sensor-Rohdaten (für Sphero) ──────────────────────────────────────────────
latest_data = {
    "accel_x":    0.0,   # neu
    "accel_y":    0.0,
    "roll":       0.0,
    "pitch":      0.0,
    "heart_rate": 0,
}

#globale variablen zum Stoppen von Hauptthread 
sphero_api = None  # Globale Referenz auf Sphero
stop_event  = threading.Event()  # Sauberes Stopp-Signal

# ── Konfiguration ─────────────────────────────────────────────────────────────
HR_WARN            = 100
HR_DANGER          = 120
SMOOTH_WIN         = 10

# Sphero
MAX_SPEED          = 255
MIN_SPEED          = 80
STOP_THRESHOLD     = 0.03   # Unter diesem Wert = keine Bewegung
STOP_TIME          = 1.5    # Sekunden bis Stopp
ROTATION_SCALE     = 60     # Wie stark Rotation die Richtung ändert (Grad)
ROTATION_DEADZONE  = 0.15   # Kleine Zitterbewegungen ignorieren


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def moving_average(data: list, window: int):
    if len(data) < window:
        return list(data)
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(list(data)[start:i+1]) / (i - start + 1))
    return result


def compute_load_index(intensities: list, heart_rates: list) -> list:
    if not intensities:
        return []
    max_i = max(intensities) if max(intensities) > 0 else 1
    load  = []
    for i, hr in zip(intensities, heart_rates):
        norm_i  = min(i / max_i, 1.0)
        norm_hr = max(min((hr - 60) / 120, 1.0), 0)
        load.append((0.6 * norm_i + 0.4 * norm_hr) * 100)
    return load


# ── Flask Route ───────────────────────────────────────────────────────────────
@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json
    if data is None:
        return "No data", 400

    x     = data.get("motionUserAccelerationX", 0)
    y     = data.get("motionUserAccelerationY", 0)
    z     = data.get("motionUserAccelerationZ", 0)
    hr    = data.get("heartRate", 0)
    roll  = float(data.get("motionRoll",  0))
    pitch = float(data.get("motionPitch", 0))

    # DEBUG: alle Werte auf einmal
    print(f"x={x:.3f} | y={y:.3f} | z={z:.3f} | roll={roll:.3f} | pitch={pitch:.3f}")

    intensity    = math.sqrt(x**2 + y**2 + z**2)
    current_time = time.time() - start_time

    with lock:
        intensity_values.append(intensity)
        heart_rate_values.append(hr)
        time_values.append(current_time)
        latest_data["accel_x"] = float(x)
        latest_data["accel_y"] = float(y)
        latest_data["roll"]    = roll
        latest_data["pitch"]   = pitch
        latest_data["heart_rate"] = float(hr)

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
        sphero.set_heading(0)
        sphero_heading = 0
        last_move_time = time.time()
        is_stopped     = True

        # Kalibrierung: Ruheposition der Watch merken
        print("⏳ Kalibrierung läuft (2 Sekunden still halten)...")
        time.sleep(2.0)
        with lock:
            roll_offset  = latest_data["roll"]   # Ruhe-Roll merken
            pitch_offset = latest_data["pitch"]  # Ruhe-Pitch merken
        print(f"✅ Kalibriert: roll_offset={roll_offset:.3f}, pitch_offset={pitch_offset:.3f}")
        while not stop_event.is_set():
            with lock:
                accel_x = latest_data.get("accel_x", 0)
                accel_y = latest_data["accel_y"]
                roll    = latest_data["roll"]  - roll_offset
                pitch   = latest_data["pitch"] - pitch_offset

            ROLL_DEADZONE  = 0.25
            ROLL_SCALE     = 60
            PITCH_FORWARD  = 0.15   # Pitch über diesen Wert = vorwärts
            PITCH_BACK     = -0.15  # Pitch unter diesen Wert = rückwärts/stopp

            # ── Richtung: NUR wenn Arm still (kein Pitch-Ausschlag) ──────────────
            if abs(pitch) < PITCH_FORWARD and abs(roll) > ROLL_DEADZONE:
                sphero_heading = (sphero_heading + roll * ROLL_SCALE * 0.05) % 360
                print(f"↩️  Heading: {sphero_heading:.0f}°  (roll={roll:.3f})")

            # ── Vorwärts: Pitch positiv (Hand nach vorne geneigt) ────────────────
            if pitch > PITCH_FORWARD:
                speed = int(min(pitch * 400, MAX_SPEED))
                speed = max(speed, MIN_SPEED)
                last_move_time = time.time()
                is_stopped     = False

                sphero.roll(int(sphero_heading), speed, 0.1)

                if speed > 180:
                    sphero.set_main_led(Color(r=0, g=0, b=255))
                elif speed > 130:
                    sphero.set_main_led(Color(r=0, g=255, b=0))
                else:
                    sphero.set_main_led(Color(r=255, g=255, b=0))

            elif time.time() - last_move_time > STOP_TIME and not is_stopped:
                sphero.stop_roll(int(sphero_heading))
                sphero.set_main_led(Color(r=255, g=0, b=0))
                is_stopped = True
                print("⏸️  Sphero gestoppt")

            time.sleep(0.05)

        # Nach der while-Schleife – wird jetzt sauber erreicht!
        print("🔌 Sphero wird getrennt...")
        sphero.stop_roll(0)
        sphero.set_main_led(Color(r=0, g=0, b=0))  # LED aus
        time.sleep(0.3)
        print("✅ Sphero getrennt.")


# ── Live Plot ─────────────────────────────────────────────────────────────────
def live_plot():
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Reha Live-Monitoring  |  [Q] zum Beenden", fontsize=14, fontweight='bold')

    def beenden(event=None):
        print("\nProgramm wird beendet.")
        stop_event.set()  # Signal an Hauptthread
        plt.close('all')
        #os._exit(0)

    fig.canvas.mpl_connect('close_event',     beenden)
    fig.canvas.mpl_connect('key_press_event', lambda e: beenden() if e.key in ('q', 'escape') else None)

    while True:
        with lock:
            if len(time_values) < 2:
                time.sleep(0.1)
                continue
            t   = list(time_values)
            raw = list(intensity_values)
            hr  = list(heart_rate_values)

        smoothed = moving_average(raw, SMOOTH_WIN)
        load     = compute_load_index(raw, hr)

        ax1.clear(); ax2.clear(); ax3.clear()

        # ── Plot 1: Bewegungsintensität ───────────────────────────────────────
        ax1.plot(t, raw,      color='lightsteelblue', alpha=0.5, linewidth=0.8, label='Roh')
        ax1.plot(t, smoothed, color='steelblue',      linewidth=1.8,            label=f'Geglättet (n={SMOOTH_WIN})')
        ax1.set_title("Bewegungsintensität")
        ax1.set_ylabel("Intensität")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.4)

        # ── Plot 2: Herzfrequenz ──────────────────────────────────────────────
        ax2.plot(t, hr, color='steelblue', linewidth=1.8)
        ax2.axhspan(0,         HR_WARN,   alpha=0.08, color='green')
        ax2.axhspan(HR_WARN,   HR_DANGER, alpha=0.08, color='orange')
        ax2.axhspan(HR_DANGER, 220,       alpha=0.08, color='red')
        ax2.axhline(HR_WARN,   color='orange', linestyle='--', linewidth=1.2, label=f'Warnung {HR_WARN} BPM')
        ax2.axhline(HR_DANGER, color='red',    linestyle='--', linewidth=1.2, label=f'Gefahr {HR_DANGER} BPM')

        if hr:
            current_hr = hr[-1]
            color = 'green' if current_hr < HR_WARN else ('orange' if current_hr < HR_DANGER else 'red')
            ax2.text(0.99, 0.95, f'{current_hr:.0f} BPM',
                     transform=ax2.transAxes, ha='right', va='top',
                     fontsize=13, fontweight='bold', color=color)

        ax2.set_title("Herzfrequenz")
        ax2.set_ylabel("BPM")
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.4)

        # ── Plot 3: Belastungsindex ───────────────────────────────────────────
        if load:
            current_load = load[-1]
            bar_color = 'green' if current_load < 40 else ('orange' if current_load < 70 else 'red')
            ax3.plot(t, load, color='steelblue', linewidth=1.8)
            ax3.fill_between(t, load, alpha=0.2, color='steelblue')
            ax3.axhline(40, color='orange', linestyle=':', linewidth=1.0, label='Moderat (40)')
            ax3.axhline(70, color='red',    linestyle=':', linewidth=1.0, label='Hoch (70)')
            ax3.set_ylim(0, 105)
            ax3.text(0.99, 0.95, f'Index: {current_load:.0f}',
                     transform=ax3.transAxes, ha='right', va='top',
                     fontsize=13, fontweight='bold', color=bar_color)

        ax3.set_title("Belastungsindex (kombiniert)")
        ax3.set_xlabel("Zeit (s)")
        ax3.set_ylabel("Index (0–100)")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.pause(0.1)
        time.sleep(0.05)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Flask Server → Nebenthread (kein asyncio)
    threading.Thread(target=run_server, daemon=True).start()

    # Live Plot → Nebenthread
    threading.Thread(target=live_plot, daemon=True).start()

    # Sphero → Hauptthread (asyncio muss hier laufen!)
    control_sphero()