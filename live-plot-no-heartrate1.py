from flask import Flask, request
import threading
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

app = Flask(__name__)

# ── Thread-sicherer Datenspeicher ─────────────────────────────────────────────
lock = threading.Lock()
MAX_POINTS = 300

intensity_values  = deque(maxlen=MAX_POINTS)
heart_rate_values = deque(maxlen=MAX_POINTS)
time_values       = deque(maxlen=MAX_POINTS)

start_time = time.time()

# ── Konfiguration (leicht anpassbar) ──────────────────────────────────────────
HR_WARN    = 100   # BPM: gelbe Warnschwelle
HR_DANGER  = 120   # BPM: rote Gefahrenschwelle
SMOOTH_WIN = 10    # Fenster für gleitenden Durchschnitt


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def moving_average(data: list, window: int) -> np.ndarray:
    """Gleitender Durchschnitt – gibt Array gleicher Länge zurück."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    # 'same' behält die ursprüngliche Länge
    return np.convolve(data, kernel, mode='same')


def compute_load_index(intensities: list, heart_rates: list) -> list:
    """
    Einfacher Belastungsindex (0–100):
      - 60% Gewichtung: normierte Bewegungsintensität
      - 40% Gewichtung: normierte Herzfrequenz (Ruhepuls ~60, Max ~180)
    """
    if not intensities:
        return []
    
    max_i = max(intensities) if max(intensities) > 0 else 1
    load  = []
    
    for i, hr in zip(intensities, heart_rates):
        norm_intensity = min(i / max_i, 1.0)          # 0–1
        norm_hr        = min((hr - 60) / 120, 1.0)    # 0–1  (60=Ruhe, 180=Max)
        norm_hr        = max(norm_hr, 0)
        index          = (0.6 * norm_intensity + 0.4 * norm_hr) * 100
        load.append(index)
    
    return load


# ── Flask Route ───────────────────────────────────────────────────────────────
@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json
    if data is None:
        return "No data", 400

    x  = data.get("motionUserAccelerationX", 0)
    y  = data.get("motionUserAccelerationY", 0)
    z  = data.get("motionUserAccelerationZ", 0)
    hr = data.get("heartRate", 0)

    intensity    = math.sqrt(x**2 + y**2 + z**2)
    current_time = time.time() - start_time

    with lock:
        intensity_values.append(intensity)
        heart_rate_values.append(hr)
        time_values.append(current_time)

    return "OK", 200


# ── Server-Thread ─────────────────────────────────────────────────────────────
def run_server():
    app.run(host='0.0.0.0', port=56671, debug=False)


# ── Live-Plot ─────────────────────────────────────────────────────────────────
def live_plot():
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Reha Live-Monitoring  |  [Q] zum Beenden", fontsize=14, fontweight='bold')

    def beenden(event=None):
        print("Programm wird beendet.")
        plt.close('all')
        import os
        os._exit(0)

    fig.canvas.mpl_connect('close_event',    beenden)
    fig.canvas.mpl_connect('key_press_event', lambda e: beenden() if e.key in ('q', 'escape') else None)

    while True:
        with lock:
            if len(time_values) < 2:
                time.sleep(0.1)
                continue
            t   = list(time_values)
            raw = list(intensity_values)
            hr  = list(heart_rate_values)

        # ── Berechnungen ──────────────────────────────────────────────────────
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

        # ── Plot 2: Herzfrequenz mit Zonen ────────────────────────────────────
        ax2.plot(t, hr, color='steelblue', linewidth=1.8)

        # Farbige Hintergrund-Zonen
        ax2.axhspan(0,        HR_WARN,   alpha=0.08, color='green')
        ax2.axhspan(HR_WARN,  HR_DANGER, alpha=0.08, color='orange')
        ax2.axhspan(HR_DANGER, 220,      alpha=0.08, color='red')

        # Schwellenlinien
        ax2.axhline(HR_WARN,   color='orange', linestyle='--', linewidth=1.2, label=f'Warnung {HR_WARN} BPM')
        ax2.axhline(HR_DANGER, color='red',    linestyle='--', linewidth=1.2, label=f'Gefahr {HR_DANGER} BPM')

        # Aktueller HF-Wert als Text
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
            # Farbe je nach aktuellem Belastungslevel
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
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    try:
        live_plot()
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
        plt.close('all')