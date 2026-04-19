from flask import Flask, request
import threading
import time
import math
import matplotlib.pyplot as plt
from collections import deque

app = Flask(__name__)

# Speicher für Daten
intensity_values = deque(maxlen=200)
heart_rate_values = deque(maxlen=200)
time_values = deque(maxlen=200)
load_values = deque(maxlen=200)

start_time = time.time()

# Flask Route
@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json

    if data is not None:
        x = data.get("motionUserAccelerationX", 0)
        y = data.get("motionUserAccelerationY", 0)
        z = data.get("motionUserAccelerationZ", 0)
        heart_rate = data.get("heartRate", 0)

        # Bewegungsintensität berechnen
        intensity = math.sqrt(x**2 + y**2 + z**2)
        current_time = time.time() - start_time

        intensity_values.append(intensity)
        heart_rate_values.append(heart_rate)
        time_values.append(current_time)

        # Normierung (einfach)
        norm_intensity = min(intensity / 2.0, 1.0)   # 2.0 ≈ max Bewegung
        norm_hr = min(heart_rate / 180.0, 1.0)       # 180 BPM als Referenz

        load = norm_intensity + norm_hr
        load_values.append(load)

        #print(f"Intensity: {intensity:.3f}, Heart Rate: {heart_rate}")

    return "OK", 200


def run_server():
    app.run(host='0.0.0.0', port=56671, debug=False)


def live_plot():
    plt.ion()
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    while True:
        if len(time_values) > 0:
            ax1.clear()
            ax2.clear()

            # Oberer Plot: Bewegungsintensität
            ax1.plot(time_values, intensity_values)
            ax1.set_title("Live Bewegungsintensität")
            ax1.set_ylabel("Intensität")
            ax1.grid(True)

            # Unterer Plot: Herzfrequenz
            ax2.plot(time_values, heart_rate_values)
            ax2.set_title("Live Herzfrequenz")
            ax2.set_xlabel("Zeit (s)")
            ax2.set_ylabel("BPM")
            ax2.grid(True)

            # Dritter Plot: Belastung
            ax3.plot(time_values, load_values)
            ax3.set_title("Belastungsindex")
            ax3.set_xlabel("Zeit (s)")
            ax3.set_ylabel("Level")
            ax3.grid(True)

            plt.tight_layout()
            plt.pause(0.05)

        time.sleep(0.05)


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    live_plot()