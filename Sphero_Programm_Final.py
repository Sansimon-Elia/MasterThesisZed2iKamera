import logging
import threading
import time
import math
import signal
import sys
from flask import Flask, request
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from messuhr_module_copy import monitor_and_plot

app = Flask(__name__)

# Deaktiviere HTTP-Request-Logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Globale Variablen für Sensordaten
user_accel_x = None
user_accel_y = None
initial_heading = None
current_heading = None
sphero_heading = 0
last_movement_time = time.time()

# **Adaptive Thresholds**
HEADING_CHANGE_THRESHOLD = 5  # Empfindlich für Richtungsänderungen
BASE_MOVEMENT_THRESHOLD = 0.04  # Grundbewegungsschwelle
STOP_THRESHOLD = 0.02  # Stopp, wenn Bewegung zu gering ist
MAX_SPEED = 255  # Höchstgeschwindigkeit
MIN_SPEED = 160  # Mindestgeschwindigkeit für kleine Bewegungen
STOP_TIME_THRESHOLD = 1.5  # Zeit, nach der der Roboter anhält
TURN_SPEED_REDUCTION = 100  # Reduzierte Geschwindigkeit bei Richtungswechsel

# **Verstärkungsfaktor für die Y-Bewegung**
Y_WEIGHT = 1.5  # Y-Beschleunigung wird stärker gewichtet als X
X_WEIGHT = 1.0  # X-Beschleunigung bleibt gleich

def get_dynamic_threshold(acceleration):
    return max(BASE_MOVEMENT_THRESHOLD, abs(acceleration) * 0.2)

def get_heading_difference(initial, current):
    diff = (current - initial + 180) % 360 - 180
    return diff

@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    global user_accel_x, user_accel_y, initial_heading, current_heading
    data = request.json

    if 'motionUserAccelerationX' in data:
        user_accel_x = float(data['motionUserAccelerationX'])

    if 'motionUserAccelerationY' in data:
        user_accel_y = float(data['motionUserAccelerationY'])

    if 'locationMagneticHeading' in data:
        if initial_heading is None:
            initial_heading = float(data['locationMagneticHeading'])
        current_heading = float(data['locationMagneticHeading'])

    return "Daten empfangen", 200

def run_server():
    app.run(host='0.0.0.0', port=56671)

def control_sphero():
    toy = scanner.find_toy()
    if toy:
        with SpheroEduAPI(toy) as sphero:
            global initial_heading, current_heading, user_accel_x, user_accel_y, sphero_heading, last_movement_time

            while initial_heading is None or current_heading is None or user_accel_x is None or user_accel_y is None:
                time.sleep(0.1)

            sphero.set_heading(0)
            sphero_heading = 0  
            print(f"Sphero auf 0° kalibriert.")

            while True:
                if None not in (user_accel_x, user_accel_y, current_heading):
                    # **1. Berechne die Bewegung unter Berücksichtigung der Y-Gewichtung**
                    weighted_accel_x = user_accel_x * X_WEIGHT
                    weighted_accel_y = user_accel_y * Y_WEIGHT
                    acceleration_magnitude = math.sqrt(weighted_accel_x**2 + weighted_accel_y**2)
                    dynamic_threshold = get_dynamic_threshold(acceleration_magnitude)

                    # **2. Prüfe Richtungsänderung**
                    heading_difference = get_heading_difference(initial_heading, current_heading)

                    if heading_difference > HEADING_CHANGE_THRESHOLD:
                        #print("Rechtsbewegung erkannt!")
                        new_direction = (sphero_heading + heading_difference) % 360
                        speed = max(MIN_SPEED - TURN_SPEED_REDUCTION, 100)  
                    elif heading_difference < -HEADING_CHANGE_THRESHOLD:
                        #print("Linksbewegung erkannt!")
                        new_direction = (sphero_heading + heading_difference) % 360
                        speed = max(MIN_SPEED - TURN_SPEED_REDUCTION, 100)  
                    else:
                        new_direction = sphero_heading
                        speed = min(max(int(acceleration_magnitude * 100), MIN_SPEED), MAX_SPEED)

                   # 3. Bewegung ausführen (nur bei positiver Y-Beschleunigung!)
                    if user_accel_y > 0 and acceleration_magnitude > dynamic_threshold:
                        last_movement_time = time.time()
                        sphero.roll(int(new_direction), speed, 0.1)

                        if speed > 180:
                            sphero.set_main_led(Color(r=0, g=0, b=255))  # Blau für hohe Geschwindigkeit
                        elif speed > 160:
                            sphero.set_main_led(Color(r=0, g=255, b=0))  # Grün für mittlere Geschwindigkeit
                        else:
                            sphero.set_main_led(Color(r=255, g=255, b=0))  # Gelb für niedrige Geschwindigkeit

                    elif time.time() - last_movement_time > STOP_TIME_THRESHOLD:
                        print("⏸️ Bewegung gestoppt!")
                        sphero.stop_roll(int(current_heading))
                        sphero.set_main_led(Color(r=255, g=0, b=0))

                        # Neues 0° setzen
                        #sphero_heading = 0
                        #initial_heading = current_heading
                        #print(f"📌 Neues initiales Heading gesetzt: {initial_heading}°")
                        last_movement_time = time.time()

                time.sleep(0.1)

def signal_handler(sig, frame):
    print("Programm wird beendet...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Thread für SensorLog Webserver
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Thread für Messuhr / Empatica-Monitoring
    messuhr_thread = threading.Thread(target=monitor_and_plot)
    messuhr_thread.daemon = True
    messuhr_thread.start()

    # Hauptfunktion Sphero läuft direkt
    control_sphero()
