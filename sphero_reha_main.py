"""
Sphero Reha-Controller – Unified Application
Kombiniert: Sphero-Steuerung, ZED2i Body-Tracking, Live-Graphen
"""

# ── Basis-Imports ─────────────────────────────────────────────────────────────
from flask import Flask, request
import asyncio
import threading
import time
import math
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import numpy as np

# Matplotlib – TkAgg Backend VOR plt-Import setzen
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sphero
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ── Windows Event Loop Policy ─────────────────────────────────────────────────
if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Flask App ─────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

# ── Globale Thread-sichere Zustände ──────────────────────────────────────────
data_lock      = threading.Lock()
stop_sphero    = threading.Event()
stop_camera    = threading.Event()
server_started = threading.Event()

control_thread  = None
server_thread   = None
camera_thread   = None
sphero_api      = None
last_status     = "Bereit. Starte die Sphero-Steuerung per Button."

# Sphero-Sensordaten
latest_data = {
    "gx": 0.0, "gy": 0.0, "gz": 0.0,
    "state": "neutral", "heading": 0,
    "backward_until": 0.0,
}

# Live-Graph-Daten
MAX_POINTS        = 300
graph_start_time  = time.time()
intensity_values  = deque(maxlen=MAX_POINTS)
heart_rate_values = deque(maxlen=MAX_POINTS)
graph_time_values = deque(maxlen=MAX_POINTS)

# ── Sphero-Konfiguration ──────────────────────────────────────────────────────
MIN_SPEED_DYN        = 30
MAX_SPEED_DYN        = 120
TURN_SPEED_FACTOR    = 0.6
STOP_TIME            = 1.5
GY_RIGHT_THRESHOLD   = -0.80
GY_LEFT_THRESHOLD    = +0.80
GX_FORWARD_MAX       = +0.1
GX_NEUTRAL_THRESHOLD = +0.12
MAX_TURN_ANGLE       = 90
TURN_DEADZONE        = 0.80
BACKWARD_DURATION = 2.0   # Sekunden Rückwärtsfahrt pro Double Tap
BACKWARD_SPEED    = 80    # Geschwindigkeit während der Rückwärtsfahrt

# ── Live-Graph-Konfiguration ─────────────────────────────────────────────────
HR_WARN    = 100
HR_DANGER  = 120
SMOOTH_WIN = 10


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def set_status(msg: str):
    global last_status
    with data_lock:
        last_status = msg


def get_state(gx, gy, gz) -> str:
    if gy < GY_RIGHT_THRESHOLD:   return "right"
    if gy > GY_LEFT_THRESHOLD:    return "left"
    if gx > GX_NEUTRAL_THRESHOLD: return "neutral"
    if gx < GX_FORWARD_MAX:       return "forward"
    return "neutral"


def calc_speed(gx) -> int:
    intensity = (0.10 - gx) / (0.10 - (-0.95))
    intensity = max(0.0, min(1.0, intensity))
    return int(MIN_SPEED_DYN + intensity * (MAX_SPEED_DYN - MIN_SPEED_DYN))


def calc_turn(gy_value) -> float:
    intensity = (abs(gy_value) - TURN_DEADZONE) / (1.0 - TURN_DEADZONE)
    return max(0.0, min(1.0, intensity)) * MAX_TURN_ANGLE


def moving_average(data: list, window: int) -> np.ndarray:
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='same')


def compute_load_index(intensities: list, heart_rates: list) -> list:
    if not intensities:
        return []
    max_i = max(intensities) if max(intensities) > 0 else 1
    result = []
    for i, hr in zip(intensities, heart_rates):
        norm_i  = min(i / max_i, 1.0)
        norm_hr = max(min((hr - 60) / 120, 1.0), 0)
        result.append((0.6 * norm_i + 0.4 * norm_hr) * 100)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Flask Route (empfängt alle Sensordaten)
# ─────────────────────────────────────────────────────────────────────────────

@flask_app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json
    if data is None:
        return "No data", 400
    
    # ── NEU: Double Tap ist ein reiner Event-POST ohne Sensordaten ──
    if data.get("doubleTap"):
        with data_lock:
            latest_data["backward_until"] = time.time() + BACKWARD_DURATION
        print("[EVENT] Double Tap → Rückwärts")
        return "OK", 200          # ← wichtig: hier beenden!

    # Sphero-Steuerung (Schwerkraft)
    gx    = float(data.get("gravityX", 0))
    gy    = float(data.get("gravityY", 0))
    gz    = float(data.get("gravityZ", 0))
    state = get_state(gx, gy, gz)

    # Live-Graph (Beschleunigung + Herzfrequenz)
    ax = float(data.get("motionUserAccelerationX", 0))
    ay = float(data.get("motionUserAccelerationY", 0))
    az = float(data.get("motionUserAccelerationZ", 0))
    hr = float(data.get("heartRate", 0))
    intensity    = math.sqrt(ax**2 + ay**2 + az**2)
    current_time = time.time() - graph_start_time

    with data_lock:
        latest_data["gx"]    = gx
        latest_data["gy"]    = gy
        latest_data["gz"]    = gz
        latest_data["state"] = state
        intensity_values.append(intensity)
        heart_rate_values.append(hr)
        graph_time_values.append(current_time)

    return "OK", 200


def run_server():
    flask_app.run(host='0.0.0.0', port=56671, debug=False, use_reloader=False)


def start_server_once():
    global server_thread
    if server_started.is_set():
        return
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_started.set()


# ─────────────────────────────────────────────────────────────────────────────
# Sphero Steuerung
# ─────────────────────────────────────────────────────────────────────────────

def _is_connection_error(exc: Exception) -> bool:
    """Erkennt BleakError 'Not connected' und concurrent.futures.TimeoutError."""
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    return "not connected" in msg or "bleakerror" in msg or "timeout" in msg


def control_sphero():
    global sphero_api

    MAX_RECONNECTS   = 5
    RECONNECT_DELAY  = 3.0   # Sekunden zwischen Reconnect-Versuchen
    reconnect_count  = 0
    sphero_heading   = 0     # Heading über Reconnects hinweg behalten

    while not stop_sphero.is_set() and reconnect_count <= MAX_RECONNECTS:

        # ── Verbinden ─────────────────────────────────────────────────────────
        if reconnect_count == 0:
            set_status("Suche Sphero BOLT...")
        else:
            set_status(f"Reconnect {reconnect_count}/{MAX_RECONNECTS} – suche Sphero...")

        try:
            toy = scanner.find_toy()
            if not toy:
                set_status("Kein Sphero gefunden. Bluetooth und Sphero prüfen.")
                return
        except Exception as e:
            set_status(f"Sphero-Start fehlgeschlagen: {e}")
            return

        # ── Steuerungs-Loop ───────────────────────────────────────────────────
        connection_lost = False
        try:
            with SpheroEduAPI(toy) as sphero:
                sphero_api     = sphero
                last_move_time = time.time()
                is_stopped     = True

                sphero.set_heading(sphero_heading)
                sphero.set_main_led(Color(r=255, g=255, b=255))

                if reconnect_count == 0:
                    set_status("Sphero verbunden. Sensor-App kann Daten senden.")
                else:
                    set_status(f"Sphero wieder verbunden (Versuch {reconnect_count}).")
                reconnect_count = 0  # bei Erfolg zurücksetzen

                while not stop_sphero.is_set():
                    with data_lock:
                        gx             = latest_data["gx"]
                        gy             = latest_data["gy"]
                        gz             = latest_data["gz"]
                        backward_until = latest_data["backward_until"]

                    # ── Double-Tap-Rückwärtsfahrt hat Vorrang vor Gravity-Steuerung ──
                    if time.time() < backward_until:
                        try:
                            backward_heading = (sphero_heading + 180) % 360
                            sphero.roll(int(backward_heading), BACKWARD_SPEED, 0.1)
                            sphero.set_main_led(Color(r=160, g=0, b=255))
                            last_move_time = time.time()
                            is_stopped     = False
                        except Exception as e:
                            if _is_connection_error(e):
                                connection_lost = True
                                break
                            print(f"[WARN] Rückwärts-Befehl fehlgeschlagen: {e}")
                        time.sleep(0.05)
                        continue

                    state = get_state(gx, gy, gz)

                    try:
                        if state == "right":
                            turn           = calc_turn(gy)
                            speed          = int(calc_speed(gx) * TURN_SPEED_FACTOR)
                            sphero_heading = (sphero_heading + turn) % 360
                            sphero.roll(int(sphero_heading), speed, 0.1)
                            sphero.set_main_led(Color(r=255, g=100, b=0))
                            last_move_time = time.time()
                            is_stopped     = False
                            with data_lock:
                                latest_data["heading"] = sphero_heading

                        elif state == "left":
                            turn           = calc_turn(gy)
                            speed          = int(calc_speed(gx) * TURN_SPEED_FACTOR)
                            sphero_heading = (sphero_heading - turn) % 360
                            sphero.roll(int(sphero_heading), speed, 0.1)
                            sphero.set_main_led(Color(r=0, g=200, b=255))
                            last_move_time = time.time()
                            is_stopped     = False
                            with data_lock:
                                latest_data["heading"] = sphero_heading

                        elif state == "forward":
                            speed = calc_speed(gx)
                            sphero.roll(int(sphero_heading), speed, 0.1)
                            sphero.set_main_led(Color(r=0, g=255, b=0))
                            last_move_time = time.time()
                            is_stopped     = False

                        elif state == "neutral":
                            if not is_stopped and time.time() - last_move_time > STOP_TIME:
                                sphero.stop_roll(int(sphero_heading))
                                sphero.set_main_led(Color(r=255, g=0, b=0))
                                is_stopped = True

                    except Exception as e:
                        if _is_connection_error(e):
                            connection_lost = True
                            break
                        # Andere Fehler: loggen, aber weiterlaufen
                        print(f"[WARN] Sphero-Befehl fehlgeschlagen: {e}")

                    time.sleep(0.05)

                # Sauber beenden wenn gewollt gestoppt
                if not connection_lost:
                    try:
                        sphero.stop_roll(0)
                        sphero.set_main_led(Color(r=0, g=0, b=0))
                        time.sleep(0.5)
                    except Exception:
                        pass

        except Exception as e:
            if _is_connection_error(e):
                connection_lost = True
            else:
                set_status(f"Sphero-Fehler: {e}")
                return

        # ── Verbindungsverlust behandeln ──────────────────────────────────────
        if connection_lost and not stop_sphero.is_set():
            reconnect_count += 1
            if reconnect_count <= MAX_RECONNECTS:
                set_status(
                    f"Verbindung verloren! Reconnect in {int(RECONNECT_DELAY)}s "
                    f"({reconnect_count}/{MAX_RECONNECTS})..."
                )
                time.sleep(RECONNECT_DELAY)
            else:
                set_status(
                    f"Verbindung nach {MAX_RECONNECTS} Versuchen verloren. "
                    "Sphero bitte neu starten und erneut verbinden."
                )
                return

    set_status("Sphero getrennt.")


def start_sphero_control() -> bool:
    global control_thread
    if control_thread and control_thread.is_alive():
        return False
    stop_sphero.clear()
    start_server_once()
    control_thread = threading.Thread(target=control_sphero, daemon=True)
    control_thread.start()
    return True


def stop_sphero_control():
    stop_sphero.set()
    set_status("Sphero-Steuerung wird gestoppt...")


# ─────────────────────────────────────────────────────────────────────────────
# ZED2i Kamera – Body-Tracking mit Tiefendaten
# (läuft in eigenem Thread, zeigt eigenes OpenCV-Fenster)
# ─────────────────────────────────────────────────────────────────────────────

# Kamera-Hilfsfunktionen (benötigen kein pyzed beim Import)

_WICHTIGE_PUNKTE = {11, 2, 12, 5, 13, 6, 15, 8, 27}
_HAND_KP     = {8, 15}
_ARM_KP      = {13, 6}
_SHOULDER_KP = {5, 12}

_COLOR_HAND    = (0, 255, 0)
_COLOR_ARM     = (255, 165, 0)
_COLOR_BODY    = (0, 180, 255)
_COLOR_HEAD    = (255, 255, 0)
_COLOR_LINE    = (200, 200, 200)
_COLOR_GOOD    = (0, 255, 0)
_COLOR_WARNING = (0, 165, 255)
_COLOR_BAD     = (0, 0, 255)

_BODY_CONNECTIONS = [
    (27, 11), (11, 2), (11, 12), (11, 5),
    (12, 13), (5, 6), (6, 8), (13, 15),
]

_pygame_ready = False

# ── Adaptives Audio-Feedback ──────────────────────────────────────────────────
# Wissenschaftlicher Hintergrund:
#   Schmidt, R.A. & Lee, T.D. (2011). Motor Control and Learning (5th ed.).
#   Human Kinetics. → "Guidance Hypothesis": zu häufiges Feedback erzeugt
#   Abhängigkeit und reduziert den Lerneffekt.
#   Winstein & Schmidt (1990). J. Exp. Psychol.: Learn. Mem. Cogn., 16(4),
#   677-691. → Seltenereres Feedback verbessert motorisches Lernen.
#
# Implementierung: Exponentieller Backoff + Phrasen-Variation
#   - Intervall verdoppelt sich nach jeder Meldung (5s → 10s → 20s → 60s max)
#   - Mehrere Formulierungen rotieren, um Gewöhnung zu vermeiden
#   - Intervall wird zurückgesetzt wenn der Zustand sich ändert

_FEEDBACK_MIN_INTERVAL = 5.0    # Sekunden – erstes Feedback
_FEEDBACK_MAX_INTERVAL = 60.0   # Sekunden – maximales Intervall
_FEEDBACK_BACKOFF      = 2.0    # Multiplikator nach jeder Wiedergabe

_FEEDBACK_PHRASES = {
    "zu_nah": [
        "Bitte treten Sie zurueck",
        "Sie stehen zu nah. Bitte etwas Abstand halten",
        "Bitte einen Schritt zuruecktreten",
    ],
    "zu_weit": [
        "Bitte treten Sie naeher",
        "Sie stehen zu weit entfernt. Bitte naeher kommen",
        "Bitte einen Schritt nach vorne treten",
    ],
}

_feedback_state = {
    "zu_nah":  {"last_time": 0.0, "interval": _FEEDBACK_MIN_INTERVAL, "phrase_idx": 0},
    "zu_weit": {"last_time": 0.0, "interval": _FEEDBACK_MIN_INTERVAL, "phrase_idx": 0},
}
_last_distance_condition = "ok"


def _reset_feedback(kategorie: str):
    """Intervall zurücksetzen wenn Zustand neu eintritt."""
    _feedback_state[kategorie]["interval"]   = _FEEDBACK_MIN_INTERVAL
    _feedback_state[kategorie]["phrase_idx"] = 0


def _init_audio():
    global _pygame_ready
    try:
        import pygame
        pygame.mixer.init()
        _pygame_ready = True
    except Exception:
        _pygame_ready = False


def _sprich(kategorie: str):
    """
    Spielt Feedback mit exponentiellem Backoff und Phrasen-Rotation ab.
    Intervall: 5s → 10s → 20s → 40s → 60s (Deckel).
    """
    state = _feedback_state[kategorie]
    jetzt = time.time()
    if jetzt - state["last_time"] < state["interval"]:
        return

    phrases  = _FEEDBACK_PHRASES[kategorie]
    text     = phrases[state["phrase_idx"] % len(phrases)]
    state["phrase_idx"] += 1
    state["last_time"]   = jetzt
    state["interval"]    = min(state["interval"] * _FEEDBACK_BACKOFF, _FEEDBACK_MAX_INTERVAL)

    def sprechen():
        try:
            from gtts import gTTS
            import pygame
            import tempfile, os
            tts = gTTS(text=text, lang='de')
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tmp.close()
            tts.save(tmp.name)
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            time.sleep(0.1)
            os.unlink(tmp.name)
        except Exception as ex:
            print(f"[Audio] Fehler: {ex}")

    threading.Thread(target=sprechen, daemon=True).start()


def _berechne_winkel_3d(p1, p2, p3):
    a = np.array([p1[0], p1[1], p1[2]])
    b = np.array([p2[0], p2[1], p2[2]])
    c = np.array([p3[0], p3[1], p3[2]])
    if np.allclose(a, 0) or np.allclose(b, 0) or np.allclose(c, 0):
        return None
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)


def _winkel_farbe(w):
    if w >= 150: return _COLOR_GOOD
    if w >= 90:  return _COLOR_WARNING
    return _COLOR_BAD


def _winkel_text(w):
    if w >= 150: return "Gut gestreckt!"
    if w >= 90:  return "Weiter strecken..."
    return "Mehr strecken!"


def _get_kp_color(idx):
    if idx in _HAND_KP:     return _COLOR_HAND
    if idx in _ARM_KP:      return _COLOR_ARM
    if idx in _SHOULDER_KP: return _COLOR_ARM
    if idx == 27:           return _COLOR_HEAD
    return _COLOR_BODY


def _draw_skeleton(frame, kps_2d, cv2):
    h, w = frame.shape[:2]
    for (i, j) in _BODY_CONNECTIONS:
        if i >= len(kps_2d) or j >= len(kps_2d): continue
        x1, y1 = int(kps_2d[i][0]), int(kps_2d[i][1])
        x2, y2 = int(kps_2d[j][0]), int(kps_2d[j][1])
        if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
            cv2.line(frame, (x1, y1), (x2, y2), _COLOR_LINE, 2)
    for idx in _WICHTIGE_PUNKTE:
        if idx >= len(kps_2d): continue
        x, y = int(kps_2d[idx][0]), int(kps_2d[idx][1])
        if 0 < x < w and 0 < y < h:
            radius = 8 if idx in _HAND_KP else 5
            cv2.circle(frame, (x, y), radius, _get_kp_color(idx), -1)


def _draw_winkel(frame, kps_2d, kps_3d, schulter_idx, ellbogen_idx, handgelenk_idx, seite, cv2):
    h, w = frame.shape[:2]
    e2d  = kps_2d[ellbogen_idx]
    ex, ey = int(e2d[0]), int(e2d[1])
    if not (0 < ex < w and 0 < ey < h):
        return
    winkel = _berechne_winkel_3d(
        kps_3d[schulter_idx], kps_3d[ellbogen_idx], kps_3d[handgelenk_idx]
    )
    if winkel is None:
        cv2.putText(frame, f"{seite}: Nicht sichtbar",
                    (20, 80 if seite == "Links" else 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return
    farbe = _winkel_farbe(winkel)
    cv2.circle(frame, (ex, ey), 14, farbe, -1)
    cv2.circle(frame, (ex, ey), 14, (255, 255, 255), 2)
    cv2.putText(frame, f"{winkel}", (ex - 20, ey - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"{seite}: {winkel} Grad _ {_winkel_text(winkel)}",
                (20, 80 if seite == "Links" else 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)


def _draw_abstand(frame, kps_3d, cv2):
    global _last_distance_condition
    p = kps_3d[2]
    if p[2] <= 0:
        return
    abstand = p[2]
    if abstand < 1.0:
        neue_bedingung = "zu_nah"
        farbe          = _COLOR_BAD
        hinweis        = "Zu nah _ Bitte zuruecktreten"
        if _last_distance_condition != "zu_nah":
            _reset_feedback("zu_nah")
        _sprich("zu_nah")
    elif abstand > 3.5:
        neue_bedingung = "zu_weit"
        farbe          = _COLOR_WARNING
        hinweis        = "Zu weit _ Bitte naeher treten"
        if _last_distance_condition != "zu_weit":
            _reset_feedback("zu_weit")
        _sprich("zu_weit")
    else:
        neue_bedingung = "ok"
        farbe          = _COLOR_GOOD
        hinweis        = "Abstand OK"
    _last_distance_condition = neue_bedingung
    cv2.putText(frame, f"Abstand: {abstand:.2f}m _ {hinweis}",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)


def run_camera():
    """ZED2i Body-Tracking in eigenem Thread. Zeigt OpenCV-Fenster."""
    try:
        import pyzed.sl as sl
        import cv2
    except ImportError as e:
        set_status(f"Kamera-Import fehlgeschlagen: {e}")
        return

    _init_audio()

    zed         = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps        = 30
    init_params.coordinate_units  = sl.UNIT.METER
    init_params.depth_mode        = sl.DEPTH_MODE.NEURAL

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        set_status("ZED2i: Kamera konnte nicht geöffnet werden.")
        return

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_area_memory = True
    zed.enable_positional_tracking(tracking_params)

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking     = True
    body_params.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format         = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting = True

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        set_status("ZED2i: Body Tracking konnte nicht aktiviert werden.")
        zed.close()
        return

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    image   = sl.Mat()
    bodies  = sl.Bodies()
    runtime = sl.RuntimeParameters()

    set_status("Kamera läuft. [q] im Kamerafenster zum Stoppen.")

    while not stop_camera.is_set():
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        zed.retrieve_bodies(bodies, body_runtime)

        if bodies.is_new:
            person_count = 0
            for body in bodies.body_list:
                if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    person_count += 1
                    kps_2d = body.keypoint_2d
                    kps_3d = body.keypoint

                    _draw_skeleton(frame, kps_2d, cv2)
                    _draw_abstand(frame, kps_3d, cv2)

                    _draw_winkel(frame, kps_2d, kps_3d,
                                 schulter_idx=12, ellbogen_idx=13,
                                 handgelenk_idx=15, seite="Links", cv2=cv2)
                    _draw_winkel(frame, kps_2d, kps_3d,
                                 schulter_idx=5, ellbogen_idx=6,
                                 handgelenk_idx=8, seite="Rechts", cv2=cv2)

                    head = kps_2d[27]
                    hx, hy = int(head[0]), int(head[1])
                    if 0 < hx < frame.shape[1]:
                        cv2.putText(frame, f"Person {body.id}",
                                    (hx - 40, hy - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Personen: {person_count}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "q = Stopp", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow("ZED 2i – 3D Reha Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera.set()
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    set_status("Kamera gestoppt.")


# ─────────────────────────────────────────────────────────────────────────────
# Live-Graph Fenster (eingebettet in Tkinter)
# ─────────────────────────────────────────────────────────────────────────────

class LiveGraphWindow:
    """Öffnet ein Tkinter-Toplevel-Fenster mit eingebetteten Live-Graphen."""

    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("Live Graphen – Reha Monitoring")
        self.win.geometry("960x720")
        self.win.minsize(700, 500)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(9, 7), sharex=True
        )
        self.fig.suptitle("Reha Live-Monitoring", fontsize=13, fontweight='bold')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Status-Zeile unten
        self.info_var = tk.StringVar(value="Warte auf Sensordaten...")
        ttk.Label(self.win, textvariable=self.info_var,
                  font=("Consolas", 9), foreground="#555").pack(pady=(0, 4))

        self._running = True
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._update()

    def _update(self):
        if not self._running:
            return

        with data_lock:
            n = len(graph_time_values)
            if n < 2:
                self.info_var.set("Warte auf Sensordaten vom Handy...")
                self.win.after(500, self._update)
                return
            t   = list(graph_time_values)
            raw = list(intensity_values)
            hr  = list(heart_rate_values)

        smoothed = moving_average(raw, SMOOTH_WIN)
        load     = compute_load_index(raw, hr)

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # ── Plot 1: Bewegungsintensität ───────────────────────────────────────
        self.ax1.plot(t, raw, color='lightsteelblue', alpha=0.5,
                      linewidth=0.8, label='Roh')
        self.ax1.plot(t, smoothed, color='steelblue', linewidth=1.8,
                      label=f'Geglättet (n={SMOOTH_WIN})')
        self.ax1.set_title("Bewegungsintensität")
        self.ax1.set_ylabel("Intensität")
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.4)

        # ── Plot 2: Herzfrequenz ──────────────────────────────────────────────
        self.ax2.plot(t, hr, color='steelblue', linewidth=1.8)
        self.ax2.axhspan(0,        HR_WARN,   alpha=0.08, color='green')
        self.ax2.axhspan(HR_WARN,  HR_DANGER, alpha=0.08, color='orange')
        self.ax2.axhspan(HR_DANGER, 220,      alpha=0.08, color='red')
        self.ax2.axhline(HR_WARN,   color='orange', linestyle='--',
                         linewidth=1.2, label=f'Warnung {HR_WARN} BPM')
        self.ax2.axhline(HR_DANGER, color='red',    linestyle='--',
                         linewidth=1.2, label=f'Gefahr {HR_DANGER} BPM')
        if hr:
            cur_hr = hr[-1]
            col = 'green' if cur_hr < HR_WARN else ('orange' if cur_hr < HR_DANGER else 'red')
            self.ax2.text(0.99, 0.95, f'{cur_hr:.0f} BPM',
                          transform=self.ax2.transAxes, ha='right', va='top',
                          fontsize=13, fontweight='bold', color=col)
        self.ax2.set_title("Herzfrequenz")
        self.ax2.set_ylabel("BPM")
        self.ax2.legend(loc='upper left', fontsize=8)
        self.ax2.grid(True, alpha=0.4)

        # ── Plot 3: Belastungsindex ───────────────────────────────────────────
        if load:
            cur_load  = load[-1]
            load_col  = 'green' if cur_load < 40 else ('orange' if cur_load < 70 else 'red')
            self.ax3.plot(t, load, color='steelblue', linewidth=1.8)
            self.ax3.fill_between(t, load, alpha=0.2, color='steelblue')
            self.ax3.axhline(40, color='orange', linestyle=':', linewidth=1.0, label='Moderat (40)')
            self.ax3.axhline(70, color='red',    linestyle=':', linewidth=1.0, label='Hoch (70)')
            self.ax3.set_ylim(0, 105)
            self.ax3.text(0.99, 0.95, f'Index: {cur_load:.0f}',
                          transform=self.ax3.transAxes, ha='right', va='top',
                          fontsize=13, fontweight='bold', color=load_col)
        self.ax3.set_title("Belastungsindex (kombiniert)")
        self.ax3.set_xlabel("Zeit (s)")
        self.ax3.set_ylabel("Index (0–100)")
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.4)

        self.fig.tight_layout()
        self.canvas.draw()
        self.info_var.set(f"Datenpunkte: {len(t)}  |  Port: 56671")
        self.win.after(200, self._update)  # ~5 Aktualisierungen/Sek.

    def close(self):
        self._running = False
        plt.close(self.fig)
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-UI
# ─────────────────────────────────────────────────────────────────────────────

def show_controller_ui():
    root = tk.Tk()
    root.title("Sphero Reha-Controller")
    root.geometry("480x380")
    root.minsize(420, 320)
    root.resizable(True, True)

    status_var  = tk.StringVar(value=last_status)
    sensor_var  = tk.StringVar(value="Keine Sensordaten")
    state_var   = tk.StringVar(value="◯  neutral")
    heading_var = tk.StringVar(value="Heading: 0°")

    graph_window_ref = [None]
    camera_thread_ref = [None]

    # ── Haupt-Frame ────────────────────────────────────────────────────────────
    main = ttk.Frame(root, padding=18)
    main.pack(fill="both", expand=True)

    ttk.Label(main, text="Sphero Reha-Controller",
              font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(main, textvariable=status_var,
              wraplength=430).pack(anchor="w", pady=(6, 12))

    # ── Buttons ────────────────────────────────────────────────────────────────
    buttons = ttk.Frame(main)
    buttons.pack(fill="x")

    start_button  = ttk.Button(buttons, text="▶  Sphero starten")
    stop_button   = ttk.Button(buttons, text="■  Sphero stoppen")
    graph_button  = ttk.Button(buttons, text="📊  Live Graphen")
    camera_button = ttk.Button(buttons, text="📷  Kamera starten")

    start_button.grid( row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
    stop_button.grid(  row=0, column=1, sticky="ew",              pady=3)
    graph_button.grid( row=1, column=0, sticky="ew", padx=(0, 6), pady=3)
    camera_button.grid(row=1, column=1, sticky="ew",              pady=3)
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    ttk.Separator(main).pack(fill="x", pady=12)

    # ── Live-Anzeige ──────────────────────────────────────────────────────────
    info_frame = ttk.Frame(main)
    info_frame.pack(fill="x")

    state_label = tk.Label(info_frame, textvariable=state_var,
                           font=("Segoe UI", 12, "bold"), fg="#888888", anchor="w")
    state_label.grid(row=0, column=0, sticky="w")

    ttk.Label(info_frame, textvariable=heading_var,
              font=("Segoe UI", 11)).grid(row=0, column=1, sticky="e", padx=(20, 0))

    ttk.Label(info_frame, textvariable=sensor_var,
              font=("Consolas", 9), foreground="#555").grid(
              row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    info_frame.columnconfigure(0, weight=1)

    # Trennlinie + Port-Info
    ttk.Separator(main).pack(fill="x", pady=(12, 4))
    ttk.Label(main, text="Flask-Server: http://0.0.0.0:56671/sensorlog",
              font=("Consolas", 8), foreground="#777").pack(anchor="w")

    STATE_COLORS = {
        "forward": ("#007700", "▶  vorwärts"),
        "right":   ("#cc5500", "↻  rechts"),
        "left":    ("#0055cc", "↺  links"),
        "neutral": ("#888888", "◯  neutral"),
    }

    # ── UI-Refresh-Loop ────────────────────────────────────────────────────────
    def refresh_ui():
        with data_lock:
            gx          = latest_data["gx"]
            gy          = latest_data["gy"]
            gz          = latest_data["gz"]
            state       = latest_data["state"]
            heading     = latest_data["heading"]
            status_text = last_status

        sensor_var.set(f"gX={gx:+.2f}  gY={gy:+.2f}  gZ={gz:+.2f}")
        status_var.set(status_text)
        heading_var.set(f"Heading: {int(heading)}°")

        color, label = STATE_COLORS.get(state, ("#888888", "◯  neutral"))
        state_var.set(label)
        state_label.config(fg=color)

        sphero_running = control_thread is not None and control_thread.is_alive()
        start_button.config(state="disabled" if sphero_running else "normal")
        stop_button.config( state="normal"   if sphero_running else "disabled")

        cam_running = (camera_thread_ref[0] is not None and
                       camera_thread_ref[0].is_alive())
        camera_button.config(
            text="⏹  Kamera stoppen" if cam_running else "📷  Kamera starten"
        )

        root.after(200, refresh_ui)

    # ── Button-Handler ─────────────────────────────────────────────────────────
    def start_clicked():
        started = start_sphero_control()
        set_status("Suche Sphero BOLT..." if started else "Sphero-Steuerung läuft bereits.")

    def stop_clicked():
        stop_sphero_control()

    def open_graphs():
        start_server_once()
        gw = graph_window_ref[0]
        if gw is None or not gw.win.winfo_exists():
            graph_window_ref[0] = LiveGraphWindow(root)
        else:
            gw.win.lift()

    def toggle_camera():
        cam = camera_thread_ref[0]
        if cam is not None and cam.is_alive():
            stop_camera.set()
            set_status("Kamera wird gestoppt...")
        else:
            stop_camera.clear()
            start_server_once()
            t = threading.Thread(target=run_camera, daemon=True)
            t.start()
            camera_thread_ref[0] = t
            set_status("Kamera startet (ZED2i)...")

    def on_close():
        stop_sphero_control()
        stop_camera.set()
        root.destroy()


    # ── Callbacks zuweisen ────────────────────────────────────────────────────
    start_button.config( command=start_clicked)
    stop_button.config(  command=stop_clicked,  state="disabled")
    graph_button.config( command=open_graphs)
    camera_button.config(command=toggle_camera)
    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_ui()
    root.mainloop()
   

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        show_controller_ui()
    except KeyboardInterrupt:
        stop_sphero.set()
        stop_camera.set()
