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
import csv
import os
import json
import concurrent.futures
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import numpy as np

# Matplotlib – TkAgg Backend VOR plt-Import setzen
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sphero
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# Probandenverwaltung (pseudonymisierte Stammdaten + Eingabemaske)
import probanden

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Referenzuhr – EINE Zeitbasis für Graphen, CSV-Dateien und Video
# ─────────────────────────────────────────────────────────────────────────────

class MasterClock:
    """
    Gemeinsame Zeitbasis für alle Datenquellen der Anwendung.

    Warum nicht einfach time.time()/datetime.now() an jeder Stelle einzeln?
      1. time.time() ist NICHT monoton. Eine NTP-Korrektur, ein Wechsel der
         Zeitzone oder die Sommerzeitumstellung lassen die Uhr springen – in
         einer 20-minütigen Aufnahme reicht ein Sprung, um Video, Sensor-CSV
         und Graph gegeneinander zu verschieben. Für Zeitdifferenzen wird
         deshalb ausschließlich perf_counter() (monoton) verwendet.
      2. Zwei getrennte Uhrenabfragen (einmal für t_rel, einmal für den
         ISO-Zeitstempel) liefern zwei minimal verschiedene Zeitpunkte. Hier
         wird pro Messwert genau EIN monotoner Zeitstempel gelesen und die
         Wanduhrzeit daraus berechnet. Dadurch gilt in jeder Zeile exakt:
             timestamp == clock_start_iso + t_abs_s
         und t_rel_s / t_abs_s sind untereinander vergleichbar.

    Zeitfelder, die in allen CSV-Dateien auftauchen:
      t_abs_s   Sekunden seit Programmstart – gemeinsame Achse aller Quellen,
                auch über mehrere Aufzeichnungen hinweg (Basis der Live-Graphen)
      t_rel_s   Sekunden seit Start DIESER Aufzeichnung (0 = Aufnahmebeginn)
      timestamp Wanduhrzeit als ISO-8601 mit Millisekunden
    """

    def __init__(self):
        self._t0_mono = time.perf_counter()
        self._t0_wall = datetime.now()

    @property
    def start_wall(self) -> datetime:
        return self._t0_wall

    def now(self):
        """Liefert (t_abs_s, Wanduhrzeit) aus EINEM monotonen Zeitstempel."""
        t_abs = time.perf_counter() - self._t0_mono
        return t_abs, self._t0_wall + timedelta(seconds=t_abs)

    def t_abs(self) -> float:
        return time.perf_counter() - self._t0_mono

    def wall_of(self, t_abs: float) -> datetime:
        """Rechnet eine t_abs-Sekundenangabe in die Wanduhrzeit zurück."""
        return self._t0_wall + timedelta(seconds=t_abs)

    @staticmethod
    def iso(wall: datetime) -> str:
        return wall.isoformat(timespec="milliseconds")


# Wird beim Programmstart einmal angelegt und nie zurückgesetzt: nur so bleiben
# Live-Graph und alle Aufzeichnungen einer Sitzung auf derselben Achse.
clock = MasterClock()

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
    "last_update": 0.0,   # time.time() der letzten echten Sensor-POST; 0.0 = noch nie
}

# Live-Graph-Daten (Zeitachse = t_abs_s der gemeinsamen Referenzuhr)
MAX_POINTS        = 300
intensity_values  = deque(maxlen=MAX_POINTS)
heart_rate_values = deque(maxlen=MAX_POINTS)
graph_time_values = deque(maxlen=MAX_POINTS)

# ── Sphero-Konfiguration ──────────────────────────────────────────────────────
MIN_SPEED_DYN        = 30
MAX_SPEED_DYN        = 120
TURN_SPEED_FACTOR    = 0.6

# Wartezeit in Neutralstellung, bis der Stopp-Befehl geht (war 1.5).
# Kleiner = der Sphero bleibt schneller stehen, wenn die Hand zurückgenommen
# wird. Zu klein sollte er nicht sein, sonst löst jedes Wackeln um die
# Neutralschwelle ein Stopp/Start-Paar aus.
STOP_TIME            = 0.6

# Kippschwellen für die Drehung. Der Betrag der Schwelle ist zugleich der
# Punkt, ab dem calc_turn() zu drehen beginnt (siehe dort).
# Links liegt bewusst NIEDRIGER als rechts: Bei am Handgelenk getragener Uhr
# ist die Drehung in die eine Richtung anatomisch schwerer zu erreichen als in
# die andere. Wer stattdessen symmetrisch fahren möchte, setzt beide auf 0.80.
GY_RIGHT_THRESHOLD   = -0.80
GY_LEFT_THRESHOLD    = +0.72
GX_FORWARD_MAX       = +0.1
GX_NEUTRAL_THRESHOLD = +0.12
MAX_TURN_ANGLE       = 90

# ── Reaktionsgeschwindigkeit der Steuerung ────────────────────────────────────
# Ein Schleifendurchlauf dauert ungefähr ROLL_COMMAND_DURATION + CONTROL_LOOP_SLEEP,
# denn sphero.roll() schläft die angegebene Dauer selbst ab. Kleinere Werte
# heißt: der Sphero setzt Änderungen der Handhaltung schneller um.
# ACHTUNG, das ist ein Kompromiss: jeder Durchlauf sendet zwei quittierte
# BLE-Befehle (roll + internes stop_roll). Kürzere Zeiten erhöhen also die
# Funklast – und hohe Funklast ist genau das, was die Verbindung im
# Kamerabetrieb abreißen lässt. Gemessen:
#     0.10 / 0.05  ->  ~150 ms je Durchlauf, ~13 Befehle/s   (bisheriger Stand)
#     0.07 / 0.04  ->  ~118 ms je Durchlauf, ~17 Befehle/s   (jetzt eingestellt)
# ERSTE MASSNAHME, falls die Verbindung wieder instabil wird: hier zurück auf
# 0.1 und 0.05. Das kostet Reaktionsschnelligkeit, aber nichts an der Funktion.
ROLL_COMMAND_DURATION = 0.07   # Sekunden, die roll() intern wartet
CONTROL_LOOP_SLEEP    = 0.04   # Sekunden Pause am Ende jedes Durchlaufs
BACKWARD_DURATION = 2.0   # Sekunden Rückwärtsfahrt pro Double Tap
BACKWARD_SPEED    = 80    # Geschwindigkeit während der Rückwärtsfahrt
DATA_TIMEOUT      = 1.0   # Sekunden ohne Sensor-POST → Sphero gilt als "keine Daten", bleibt stehen

# ── Live-Graph-Konfiguration ─────────────────────────────────────────────────
HR_WARN    = 100
HR_DANGER  = 120
SMOOTH_WIN = 10

# ── Video ─────────────────────────────────────────────────────────────────────
# Bildrate, mit der video.mp4 geschrieben wird. Sie ist NUR nominell: liefert
# die ZED tatsächlich weniger Bilder (Rechenlast, Body-Tracking), läuft das
# Video schneller ab als die Realität. Für die Postanalyse ist deshalb
# video_frames.csv maßgeblich – dort steht zu jedem Bild die exakte Zeit. Die
# gemessene Bildrate wird zusätzlich in metadata.json festgehalten.
VIDEO_FPS_NOMINAL = 30.0

# ── Kamera-Qualitätsstufen ────────────────────────────────────────────────────
# Die hier eingestellte Kombination (NEURAL-Tiefe + ACCURATE-Körpermodell +
# Body-Fitting bei HD720@30) ist die rechenintensivste, die die ZED 2i anbietet,
# und erzeugt gleichzeitig die höchste USB3-Datenrate. Beides wirkt auf die
# Bluetooth-Verbindung: USB3 strahlt im 2,4-GHz-Band, in dem auch BLE arbeitet.
# Der Fahrbetrieb ist inzwischen gegen dadurch beschädigte Antwortpakete
# abgesichert (siehe _send_drive_packet). Falls die Störungszählung im Log
# ("beschaedigte Antwortpakete") dennoch hoch bleibt, sind das hier die
# Stellschrauben – zuerst DEPTH_MODE, dann das Körpermodell.
CAM_DEPTH_MODE        = "NEURAL"              # NEURAL | ULTRA | QUALITY | PERFORMANCE
CAM_BODY_MODEL        = "HUMAN_BODY_ACCURATE"  # ..._ACCURATE | ..._MEDIUM | ..._FAST
CAM_ENABLE_BODY_FIT   = True
CAM_RESOLUTION        = "HD720"
CAM_FPS               = 30


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
    """
    Drehwinkel pro Schleifendurchlauf, 0° an der Kippschwelle bis
    MAX_TURN_ANGLE bei voller Kippung.

    Der Nullpunkt ist die Schwelle DERSELBEN Seite. Vorher stand hier eine
    gemeinsame Konstante (TURN_DEADZONE = 0.80). Solange beide Schwellen bei
    0.80 lagen, war das gleichwertig – sobald die linke Schwelle aber tiefer
    liegt, entstünde dazwischen ein toter Bereich: Der Zustand wäre "links",
    der Drehwinkel aber 0. Der Sphero würde also in den Kurvenmodus wechseln
    (langsamer werden, Farbe umschalten), ohne sich zu drehen.
    """
    threshold = abs(GY_LEFT_THRESHOLD) if gy_value > 0 else abs(GY_RIGHT_THRESHOLD)
    span      = max(1.0 - threshold, 1e-6)
    intensity = (abs(gy_value) - threshold) / span
    return max(0.0, min(1.0, intensity)) * MAX_TURN_ANGLE


def moving_average(data: list, window: int) -> np.ndarray:
    """
    Nachlaufender (kausaler) gleitender Mittelwert.

    Vorher wurde np.convolve(..., mode='same') verwendet. Das zentriert das
    Fenster und füllt die Ränder mit Nullen auf – die Kurve wird dadurch am
    Anfang UND am aktuellen Ende künstlich nach unten gezogen. Im Live-Graph
    heißt das, dass genau der neueste Messwert zu klein dargestellt wird, und
    in den gespeicherten Sitzungsdiagrammen entsteht am Rand ein Artefakt, das
    keine Bewegung abbildet. Jeder Punkt ist hier der Mittelwert der letzten
    `window` Werte (am Anfang entsprechend über weniger Werte).
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    counts = np.minimum(np.arange(1, arr.size + 1), window)
    starts = np.maximum(np.arange(arr.size) + 1 - window, 0)
    return (cumsum[1:] - cumsum[starts]) / counts


def valid_hr(hr) -> bool:
    """
    Plausibilitätsprüfung der Herzfrequenz.

    Die Uhr sendet 0, solange noch kein Puls gemessen wurde. Diese Nullen
    dürfen nicht als "0 BPM" in Diagramme oder in den Belastungsindex
    einfließen – sie sind fehlende Werte, keine Messwerte.
    """
    try:
        return 30.0 <= float(hr) <= 240.0
    except (TypeError, ValueError):
        return False


def compute_load_index(intensities: list, heart_rates: list) -> list:
    """
    Kombinierter Belastungsindex 0–100 aus Bewegungsintensität und Puls.

    Ohne gültige Herzfrequenz ist der Index NICHT definiert und wird als NaN
    (Lücke in der Kurve) geliefert. Vorher floss eine fehlende Herzfrequenz als
    0 BPM ein, was den Index systematisch zu niedrig ausgewiesen hat. Ihn
    stattdessen nur aus der Bewegung zu berechnen wäre genauso falsch: dann
    stünden in einer Kurve zwei unterschiedlich definierte Größen
    nebeneinander, die in der Auswertung nicht vergleichbar sind.
    """
    if not intensities:
        return []
    max_i = max(intensities) if max(intensities) > 0 else 1
    result = []
    for i, hr in zip(intensities, heart_rates):
        if not valid_hr(hr):
            result.append(float("nan"))
            continue
        norm_i  = min(i / max_i, 1.0)
        norm_hr = max(min((float(hr) - 60) / 120, 1.0), 0)
        result.append((0.6 * norm_i + 0.4 * norm_hr) * 100)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sitzungsaufzeichnung – synchronisierte Roh- und Auswertungsdaten je Sitzung
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

# Zeitspalten in ALLEN Tabellen (identische Bedeutung, siehe MasterClock):
#   t_rel_s   Sekunden seit Start dieser Aufzeichnung  → Sync-Schlüssel innerhalb der Sitzung
#   t_abs_s   Sekunden seit Programmstart              → gemeinsame Achse mit den Live-Graphen
#   timestamp Wanduhrzeit ISO-8601 mit Millisekunden   → für Berichte und externe Quellen
_CSV_SCHEMAS = {
    "sensor":       ["t_rel_s", "t_abs_s", "timestamp", "gx", "gy", "gz",
                      "accel_x", "accel_y", "accel_z", "heart_rate", "intensity"],
    "control":      ["t_rel_s", "t_abs_s", "timestamp", "gx", "gy", "gz",
                      "state", "heading_deg", "speed_cmd", "is_stopped"],
    "tracking":     ["t_rel_s", "t_abs_s", "timestamp", "person_id", "distance_m",
                      "angle_left_deg", "angle_right_deg"],
    "events":       ["t_rel_s", "t_abs_s", "timestamp", "event", "detail"],
    "video_frames": ["frame_index", "t_rel_s", "t_abs_s", "timestamp"],
}


class SessionRecorder:
    """
    Zeichnet eine komplette Reha-Sitzung threadsicher und zeitsynchron auf.

    Jede Zeile jeder Teil-Tabelle (Sensor, Steuerung, Tracking, Events) und
    jeder Videoframe bekommt dieselbe Referenzuhr (t_rel_s = Sekunden seit
    Sitzungsstart). Dadurch lässt sich in der Postanalyse jeder Messwert
    exakt einem Videoabschnitt zuordnen, unabhängig von Sensor-Sende- oder
    Kamera-Framerate-Schwankungen.
    """

    FLUSH_INTERVAL_S = 0.5   # gebündeltes Flush-Intervall statt Flush pro Zeile

    def __init__(self):
        self._lock          = threading.RLock()
        self.active         = False
        self.session_dir    = None
        self.start_time     = None
        self._start_wall    = None
        self.participant_id = None
        self.video_consent  = False
        self._participant   = None
        self._t_offset      = 0.0   # t_abs der Referenzuhr beim Aufnahmestart
        self._video_first_t = None
        self._video_last_t  = None
        self._files         = {}
        self._writers       = {}
        self._counts        = {}
        self._last_flush    = 0.0
        self._video_writer  = None
        self._video_frame_count = 0
        self._subsystems_seen = {"sphero": False, "camera": False}

        # Gepufferte Werte für die automatisch erzeugten Auswertungsdiagramme
        self._plot = {
            "t": [], "intensity": [], "heart_rate": [],
            "angle_t": [], "angle_left": [], "angle_right": [],
            "dist_t": [], "distance": [],
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, participant: dict, video_consent: bool):
        """
        Startet eine Aufzeichnung für eine konkrete, bereits angelegte Testperson.

        `participant` ist der pseudonymisierte Stammdatensatz (ohne Klarnamen);
        ohne ihn wird nicht aufgezeichnet. `video_consent` entscheidet, ob
        überhaupt eine Videodatei entsteht – die Videoaufzeichnung ist
        freiwillig und kann pro Sitzung einzeln widerrufen werden.
        """
        with self._lock:
            if self.active:
                return None
            if not participant or not participant.get("participant_id"):
                raise ValueError("Aufzeichnung ohne ausgewählte Testperson nicht möglich.")

            self.participant_id = participant["participant_id"]
            self.video_consent  = bool(video_consent)
            self._participant   = probanden.analysis_view(participant)

            # Sitzungen liegen unter sessions/<Teilnehmer-ID>/, damit in der
            # Auswertung alle Aufnahmen einer Person beisammen liegen.
            participant_dir = os.path.join(SESSIONS_DIR, self.participant_id)
            os.makedirs(participant_dir, exist_ok=True)
            session_id = (f"{self.participant_id}_session_"
                          + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            self.session_dir = os.path.join(participant_dir, session_id)
            os.makedirs(os.path.join(self.session_dir, "plots"), exist_ok=True)

            # Zeitbasis der Aufnahme: EIN Zugriff auf die Referenzuhr, alles
            # weitere wird daraus abgeleitet (t_rel_s = t_abs_s - _t_offset).
            t_abs, wall      = clock.now()
            self._t_offset   = t_abs
            self.start_time  = time.time()      # nur für die Anzeige "REC 12s"
            self._start_wall = MasterClock.iso(wall)
            self._counts     = {name: 0 for name in _CSV_SCHEMAS}
            self._last_flush = 0.0
            self._subsystems_seen = {"sphero": False, "camera": False}
            self._video_writer = None
            self._video_frame_count = 0
            self._video_first_t = None
            self._video_last_t  = None
            for key in self._plot:
                self._plot[key] = []

            for name, header in _CSV_SCHEMAS.items():
                # Ohne Video-Einwilligung entsteht auch kein Frame-Index.
                if name == "video_frames" and not self.video_consent:
                    continue
                path = os.path.join(self.session_dir, f"{name}.csv")
                f = open(path, "w", newline="", encoding="utf-8")
                writer = csv.writer(f)
                writer.writerow(header)
                self._files[name]   = f
                self._writers[name] = writer

            self._write_metadata(final=False)
            self.active = True
            self._log_event_locked("session_start", session_id)
            self._log_event_locked("participant", self.participant_id)
            self._log_event_locked(
                "video_consent", "erteilt" if self.video_consent else "nicht erteilt")
            return self.session_dir

    def stop(self):
        with self._lock:
            if not self.active:
                return None
            self._log_event_locked("session_end", "")
            self.active = False

            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

            for f in self._files.values():
                f.close()
            self._files.clear()
            self._writers.clear()

            self._write_metadata(final=True)
            self._save_plots()

            finished_dir = self.session_dir
            self.session_dir = None
            return finished_dir

    # ── Logging-Methoden (threadsicher) ──────────────────────────────────────

    def _now(self):
        """
        Ein einziger Uhrenzugriff pro Datenzeile – daraus werden alle drei
        Zeitangaben abgeleitet, sodass sie garantiert denselben Zeitpunkt
        bezeichnen (siehe MasterClock).
        """
        t_abs, wall = clock.now()
        return t_abs - self._t_offset, t_abs, MasterClock.iso(wall)

    def _maybe_flush(self):
        """
        Puffert Schreibzugriffe und flusht nur alle FLUSH_INTERVAL_S auf die
        Platte, statt bei jeder einzelnen Zeile. log_sensor()/log_control()
        laufen im Flask-Request-Thread bzw. im 20-Hz-Sphero-Steuerungs-Thread –
        ein flush() (Disk-Syscall) bei jedem einzelnen Aufruf hat dort spürbare
        Latenz verursacht (verzögerte Sensordaten, BLE-Timing-Störungen).
        """
        now = time.time()
        if now - self._last_flush >= self.FLUSH_INTERVAL_S:
            for f in self._files.values():
                f.flush()
            self._last_flush = now

    def log_sensor(self, gx, gy, gz, ax, ay, az, hr, intensity):
        if not self.active:
            return
        with self._lock:
            if not self.active:
                return
            t_rel, t_abs, ts = self._now()
            # Ungültige Herzfrequenz (Uhr sendet 0, solange kein Puls vorliegt)
            # wird als leeres Feld geschrieben = fehlender Wert (NaN in pandas),
            # nicht als Messwert 0.
            hr_out = hr if valid_hr(hr) else ""
            self._writers["sensor"].writerow(
                [f"{t_rel:.3f}", f"{t_abs:.3f}", ts, gx, gy, gz, ax, ay, az,
                 hr_out, intensity])
            self._counts["sensor"] += 1
            self._plot["t"].append(t_rel)
            self._plot["intensity"].append(intensity)
            self._plot["heart_rate"].append(hr)
            self._maybe_flush()

    def log_control(self, gx, gy, gz, state, heading, speed_cmd, is_stopped):
        if not self.active:
            return
        with self._lock:
            if not self.active:
                return
            self._subsystems_seen["sphero"] = True
            t_rel, t_abs, ts = self._now()
            self._writers["control"].writerow(
                [f"{t_rel:.3f}", f"{t_abs:.3f}", ts, gx, gy, gz,
                 state, heading, speed_cmd, is_stopped])
            self._counts["control"] += 1
            self._maybe_flush()

    def log_tracking(self, person_id, distance, angle_left, angle_right):
        if not self.active:
            return
        with self._lock:
            if not self.active:
                return
            self._subsystems_seen["camera"] = True
            t_rel, t_abs, ts = self._now()
            self._writers["tracking"].writerow(
                [f"{t_rel:.3f}", f"{t_abs:.3f}", ts, person_id, distance,
                 angle_left, angle_right])
            self._counts["tracking"] += 1
            self._maybe_flush()
            if distance is not None:
                self._plot["dist_t"].append(t_rel)
                self._plot["distance"].append(distance)
            if angle_left is not None or angle_right is not None:
                self._plot["angle_t"].append(t_rel)
                self._plot["angle_left"].append(angle_left)
                self._plot["angle_right"].append(angle_right)

    def log_event(self, event: str, detail: str = ""):
        if not self.active:
            return
        with self._lock:
            self._log_event_locked(event, detail)

    def _log_event_locked(self, event: str, detail: str):
        if "events" not in self._writers:
            return
        t_rel, t_abs, ts = self._now()
        self._writers["events"].writerow(
            [f"{t_rel:.3f}", f"{t_abs:.3f}", ts, event, detail])
        self._files["events"].flush()
        self._counts["events"] += 1

    def write_video_frame(self, frame, cv2):
        """
        Schreibt einen bereits mit Overlays versehenen Frame in die Session-Videodatei.

        Ohne erteilte Video-Einwilligung wird der Frame verworfen: Das
        Kamerabild bleibt zur Live-Rückmeldung auf dem Bildschirm, es entsteht
        aber keinerlei Videodatei auf der Platte.
        """
        if not self.active or not self.video_consent:
            return
        with self._lock:
            if not self.active or not self.video_consent:
                return
            self._subsystems_seen["camera"] = True
            t_rel, t_abs, ts = self._now()

            if self._video_writer is None:
                h, w = frame.shape[:2]
                path = os.path.join(self.session_dir, "video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS_NOMINAL, (w, h))
                self._video_first_t = t_rel

            self._video_writer.write(frame)
            self._writers["video_frames"].writerow(
                [self._video_frame_count, f"{t_rel:.3f}", f"{t_abs:.3f}", ts])
            self._video_last_t = t_rel
            self._video_frame_count += 1
            self._counts["video_frames"] += 1
            self._maybe_flush()

    # ── Abschluss: Metadaten + Diagramme ─────────────────────────────────────

    def rec_start_t_abs(self):
        """t_abs der laufenden Aufzeichnung (für die Markierung im Live-Graph)."""
        return self._t_offset if self.active else None

    def _measured_fps(self):
        """Tatsächlich erreichte Bildrate des aufgezeichneten Videos."""
        if (self._video_first_t is None or self._video_last_t is None
                or self._video_frame_count < 2):
            return None
        span = self._video_last_t - self._video_first_t
        if span <= 0:
            return None
        return round((self._video_frame_count - 1) / span, 2)

    def _write_metadata(self, final: bool):
        video_recorded = self.video_consent and self._subsystems_seen["camera"]
        meta = {
            "session_id": os.path.basename(self.session_dir),
            "participant_id": self.participant_id,
            "participant": self._participant,
            "video_consent": self.video_consent,
            "start_time_iso": self._start_wall,
            "end_time_iso": MasterClock.iso(clock.wall_of(clock.t_abs())) if final else None,
            "duration_s": round(clock.t_abs() - self._t_offset, 3) if final else None,
            # Alles, was zum Zusammenführen der Quellen in der Postanalyse nötig ist.
            "time_base": {
                "columns": {
                    "t_rel_s": "Sekunden seit Start dieser Aufzeichnung (0 = Aufnahmebeginn)",
                    "t_abs_s": "Sekunden seit Programmstart (gemeinsame Achse mit den Live-Graphen)",
                    "timestamp": "Wanduhrzeit ISO-8601 mit Millisekunden",
                },
                "clock": "time.perf_counter (monoton); Wanduhrzeit = program_start_iso + t_abs_s",
                "program_start_iso": MasterClock.iso(clock.start_wall),
                "session_offset_s": round(self._t_offset, 3),
                "sync_key": "t_rel_s",
            },
            "video": {
                "consent": self.video_consent,
                "fps_nominal": VIDEO_FPS_NOMINAL if video_recorded else None,
                "fps_measured": self._measured_fps() if final else None,
                "frame_count": self._video_frame_count,
                "note": ("Bildzeitpunkte stehen in video_frames.csv. Bei Abweichung "
                         "zwischen fps_nominal und fps_measured ist video.mp4 zeitlich "
                         "gestaucht/gestreckt – für die Auswertung immer die Zeiten aus "
                         "video_frames.csv verwenden."),
            },
            "subsystems_active": self._subsystems_seen,
            "row_counts": dict(self._counts),
            "config": {
                "MIN_SPEED_DYN": MIN_SPEED_DYN, "MAX_SPEED_DYN": MAX_SPEED_DYN,
                "TURN_SPEED_FACTOR": TURN_SPEED_FACTOR, "STOP_TIME": STOP_TIME,
                "GY_RIGHT_THRESHOLD": GY_RIGHT_THRESHOLD, "GY_LEFT_THRESHOLD": GY_LEFT_THRESHOLD,
                "GX_FORWARD_MAX": GX_FORWARD_MAX, "GX_NEUTRAL_THRESHOLD": GX_NEUTRAL_THRESHOLD,
                "MAX_TURN_ANGLE": MAX_TURN_ANGLE,
                "ROLL_COMMAND_DURATION": ROLL_COMMAND_DURATION,
                "CONTROL_LOOP_SLEEP": CONTROL_LOOP_SLEEP,
                "BACKWARD_DURATION": BACKWARD_DURATION, "BACKWARD_SPEED": BACKWARD_SPEED,
                "DATA_TIMEOUT": DATA_TIMEOUT,
                "HR_WARN": HR_WARN, "HR_DANGER": HR_DANGER,
            },
            "files": {
                "sensor_log": "sensor.csv", "control_log": "control.csv",
                "tracking_log": "tracking.csv", "events_log": "events.csv",
                "video": "video.mp4" if video_recorded else None,
                "video_frame_index": "video_frames.csv" if self.video_consent else None,
                "plots_dir": "plots/",
            },
        }
        with open(os.path.join(self.session_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _save_plots(self):
        plots_dir = os.path.join(self.session_dir, "plots")

        if len(self._plot["t"]) >= 2:
            t   = self._plot["t"]
            raw = self._plot["intensity"]
            hr  = self._plot["heart_rate"]
            smoothed = moving_average(raw, SMOOTH_WIN)
            load     = compute_load_index(raw, hr)
            # Fehlende Herzfrequenz als Lücke zeichnen, nicht als 0 BPM.
            hr_plot  = [float(h) if valid_hr(h) else np.nan for h in hr]

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(
                f"Sitzungs-Übersicht – {self.participant_id}   "
                f"(Start {self._start_wall})",
                fontsize=13, fontweight="bold")

            ax1.plot(t, raw, color="lightsteelblue", alpha=0.5, linewidth=0.8, label="Roh")
            ax1.plot(t, smoothed, color="steelblue", linewidth=1.8, label=f"Geglättet (n={SMOOTH_WIN})")
            ax1.set_title("Bewegungsintensität"); ax1.set_ylabel("Intensität")
            ax1.legend(loc="upper left", fontsize=8); ax1.grid(True, alpha=0.4)

            ax2.plot(t, hr_plot, color="steelblue", linewidth=1.8)
            ax2.axhline(HR_WARN, color="orange", linestyle="--", linewidth=1.2, label=f"Warnung {HR_WARN} BPM")
            ax2.axhline(HR_DANGER, color="red", linestyle="--", linewidth=1.2, label=f"Gefahr {HR_DANGER} BPM")
            ax2.set_title("Herzfrequenz"); ax2.set_ylabel("BPM")
            ax2.legend(loc="upper left", fontsize=8); ax2.grid(True, alpha=0.4)

            if load:
                ax3.plot(t, load, color="steelblue", linewidth=1.8)
                ax3.fill_between(t, load, alpha=0.2, color="steelblue")
                ax3.set_ylim(0, 105)
            ax3.set_title("Belastungsindex (kombiniert)")
            ax3.set_xlabel("Zeit (s)"); ax3.set_ylabel("Index (0–100)")
            ax3.grid(True, alpha=0.4)

            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "uebersicht.png"), dpi=150)
            plt.close(fig)

        if self._plot["angle_t"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._plot["angle_t"], self._plot["angle_left"], label="Links", color="tab:blue")
            ax.plot(self._plot["angle_t"], self._plot["angle_right"], label="Rechts", color="tab:orange")
            ax.set_title("Armstreckungswinkel"); ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Grad")
            ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "winkel.png"), dpi=150)
            plt.close(fig)

        if self._plot["dist_t"]:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(self._plot["dist_t"], self._plot["distance"], color="tab:green")
            ax.set_title("Kameraabstand"); ax.set_xlabel("Zeit (s)"); ax.set_ylabel("m")
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "abstand.png"), dpi=150)
            plt.close(fig)


recorder = SessionRecorder()


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
        # WICHTIG: time.time(), nicht die Referenzuhr. Die Steuerungsschleife
        # vergleicht backward_until und last_update mit time.time(). Beide Seiten
        # müssen dieselbe Uhr benutzen – sonst gilt jeder Sensorwert sofort als
        # veraltet, der Zustand bleibt "neutral" und der Sphero fährt nie los.
        # (Die Referenzuhr clock.t_abs() bleibt für Graphen und CSV zuständig.)
        now = time.time()
        with data_lock:
            # Watch feuert die Geste manchmal mehrfach kurz hintereinander;
            # solange das Rückwärtsfenster noch läuft, nur verlängern statt
            # jedes Mal neu zu loggen/zu drucken.
            is_new_trigger = now >= latest_data["backward_until"]
            latest_data["backward_until"] = now + BACKWARD_DURATION
        if is_new_trigger:
            recorder.log_event("double_tap", "Rückwärtsfahrt ausgelöst")
            # Nur ASCII in der Konsolenausgabe: Auf einer cp1252-Konsole wirft
            # print() bei Zeichen wie "→" einen UnicodeEncodeError. Da das hier
            # INNERHALB des Flask-Handlers passiert, würde der Double-Tap-Request
            # mit HTTP 500 abbrechen und die Rückwärtsfahrt gar nicht auslösen.
            print("[EVENT] Double Tap - Rueckwaerts")
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
    intensity = math.sqrt(ax**2 + ay**2 + az**2)

    # Zwei Uhren mit klarer Aufgabenteilung:
    #   time.time()     -> Steuerung (die Schleife vergleicht damit, s.o.)
    #   clock.t_abs()   -> Graphen und CSV (monotone Referenzuhr, siehe MasterClock)
    t_abs = clock.t_abs()

    with data_lock:
        latest_data["gx"]          = gx
        latest_data["gy"]          = gy
        latest_data["gz"]          = gz
        latest_data["state"]       = state
        latest_data["last_update"] = time.time()
        intensity_values.append(intensity)
        heart_rate_values.append(hr)
        graph_time_values.append(t_abs)

    recorder.log_sensor(gx, gy, gz, ax, ay, az, hr, intensity)
    return "OK", 200


def run_server():
    flask_app.run(host='0.0.0.0', port=56671, debug=False, use_reloader=False, threaded=True)


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
    """
    Erkennt BLE-Verbindungsabbrüche.

    Wichtig: Unter Python 3.8 ist concurrent.futures.TimeoutError NICHT
    dasselbe wie das eingebaute TimeoutError (erst ab 3.11 ein Alias!) und
    sein str() ist leer (''). Ein reiner isinstance(exc, TimeoutError)- oder
    "timeout" in msg-Check lässt diese Exception (und viele BLE-OSErrors mit
    Windows-Fehlercodes) unerkannt durchrutschen – die Steuerung erkennt den
    Verbindungsverlust dann nie und hämmert weiter erfolglos auf die tote
    Verbindung ein, was die ganze App einfrieren lässt.
    """
    if isinstance(exc, (OSError, concurrent.futures.TimeoutError)):
        return True
    # Alle bleak-Exceptions (BleakError, BleakDeviceNotFoundError, ...) sind
    # in diesem Kontext Verbindungsfehler – auch wenn ihr str() weder
    # "bleakerror" noch "not connected" enthält (z.B. "... was not found").
    if type(exc).__module__.startswith("bleak"):
        return True
    msg = str(exc).lower()
    return "not connected" in msg or "bleakerror" in msg or "timeout" in msg


SPHERO_CMD_TIMEOUT = 1.5   # Sekunden – eigene Obergrenze pro Sphero-Befehl


class _SpheroCommandGuard:
    """
    Führt Sphero-Befehle mit selbst gesetztem, kurzem Timeout aus und stellt
    sicher, dass nie zwei Befehle gleichzeitig in die spherov2-API laufen.

    Timeout: spherov2 wartet intern bis zu 10s auf eine BLE-Antwort
    (Toy._wait_packet, hartkodiert timeout=10.0) – und SpheroEduAPI.roll()
    ruft dafür sogar zweimal hintereinander in die Bibliothek hinein (Speed
    setzen + automatisches stop_roll). Bei totem Link würde ein einzelner
    Befehl unseren Steuerungs-Thread sonst bis zu ~20 Sekunden einfrieren.
    Der eigentliche Aufruf läuft deshalb in einem Daemon-Thread; wir warten
    nur `timeout` Sekunden darauf.

    Serialisierung: Läuft ein Befehl über sein Timeout hinaus weiter
    (verwaister Thread), darf der nächste nicht parallel in die nicht
    threadsichere API greifen. Neue Aufrufe warten deshalb höchstens
    `timeout` auf das Ende des vorigen Befehls und schlagen sonst
    kontrolliert fehl. Pro (Re-)Verbindung wird eine frische Instanz
    erzeugt, damit ein hängender Befehl einer alten, toten Verbindung die
    neue nicht blockiert.
    """

    def __init__(self):
        self._busy = threading.Lock()

    def call(self, func, *args, timeout=SPHERO_CMD_TIMEOUT, **kwargs):
        if not self._busy.acquire(timeout=timeout):
            raise concurrent.futures.TimeoutError(
                f"Voriger Sphero-Befehl hängt noch – {getattr(func, '__name__', func)} übersprungen"
            )
        result = {}

        def _run():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                self._busy.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise concurrent.futures.TimeoutError(
                f"Sphero-Befehl {getattr(func, '__name__', func)} antwortet nicht innerhalb {timeout}s"
            )
        if "error" in result:
            raise result["error"]
        return result.get("value")


class _SpheroAPIFastClose(SpheroEduAPI):
    """
    SpheroEduAPI, deren __exit__ bei toter Verbindung nicht lange blockiert.

    Das Original-__exit__ sendet beim Verlassen des with-Blocks noch Befehle
    an den Sphero (ToyUtil.sleep) und joint seinen Hintergrund-Thread; auf
    einer bereits abgerissenen Verbindung wartet jeder dieser Schritte bis
    zu 10s auf eine Antwort, die nie kommt. Genau das hat nach einem
    Verbindungsverlust den Reconnect um viele Sekunden verzögert – die App
    wirkte "eingefroren". Der Abbau läuft deshalb in einem Daemon-Thread
    mit kurzer Wartezeit: hängt er, wird er aufgegeben und läuft im
    Hintergrund zu Ende; beim nächsten Verbindungsaufbau wird ohnehin neu
    gescannt und frisch aufgebaut.
    """

    # Wird von der Steuerung auf True gesetzt, wenn der Abriss erkannt wurde.
    connection_dead = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        def _teardown():
            if self.connection_dead:
                # Tote Verbindung sofort hart auf OS-Ebene kappen. Erst wenn
                # Windows die Verbindung wirklich freigibt, merkt der Sphero
                # den Abriss und beginnt wieder zu advertisen – und nur einen
                # advertisenden Sphero kann der Reconnect-Scan überhaupt
                # finden. Ohne diesen Schritt hinge die alte Verbindung noch
                # ~10s in den Timeouts des Original-__exit__ fest.
                try:
                    self._SpheroEduAPI__toy._Toy__adapter.close()
                except Exception:
                    pass
            try:
                SpheroEduAPI.__exit__(self, exc_type, exc_val, exc_tb)
            except Exception:
                pass

        t = threading.Thread(target=_teardown, daemon=True)
        t.start()
        t.join(4.0)


def control_sphero():
    global sphero_api

    MAX_RECONNECTS   = 5
    RECONNECT_DELAY  = 3.0   # Sekunden zwischen Reconnect-Versuchen
    reconnect_count  = 0
    sphero_heading   = 0     # Heading über Reconnects hinweg behalten

    # Ein einzelner fehlgeschlagener Befehl bedeutet nicht zwangsläufig einen
    # echten Verbindungsabbruch – z.B. kann ein Zusammenstoß (Wand, Hindernis)
    # die Sphero-Firmware kurz blockieren, sodass ein einzelner Befehl einen
    # Timeout wirft, obwohl die BLE-Verbindung Sekundenbruchteile später
    # wieder normal reagiert. Erst nach mehreren Fehlversuchen in Folge gilt
    # die Verbindung als wirklich verloren (→ voller Reconnect-Zyklus).
    MAX_CONSECUTIVE_FAILURES = 3

    while not stop_sphero.is_set() and reconnect_count <= MAX_RECONNECTS:

        # ── Verbinden ─────────────────────────────────────────────────────────
        if reconnect_count == 0:
            set_status("Suche Sphero BOLT...")
        else:
            set_status(f"Reconnect {reconnect_count}/{MAX_RECONNECTS} – suche Sphero...")

        try:
            toy = scanner.find_toy()
            scan_error = None
        except Exception as e:
            toy        = None
            scan_error = e

        if not toy:
            if reconnect_count == 0:
                # Erststart: Sphero ist vermutlich aus oder Bluetooth fehlt –
                # klare Meldung und Ende.
                if scan_error is not None:
                    set_status(f"Sphero-Start fehlgeschlagen: {scan_error}")
                else:
                    set_status("Kein Sphero gefunden. Bluetooth und Sphero prüfen.")
                return
            # Reconnect-Fall: Direkt nach einem Abriss advertist der Sphero
            # oft noch nicht wieder (die alte Verbindung muss OS- und
            # firmwareseitig erst freigegeben werden). Ein leerer Scan ist
            # hier also zu ERWARTEN – als Fehlversuch zählen und erneut
            # scannen, statt wie früher die Steuerung komplett zu beenden.
            reconnect_count += 1
            recorder.log_event("sphero_rescan_failed",
                               f"Versuch {reconnect_count}/{MAX_RECONNECTS}")
            if reconnect_count > MAX_RECONNECTS:
                set_status(
                    f"Sphero nach {MAX_RECONNECTS} Scan-Versuchen nicht wiedergefunden. "
                    "Sphero bitte aus- und wieder einschalten, dann neu starten."
                )
                recorder.log_event("sphero_reconnect_gave_up", "")
                return
            set_status(
                f"Sphero noch nicht wieder sichtbar – neuer Scan in "
                f"{int(RECONNECT_DELAY)}s ({reconnect_count}/{MAX_RECONNECTS})..."
            )
            time.sleep(RECONNECT_DELAY)
            continue

        # ── Steuerungs-Loop ───────────────────────────────────────────────────
        connection_lost = False
        try:
            with _SpheroAPIFastClose(toy) as sphero:
                sphero_api           = sphero
                last_move_time       = time.time()
                is_stopped           = True
                consecutive_failures = 0
                was_backward         = False
                guard                = _SpheroCommandGuard()
                current_led          = None

                def set_led(r, g, b):
                    # LED nur bei Farbwechsel senden – vorher wurde dieselbe
                    # Farbe bei jedem Schleifendurchlauf erneut gefunkt und
                    # hat den BLE-Verkehr unnötig aufgebläht.
                    nonlocal current_led
                    if current_led != (r, g, b):
                        guard.call(sphero.set_main_led, Color(r=r, g=g, b=b))
                        current_led = (r, g, b)

                guard.call(sphero.set_heading, sphero_heading)
                set_led(255, 255, 255)

                if reconnect_count == 0:
                    set_status("Sphero verbunden. Sensor-App kann Daten senden.")
                    recorder.log_event("sphero_connected", "")
                else:
                    set_status(f"Sphero wieder verbunden (Versuch {reconnect_count}).")
                    recorder.log_event("sphero_reconnected", f"Versuch {reconnect_count}")
                reconnect_count = 0  # bei Erfolg zurücksetzen

                while not stop_sphero.is_set():
                    with data_lock:
                        gx             = latest_data["gx"]
                        gy             = latest_data["gy"]
                        gz             = latest_data["gz"]
                        backward_until = latest_data["backward_until"]
                        last_update    = latest_data["last_update"]

                    # ── Double-Tap-Rückwärtsfahrt hat Vorrang vor Gravity-Steuerung ──
                    if time.time() < backward_until:
                        try:
                            if not was_backward:
                                # Sanfter Übergang: erst stoppen und kurz
                                # ausrollen lassen. Eine abrupte Richtungs-
                                # umkehr unter Fahrt erzeugt eine Stromspitze
                                # in den Motoren, die (v.a. bei schwächerem
                                # Akku) die Versorgungsspannung einbrechen
                                # und die BLE-Verbindung abreißen lassen kann.
                                guard.call(sphero.stop_roll, int(sphero_heading))
                                set_led(160, 0, 255)
                                time.sleep(0.3)
                                was_backward = True
                            backward_heading = (sphero_heading + 180) % 360
                            guard.call(sphero.roll, int(backward_heading), BACKWARD_SPEED,
                                       ROLL_COMMAND_DURATION)
                            set_led(160, 0, 255)
                            last_move_time = time.time()
                            is_stopped     = False
                            consecutive_failures = 0
                        except Exception as e:
                            if _is_connection_error(e):
                                consecutive_failures += 1
                                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                    connection_lost = True
                                    break
                                print(f"[WARN] Rückwärts-Befehl fehlgeschlagen ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                            else:
                                print(f"[WARN] Rückwärts-Befehl fehlgeschlagen: {e}")
                        recorder.log_control(gx, gy, gz, "backward", sphero_heading, BACKWARD_SPEED, False)
                        time.sleep(CONTROL_LOOP_SLEEP)
                        continue

                    was_backward = False

                    # ── Ohne frische Sensordaten von Handy/Watch nichts fahren ──────
                    # (gx/gy/gz stehen sonst auf ihrem 0.0-Default, was get_state()
                    #  fälschlich als "forward" auswertet)
                    data_is_stale = (last_update == 0.0) or (time.time() - last_update > DATA_TIMEOUT)
                    state         = "neutral" if data_is_stale else get_state(gx, gy, gz)
                    applied_speed = 0

                    try:
                        if state == "right":
                            turn           = calc_turn(gy)
                            applied_speed  = int(calc_speed(gx) * TURN_SPEED_FACTOR)
                            sphero_heading = (sphero_heading + turn) % 360
                            guard.call(sphero.roll, int(sphero_heading), applied_speed,
                                       ROLL_COMMAND_DURATION)
                            set_led(255, 100, 0)
                            last_move_time = time.time()
                            is_stopped     = False
                            with data_lock:
                                latest_data["heading"] = sphero_heading

                        elif state == "left":
                            turn           = calc_turn(gy)
                            applied_speed  = int(calc_speed(gx) * TURN_SPEED_FACTOR)
                            sphero_heading = (sphero_heading - turn) % 360
                            guard.call(sphero.roll, int(sphero_heading), applied_speed,
                                       ROLL_COMMAND_DURATION)
                            set_led(0, 200, 255)
                            last_move_time = time.time()
                            is_stopped     = False
                            with data_lock:
                                latest_data["heading"] = sphero_heading

                        elif state == "forward":
                            applied_speed = calc_speed(gx)
                            guard.call(sphero.roll, int(sphero_heading), applied_speed,
                                       ROLL_COMMAND_DURATION)
                            set_led(0, 255, 0)
                            last_move_time = time.time()
                            is_stopped     = False

                        elif state == "neutral":
                            if not is_stopped and time.time() - last_move_time > STOP_TIME:
                                guard.call(sphero.stop_roll, int(sphero_heading))
                                is_stopped = True
                                set_led(255, 0, 0)

                        consecutive_failures = 0

                    except Exception as e:
                        if _is_connection_error(e):
                            consecutive_failures += 1
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                connection_lost = True
                                break
                            print(f"[WARN] Sphero-Befehl fehlgeschlagen ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                        else:
                            # Andere Fehler: loggen, aber weiterlaufen
                            print(f"[WARN] Sphero-Befehl fehlgeschlagen: {e}")

                    recorder.log_control(gx, gy, gz, state, sphero_heading, applied_speed, is_stopped)
                    time.sleep(CONTROL_LOOP_SLEEP)

                # Sauber beenden wenn gewollt gestoppt
                if connection_lost:
                    # __exit__ soll die tote Verbindung sofort hart kappen,
                    # damit der Sphero schnell wieder advertist und der
                    # Reconnect-Scan ihn finden kann.
                    sphero.connection_dead = True
                else:
                    try:
                        guard.call(sphero.stop_roll, 0)
                        set_led(0, 0, 0)
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
            recorder.log_event("sphero_connection_lost", f"Reconnect {reconnect_count}/{MAX_RECONNECTS}")
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
                recorder.log_event("sphero_connection_failed", "")
                return

    set_status("Sphero getrennt.")
    recorder.log_event("sphero_control_stopped", "")


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
        return None
    winkel = _berechne_winkel_3d(
        kps_3d[schulter_idx], kps_3d[ellbogen_idx], kps_3d[handgelenk_idx]
    )
    if winkel is None:
        cv2.putText(frame, f"{seite}: Nicht sichtbar",
                    (20, 80 if seite == "Links" else 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return None
    farbe = _winkel_farbe(winkel)
    cv2.circle(frame, (ex, ey), 14, farbe, -1)
    cv2.circle(frame, (ex, ey), 14, (255, 255, 255), 2)
    cv2.putText(frame, f"{winkel}", (ex - 20, ey - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"{seite}: {winkel} Grad _ {_winkel_text(winkel)}",
                (20, 80 if seite == "Links" else 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)
    return winkel


def _draw_abstand(frame, kps_3d, cv2):
    global _last_distance_condition
    p = kps_3d[2]
    if p[2] <= 0:
        return None
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
    return abstand


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
    init_params.camera_resolution = getattr(sl.RESOLUTION, CAM_RESOLUTION)
    init_params.camera_fps        = CAM_FPS
    init_params.coordinate_units  = sl.UNIT.METER
    init_params.depth_mode        = getattr(sl.DEPTH_MODE, CAM_DEPTH_MODE)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        set_status("ZED2i: Kamera konnte nicht geöffnet werden.")
        return

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_area_memory = True
    zed.enable_positional_tracking(tracking_params)

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking     = True
    body_params.detection_model     = getattr(sl.BODY_TRACKING_MODEL, CAM_BODY_MODEL)
    body_params.body_format         = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting = CAM_ENABLE_BODY_FIT

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
    recorder.log_event("camera_start", "")

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
                    distance = _draw_abstand(frame, kps_3d, cv2)

                    angle_left  = _draw_winkel(frame, kps_2d, kps_3d,
                                 schulter_idx=12, ellbogen_idx=13,
                                 handgelenk_idx=15, seite="Links", cv2=cv2)
                    angle_right = _draw_winkel(frame, kps_2d, kps_3d,
                                 schulter_idx=5, ellbogen_idx=6,
                                 handgelenk_idx=8, seite="Rechts", cv2=cv2)

                    recorder.log_tracking(body.id, distance, angle_left, angle_right)

                    head = kps_2d[27]
                    hx, hy = int(head[0]), int(head[1])
                    if 0 < hx < frame.shape[1]:
                        cv2.putText(frame, f"Person {body.id}",
                                    (hx - 40, hy - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Personen: {person_count}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if recorder.active:
            elapsed = time.time() - recorder.start_time
            cv2.putText(frame, f"REC {elapsed:6.1f}s", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            recorder.write_video_frame(frame, cv2)

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
    recorder.log_event("camera_stop", "")


# ─────────────────────────────────────────────────────────────────────────────
# Live-Graph Fenster (eingebettet in Tkinter)
# ─────────────────────────────────────────────────────────────────────────────

class LiveGraphWindow:
    """
    Tkinter-Fenster mit den Live-Graphen.

    Zeitachse: Wanduhrzeit (HH:MM:SS). Intern liegen die Punkte als t_abs_s der
    gemeinsamen Referenzuhr vor – dieselbe Größe steht als Spalte t_abs_s in
    jeder CSV-Datei. Die Achsenbeschriftung wird daraus nur zur Anzeige
    umgerechnet, sodass ein Zeitpunkt im Graph, in den CSV-Dateien und im Video
    denselben Moment bezeichnet.

    Zeichenweise: Die Kurven werden EINMAL angelegt und danach nur noch mit
    set_data() gefüttert. Vorher wurde bei jeder Aktualisierung alles verworfen
    und komplett neu aufgebaut (ax.clear(), plot(), legend(), axhspan() und
    tight_layout() 5-mal pro Sekunde). tight_layout() ist dabei der teuerste
    Schritt – der Tk-Mainloop kam nicht mehr nach, wodurch der Graph sichtbar
    stehen blieb bzw. nur unregelmäßig nachzog.
    """

    UPDATE_MS   = 200    # Aktualisierungsintervall
    WINDOW_MIN_S = 20.0  # Mindestbreite des Zeitfensters

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

        self._build_static_axes()

        self._running = True
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._update()

    # ── Einmaliger Aufbau ─────────────────────────────────────────────────────

    def _build_static_axes(self):
        # Plot 1: Bewegungsintensität
        self.line_raw,      = self.ax1.plot([], [], color='lightsteelblue',
                                            alpha=0.5, linewidth=0.8, label='Roh')
        self.line_smoothed, = self.ax1.plot([], [], color='steelblue', linewidth=1.8,
                                            label=f'Geglättet (n={SMOOTH_WIN})')
        self.ax1.set_title("Bewegungsintensität")
        self.ax1.set_ylabel("Intensität")
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.4)

        # Plot 2: Herzfrequenz
        self.ax2.axhspan(0,          HR_WARN,   alpha=0.08, color='green')
        self.ax2.axhspan(HR_WARN,    HR_DANGER, alpha=0.08, color='orange')
        self.ax2.axhspan(HR_DANGER,  260,       alpha=0.08, color='red')
        self.ax2.axhline(HR_WARN,   color='orange', linestyle='--',
                         linewidth=1.2, label=f'Warnung {HR_WARN} BPM')
        self.ax2.axhline(HR_DANGER, color='red',    linestyle='--',
                         linewidth=1.2, label=f'Gefahr {HR_DANGER} BPM')
        self.line_hr, = self.ax2.plot([], [], color='steelblue', linewidth=1.8)
        self.text_hr  = self.ax2.text(0.99, 0.95, "", transform=self.ax2.transAxes,
                                      ha='right', va='top', fontsize=13,
                                      fontweight='bold')
        self.ax2.set_title("Herzfrequenz")
        self.ax2.set_ylabel("BPM")
        self.ax2.legend(loc='upper left', fontsize=8)
        self.ax2.grid(True, alpha=0.4)

        # Plot 3: Belastungsindex
        self.line_load, = self.ax3.plot([], [], color='steelblue', linewidth=1.8)
        self.fill_load  = None
        self.ax3.axhline(40, color='orange', linestyle=':', linewidth=1.0, label='Moderat (40)')
        self.ax3.axhline(70, color='red',    linestyle=':', linewidth=1.0, label='Hoch (70)')
        self.text_load  = self.ax3.text(0.99, 0.95, "", transform=self.ax3.transAxes,
                                        ha='right', va='top', fontsize=13,
                                        fontweight='bold')
        self.ax3.set_ylim(0, 105)
        self.ax3.set_title("Belastungsindex (kombiniert)")
        self.ax3.set_xlabel("Uhrzeit (HH:MM:SS)")
        self.ax3.set_ylabel("Index (0–100)")
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.4)

        # Markierung des Aufzeichnungsbeginns – erleichtert später das
        # Zusammenführen von Graph, Video und CSV.
        self.rec_markers = []
        for ax in (self.ax1, self.ax2, self.ax3):
            marker = ax.axvline(0, color='crimson', linestyle='-', linewidth=1.4,
                                alpha=0.8, visible=False)
            self.rec_markers.append(marker)

        # Wanduhrzeit auf der gemeinsamen t_abs-Achse
        def fmt_time(x, _pos):
            return clock.wall_of(x).strftime("%H:%M:%S")

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_time))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))

        # Einmal statt bei jeder Aktualisierung.
        self.fig.tight_layout()

    # ── Laufende Aktualisierung ───────────────────────────────────────────────

    def _update(self):
        if not self._running:
            return
        # Ein Fehler in der Zeichenroutine darf die Aktualisierungskette nicht
        # abreißen lassen. Vorher wurde `after()` erst am ENDE von _update()
        # aufgerufen – trat davor eine Exception auf, war der Graph endgültig
        # eingefroren und musste neu geöffnet werden.
        try:
            self._redraw()
        except Exception as e:
            self.info_var.set(f"Zeichenfehler (Aktualisierung läuft weiter): {e}")
        if self._running:
            self.win.after(self.UPDATE_MS, self._update)

    def _redraw(self):
        with data_lock:
            if len(graph_time_values) < 2:
                self.info_var.set("Warte auf Sensordaten vom Handy...")
                return
            t   = list(graph_time_values)
            raw = list(intensity_values)
            hr  = list(heart_rate_values)
            rec_active = recorder.active
            rec_start  = recorder.rec_start_t_abs()

        smoothed = moving_average(raw, SMOOTH_WIN)
        load     = compute_load_index(raw, hr)
        # Fehlende Herzfrequenz (0 von der Uhr) als Lücke, nicht als 0 BPM.
        hr_plot  = np.array([float(h) if valid_hr(h) else np.nan for h in hr])

        # ── Kurven aktualisieren ──────────────────────────────────────────────
        self.line_raw.set_data(t, raw)
        self.line_smoothed.set_data(t, smoothed)
        self.line_hr.set_data(t, hr_plot)
        self.line_load.set_data(t, load)

        # fill_between erzeugt ein neues Objekt und muss ersetzt werden.
        if self.fill_load is not None:
            self.fill_load.remove()
        self.fill_load = self.ax3.fill_between(t, load, alpha=0.2, color='steelblue')

        # ── Achsen ────────────────────────────────────────────────────────────
        t_end   = t[-1]
        t_start = min(t[0], t_end - self.WINDOW_MIN_S)
        self.ax1.set_xlim(t_start, t_end + 0.5)   # sharex → gilt für alle drei

        max_raw = max(raw) if raw else 1.0
        self.ax1.set_ylim(0, max(max_raw * 1.15, 0.1))

        valid = hr_plot[~np.isnan(hr_plot)]
        if valid.size:
            self.ax2.set_ylim(min(50.0, float(valid.min()) - 10),
                              max(HR_DANGER + 20.0, float(valid.max()) + 10))
        else:
            self.ax2.set_ylim(50, HR_DANGER + 20)

        # ── Aktuelle Werte als Text ───────────────────────────────────────────
        cur_hr = hr[-1] if hr else 0
        if valid_hr(cur_hr):
            col = ('green' if cur_hr < HR_WARN else
                   'orange' if cur_hr < HR_DANGER else 'red')
            self.text_hr.set_text(f"{float(cur_hr):.0f} BPM")
            self.text_hr.set_color(col)
        else:
            self.text_hr.set_text("kein Puls")
            self.text_hr.set_color("#888888")

        cur_load = load[-1] if load else float("nan")
        if math.isnan(cur_load):
            # Ohne gültigen Puls ist der Index nicht definiert (siehe
            # compute_load_index) – das wird angezeigt statt eine Zahl zu erfinden.
            self.text_load.set_text("Index: –")
            self.text_load.set_color("#888888")
        else:
            self.text_load.set_text(f"Index: {cur_load:.0f}")
            self.text_load.set_color('green' if cur_load < 40 else
                                     'orange' if cur_load < 70 else 'red')

        # ── Markierung des Aufzeichnungsbeginns ───────────────────────────────
        show_marker = rec_active and rec_start is not None and rec_start >= t_start
        for marker in self.rec_markers:
            marker.set_visible(bool(show_marker))
            if show_marker:
                marker.set_xdata([rec_start, rec_start])

        # draw_idle() statt draw(): zeichnet gebündelt, wenn Tk Luft hat.
        self.canvas.draw_idle()

        wall_now = clock.wall_of(t_end).strftime("%H:%M:%S.%f")[:-3]
        if rec_active and rec_start is not None:
            rec_info = f"REC t_rel={t_end - rec_start:6.1f}s"
        else:
            rec_info = "keine Aufzeichnung"
        self.info_var.set(
            f"Letzter Wert: {wall_now}  |  t_abs={t_end:8.2f}s  |  "
            f"{rec_info}  |  Punkte: {len(t)}  |  Port: 56671"
        )

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
    root.geometry("520x570")
    root.minsize(460, 520)
    root.resizable(True, True)

    status_var  = tk.StringVar(value=last_status)
    sensor_var  = tk.StringVar(value="Keine Sensordaten")
    state_var   = tk.StringVar(value="◯  neutral")
    heading_var = tk.StringVar(value="Heading: 0°")

    graph_window_ref = [None]
    camera_thread_ref = [None]

    # ── Probandenverwaltung ───────────────────────────────────────────────────
    try:
        registry = probanden.ParticipantRegistry(SESSIONS_DIR)
    except RuntimeError as e:
        messagebox.showerror("Probandendaten nicht lesbar", str(e))
        root.destroy()
        return

    selected_pid   = [None]
    participant_var = tk.StringVar(value="")
    participant_info_var = tk.StringVar(value="Keine Testperson ausgewählt.")
    video_consent_var = tk.BooleanVar(value=False)

    # ── Haupt-Frame ────────────────────────────────────────────────────────────
    main = ttk.Frame(root, padding=18)
    main.pack(fill="both", expand=True)

    ttk.Label(main, text="Sphero Reha-Controller",
              font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(main, textvariable=status_var,
              wraplength=460).pack(anchor="w", pady=(6, 12))

    # ── Testperson auswählen / anlegen ────────────────────────────────────────
    person_frame = ttk.LabelFrame(main, text=" Testperson ", padding=10)
    person_frame.pack(fill="x", pady=(0, 12))

    person_row = ttk.Frame(person_frame)
    person_row.pack(fill="x")
    person_box = ttk.Combobox(person_row, textvariable=participant_var,
                              state="readonly", width=34)
    person_box.pack(side="left", fill="x", expand=True)
    new_person_button = ttk.Button(person_row, text="➕  Neu")
    new_person_button.pack(side="left", padx=(8, 0))

    ttk.Label(person_frame, textvariable=participant_info_var,
              font=("Consolas", 9), foreground="#555",
              wraplength=440, justify="left").pack(anchor="w", pady=(8, 0))

    video_check = ttk.Checkbutton(
        person_frame, variable=video_consent_var,
        text="Videoaufzeichnung dieser Sitzung (freiwillig)")
    video_check.pack(anchor="w", pady=(8, 0))

    def refresh_person_box():
        ids = registry.ids()
        person_box["values"] = [registry.label(pid) for pid in ids]
        if selected_pid[0] in ids:
            participant_var.set(registry.label(selected_pid[0]))

    def apply_selection(pid):
        selected_pid[0] = pid
        record = registry.get(pid) if pid else None
        if not record:
            participant_info_var.set("Keine Testperson ausgewählt.")
            video_consent_var.set(False)
            video_check.config(state="disabled")
            return
        allowed = record.get("consent", {}).get("video", False)
        video_check.config(state="normal" if allowed else "disabled")
        # Standard = das, was die Person bei der Aufnahme zugestimmt hat.
        # Ohne Zustimmung bleibt die Box zwangsweise aus.
        video_consent_var.set(bool(allowed))
        parq_hint = ("PAR-Q: mindestens ein „Ja“ – bitte abklären"
                     if record.get("parq_any_yes") else "PAR-Q: unauffällig")
        video_hint = ("Video: eingewilligt" if allowed
                      else "Video: NICHT eingewilligt – keine Videoaufnahme möglich")
        participant_info_var.set(
            f"{record['age_years']} J. (Gruppe {record['age_group']}) | "
            f"{record['sex']} | {record['handedness']} | "
            f"Technikaffinität {record['tech_affinity']}/5\n"
            f"{parq_hint}  |  {video_hint}"
        )

    def on_person_selected(_event=None):
        label = participant_var.get()
        for pid in registry.ids():
            if registry.label(pid) == label:
                apply_selection(pid)
                return

    def new_person_clicked():
        if recorder.active:
            messagebox.showwarning(
                "Aufzeichnung läuft",
                "Bitte zuerst die laufende Aufzeichnung stoppen.")
            return
        pid = probanden.ask_new_participant(root, registry)
        if pid:
            refresh_person_box()
            participant_var.set(registry.label(pid))
            apply_selection(pid)
            set_status(f"Testperson {pid} angelegt und ausgewählt.")

    person_box.bind("<<ComboboxSelected>>", on_person_selected)
    new_person_button.config(command=new_person_clicked)
    refresh_person_box()
    apply_selection(None)

    # ── Buttons ────────────────────────────────────────────────────────────────
    buttons = ttk.Frame(main)
    buttons.pack(fill="x")

    start_button  = ttk.Button(buttons, text="▶  Sphero starten")
    stop_button   = ttk.Button(buttons, text="■  Sphero stoppen")
    graph_button  = ttk.Button(buttons, text="📊  Live Graphen")
    camera_button = ttk.Button(buttons, text="📷  Kamera starten")
    record_button = ttk.Button(buttons, text="⏺  Aufzeichnung starten")

    start_button.grid( row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
    stop_button.grid(  row=0, column=1, sticky="ew",              pady=3)
    graph_button.grid( row=1, column=0, sticky="ew", padx=(0, 6), pady=3)
    camera_button.grid(row=1, column=1, sticky="ew",              pady=3)
    record_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 0))
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    recording_var = tk.StringVar(value="Keine Aufzeichnung aktiv")
    ttk.Label(main, textvariable=recording_var,
              font=("Consolas", 9), foreground="#aa0000").pack(anchor="w", pady=(4, 0))

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

        gw = graph_window_ref[0]
        graph_open = gw is not None and gw.win.winfo_exists()
        graph_button.config(
            text="⏹  Graphen schließen" if graph_open else "📊  Live Graphen"
        )

        if recorder.active:
            elapsed = time.time() - recorder.start_time
            record_button.config(text="⏹  Aufzeichnung stoppen", state="normal")
            # Während der Aufnahme darf die Testperson nicht gewechselt und die
            # Video-Einwilligung nicht nachträglich verändert werden.
            person_box.config(state="disabled")
            video_check.config(state="disabled")
            recording_var.set(
                f"● Aufzeichnung läuft – {elapsed:0.0f}s  "
                f"[{recorder.participant_id}"
                f"{', Video' if recorder.video_consent else ', kein Video'}]"
            )
        else:
            has_person = selected_pid[0] is not None
            record_button.config(text="⏺  Aufzeichnung starten",
                                 state="normal" if has_person else "disabled")
            person_box.config(state="readonly")
            if has_person:
                allowed = registry.get(selected_pid[0]).get("consent", {}).get("video", False)
                video_check.config(state="normal" if allowed else "disabled")
            recording_var.set(
                "Keine Aufzeichnung aktiv" if has_person
                else "Keine Aufzeichnung möglich – bitte zuerst Testperson auswählen"
            )

        root.after(200, refresh_ui)

    # ── Button-Handler ─────────────────────────────────────────────────────────
    def start_clicked():
        started = start_sphero_control()
        set_status("Suche Sphero BOLT..." if started else "Sphero-Steuerung läuft bereits.")

    def stop_clicked():
        stop_sphero_control()

    def toggle_graphs():
        gw = graph_window_ref[0]
        if gw is not None and gw.win.winfo_exists():
            gw.close()
        else:
            start_server_once()
            graph_window_ref[0] = LiveGraphWindow(root)

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

    def toggle_recording():
        if recorder.active:
            session_dir = recorder.stop()
            if session_dir:
                messagebox.showinfo(
                    "Aufzeichnung gespeichert",
                    f"Sitzung wurde gespeichert unter:\n{session_dir}"
                )
            return

        pid    = selected_pid[0]
        record = registry.get(pid) if pid else None
        if not record:
            messagebox.showwarning(
                "Keine Testperson ausgewählt",
                "Vor der Aufzeichnung muss eine Testperson ausgewählt oder über "
                "„➕ Neu“ angelegt werden."
            )
            return

        # Video nur, wenn die Person eingewilligt hat UND es für diese Sitzung
        # nicht widerrufen wurde.
        video = bool(video_consent_var.get()) and record.get("consent", {}).get("video", False)

        if not messagebox.askokcancel(
            "Aufzeichnung starten",
            f"Testperson: {pid}  ({record['age_years']} J., Gruppe {record['age_group']})\n\n"
            f"Aufgezeichnet werden: Sensordaten, Herzfrequenz, Steuerbefehle, "
            f"Körperwinkel und Abstand.\n"
            f"Videoaufzeichnung: {'JA' if video else 'NEIN'}\n\n"
            "Aufzeichnung jetzt starten?"
        ):
            return

        start_server_once()
        session_dir = recorder.start(record, video)
        if session_dir:
            set_status(f"Aufzeichnung für {pid} gestartet"
                       f"{' (mit Video)' if video else ' (ohne Video)'}.")

    def on_close():
        stop_sphero_control()
        stop_camera.set()
        gw = graph_window_ref[0]
        if gw is not None and gw.win.winfo_exists():
            gw.close()
        if recorder.active:
            recorder.stop()
        root.destroy()


    # ── Callbacks zuweisen ────────────────────────────────────────────────────
    start_button.config( command=start_clicked)
    stop_button.config(  command=stop_clicked,  state="disabled")
    graph_button.config( command=toggle_graphs)
    camera_button.config(command=toggle_camera)
    record_button.config(command=toggle_recording)
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
