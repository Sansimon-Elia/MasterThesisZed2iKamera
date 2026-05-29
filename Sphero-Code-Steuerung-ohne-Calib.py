from flask import Flask, request
import asyncio
import threading
import time
import logging
import tkinter as tk
from tkinter import ttk
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

lock       = threading.Lock()
stop_event = threading.Event()
sphero_api = None
server_thread = None
control_thread = None
server_started = threading.Event()
last_status = "Bereit. Starte die Sphero-Steuerung per Button."

latest_data = {
    "gx":      0.0,
    "gy":      0.0,
    "gz":      0.0,
    "state":   "neutral",   # neu: aktueller Zustand
    "heading": 0,           # neu: aktuelle Fahrtrichtung
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
#MAX_SPEED = 50   # Geschwindigkeit 0-255. Erhöhen = schneller, verringern = langsamer

# ── Dynamische Geschwindigkeit über Handhöhe ──────────────────────────────────
MIN_SPEED_DYN = 30    # Mindestgeschwindigkeit wenn Hand flach (gX ≈ 0)
                      # → Erhöhe wenn Sphero zu langsam bei flacher Hand
MAX_SPEED_DYN = 120   # Maximalgeschwindigkeit wenn Hand hoch (gX ≈ -0.95)
                      # → Verringere wenn Topgeschwindigkeit zu schnell ist

TURN_SPEED_FACTOR = 0.6   # Geschwindigkeit während Kurve = calc_speed × 0.6
                           # → 0.5 = halbe Geschwindigkeit (smoother)
                           # → 0.8 = 80% Geschwindigkeit (weniger Abbremsung)
                           # → 1.0 = keine Abbremsung

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
# → gZ wird nicht mehr als harte Forward-Bedingung genutzt, weil die Hand
#   bei höherer Haltung sonst fälschlich in Neutral fallen kann.
GZ_FORWARD_THRESHOLD = -0.9
# → gX: Verringere +0.10 weiter wenn Vorwärts zu leicht auslöst
# → gX: Erhöhe +0.10 wenn Vorwärts zu schwer auszulösen ist
GX_FORWARD_MAX = +0.1

# NORMAL/UNTEN: gX steigt auf ca. +0.95
# → Verringere +0.70 wenn Normal zu früh erkannt wird
# → Erhöhe auf +0.85 wenn Normal zu schwer zu erkennen ist
GX_NEUTRAL_THRESHOLD = +0.12

MAX_TURN_ANGLE = 90   # Grad bei voller Handdrehung (gY=±1.0)
                      # → Erhöhe für schärfere Kurven (z.B. 120)
                      # → Verringere für sanftere Kurven (z.B. 45)

TURN_DEADZONE = 0.80  # Ab wann Kurve beginnt – muss gleich wie GY_RIGHT/LEFT_THRESHOLD sein
                      # → Erhöhe wenn Kurve zu früh startet
                      # → Verringere wenn Kurve zu spät startet



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

    # VORWÄRTS: gX niedrig genug = Hand zeigt nach vorne/oben.
    # gZ wird absichtlich nicht hart geprüft, weil es bei höherer Hand
    # weniger negativ werden kann und sonst fälschlich Neutral auslöst.
    if gx < GX_FORWARD_MAX:
        return "forward"

    # Alles andere = neutral (Übergangsbewegungen)
    return "neutral"


#dynamische Geschwindigkei anhand gx 
def calc_speed(gx):
    """
    gX = +0.10 (flach/neutral) → MIN_SPEED_DYN (30)
    gX =  0.00 (mittig)        → ~55
    gX = -0.50 (halb hoch)     → ~75
    gX = -0.95 (ganz hoch)     → MAX_SPEED_DYN (120)

    Erhöhe MAX_SPEED_DYN für mehr Topgeschwindigkeit.
    Erhöhe MIN_SPEED_DYN wenn Sphero bei flacher Hand zu langsam ist.
    """
    # Normierung: +0.10 (flach) bis -0.95 (hoch) → 0.0 bis 1.0
    intensity = (0.10 - gx) / (0.10 - (-0.95))
    intensity = max(0.0, min(1.0, intensity))
    return int(MIN_SPEED_DYN + intensity * (MAX_SPEED_DYN - MIN_SPEED_DYN))


# ── Hilfsfunktion für proportionale Kurve ────────────────────────────────────
def calc_turn(gy_value):
    """
    Berechnet Kurvenwinkel proportional zur Handdrehung.
    
    gY = -0.80 (Schwelle)  → intensity=0.0 → turn=0°
    gY = -0.90 (mittel)    → intensity=0.5 → turn=45°
    gY = -1.00 (Maximum)   → intensity=1.0 → turn=90°
    
    Erhöhe MAX_TURN_ANGLE für schärfere Kurven.
    """
    intensity = (abs(gy_value) - TURN_DEADZONE) / (1.0 - TURN_DEADZONE)
    intensity = max(0.0, min(1.0, intensity))   # Begrenzen auf 0-1
    return intensity * MAX_TURN_ANGLE


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
    print(f"gX={gx:+.2f} | gY={gy:+.2f} | gZ={gz:+.2f} -> {state}")

    with lock:
        latest_data["gx"] = gx
        latest_data["gy"] = gy
        latest_data["gz"] = gz
        latest_data["state"] = state

    return "OK", 200


def run_server():
    app.run(host='0.0.0.0', port=56671, debug=False)


def set_status(message):
    global last_status
    with lock:
        last_status = message


def start_server_once():
    global server_thread
    if server_started.is_set():
        return

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_started.set()


def start_sphero_control():
    global control_thread
    if control_thread and control_thread.is_alive():
        return False

    stop_event.clear()
    start_server_once()
    control_thread = threading.Thread(target=control_sphero, daemon=True)
    control_thread.start()
    return True


def stop_sphero_control():
    stop_event.set()
    set_status("Sphero-Steuerung wird gestoppt...")


# ── Verbesserte UI ────────────────────────────────────────────────────────────
def show_controller_ui():
    root = tk.Tk()
    root.title("Sphero Reha-Controller")
    root.geometry("440x320")
    root.minsize(400, 280)

    status_var  = tk.StringVar(value=last_status)
    sensor_var  = tk.StringVar(value="Keine Daten")
    state_var   = tk.StringVar(value="◯  neutral")
    heading_var = tk.StringVar(value="Heading: 0°")

    main = ttk.Frame(root, padding=18)
    main.pack(fill="both", expand=True)

    ttk.Label(main, text="Sphero Reha-Controller",
              font=("Segoe UI", 16, "bold")).pack(anchor="w")

    ttk.Label(main, textvariable=status_var,
              wraplength=380).pack(anchor="w", pady=(6, 12))

    # ── Buttons ───────────────────────────────────────────────────────────────
    buttons = ttk.Frame(main)
    buttons.pack(fill="x")

    start_button  = ttk.Button(buttons, text="▶  Sphero starten")
    stop_button   = ttk.Button(buttons, text="■  Sphero stoppen",
                               command=stop_sphero_control)
    graph_button  = ttk.Button(buttons, text="📊 Live Graphen", state="disabled")
    camera_button = ttk.Button(buttons, text="📷 Kamera",       state="disabled")

    start_button.grid( row=0, column=0, sticky="ew", padx=(0,6), pady=3)
    stop_button.grid(  row=0, column=1, sticky="ew",             pady=3)
    graph_button.grid( row=1, column=0, sticky="ew", padx=(0,6), pady=3)
    camera_button.grid(row=1, column=1, sticky="ew",             pady=3)
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    ttk.Separator(main).pack(fill="x", pady=12)

    # ── Live Anzeige ──────────────────────────────────────────────────────────
    info_frame = ttk.Frame(main)
    info_frame.pack(fill="x")

    # Zustand farbig
    state_label = tk.Label(info_frame, textvariable=state_var,
                           font=("Segoe UI", 12, "bold"),
                           fg="#007700", anchor="w")
    state_label.grid(row=0, column=0, sticky="w")

    ttk.Label(info_frame, textvariable=heading_var,
              font=("Segoe UI", 11)).grid(row=0, column=1, sticky="e", padx=(20,0))

    ttk.Label(info_frame, textvariable=sensor_var,
              font=("Consolas", 9),
              foreground="#555555").grid(row=1, column=0, columnspan=2,
                                        sticky="w", pady=(4,0))

    info_frame.columnconfigure(0, weight=1)

    # ── Farben je nach Zustand ────────────────────────────────────────────────
    STATE_COLORS = {
        "forward":  ("#007700", "▶  vorwärts"),
        "right":    ("#cc5500", "↻  rechts"),
        "left":     ("#0055cc", "↺  links"),
        "neutral":  ("#888888", "◯  neutral"),
    }

    def refresh_ui():
        with lock:
            gx      = latest_data["gx"]
            gy      = latest_data["gy"]
            gz      = latest_data["gz"]
            state   = latest_data["state"]
            heading = latest_data["heading"]
            status_text = last_status

        sensor_var.set(f"gX={gx:+.2f}  gY={gy:+.2f}  gZ={gz:+.2f}")
        status_var.set(status_text)
        heading_var.set(f"Heading: {int(heading)}°")

        color, label = STATE_COLORS.get(state, ("#888888", "◯  neutral"))
        state_var.set(label)
        state_label.config(fg=color)

        is_running = control_thread is not None and control_thread.is_alive()
        start_button.config(state="disabled" if is_running else "normal")
        stop_button.config( state="normal"   if is_running else "disabled")

        root.after(200, refresh_ui)

    def start_clicked():
        started = start_sphero_control()
        set_status("Sphero-Steuerung startet. Suche nach Sphero BOLT..." if started
                   else "Sphero-Steuerung läuft bereits.")

    def on_close():
        stop_sphero_control()
        root.destroy()

    start_button.config(command=start_clicked)
    stop_button.config(state="disabled")
    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_ui()
    root.mainloop()


# ── Sphero Steuerung ──────────────────────────────────────────────────────────
def control_sphero():
    global sphero_api
    try:
        print("[INFO] Suche Sphero BOLT...")
        set_status("Suche Sphero BOLT...")
        toy = scanner.find_toy()
        if not toy:
            print("[FEHLER] Kein Sphero gefunden!")
            set_status("Kein Sphero gefunden. Bitte Bluetooth und Sphero pruefen.")
            return

        print("[OK] Sphero verbunden!")
        set_status("Sphero verbunden. Sensor-App kann Daten an /sensorlog senden.")
    except Exception as error:
        print(f"[FEHLER] Sphero-Start fehlgeschlagen: {error}")
        set_status(f"Sphero-Start fehlgeschlagen: {error}")
        return

    with SpheroEduAPI(toy) as sphero:
        sphero_api     = sphero
        sphero_heading = 0
        last_move_time = time.time()
        is_stopped     = True

        sphero.set_heading(0)
        sphero.set_main_led(Color(r=255, g=255, b=255))  # Weiß = bereit
        print("\n[OK] Bereit! Steuerung:")
        print("   Hand nach unten  = Neutral/Stopp")
        print("   Hand flach vorne = Vorwaerts")
        print("   Uhr nach rechts  = Rechts")
        print("   Uhr nach links   = Links\n")

        while not stop_event.is_set():
            with lock:
                gx = latest_data["gx"]
                gy = latest_data["gy"]
                gz = latest_data["gz"]

            state = get_state(gx, gy, gz)

            if state == "right":
                turn  = calc_turn(gy)
                speed = int(calc_speed(gx) * TURN_SPEED_FACTOR)  # 60% der aktuellen Geschwindigkeit
                sphero_heading = (sphero_heading + turn) % 360
                sphero.roll(int(sphero_heading), speed, 0.1)
                sphero.set_main_led(Color(r=255, g=100, b=0))
                last_move_time = time.time()
                is_stopped     = False
                with lock:
                    latest_data["heading"] = sphero_heading


            elif state == "left":
                turn  = calc_turn(gy)
                speed = int(calc_speed(gx) * TURN_SPEED_FACTOR)  # 60% der aktuellen Geschwindigkeit
                sphero_heading = (sphero_heading - turn) % 360
                sphero.roll(int(sphero_heading), speed, 0.1)
                sphero.set_main_led(Color(r=0, g=200, b=255))
                last_move_time = time.time()
                is_stopped     = False
                with lock:
                    latest_data["heading"] = sphero_heading

            elif state == "forward":
                speed = calc_speed(gx)   # volle dynamische Geschwindigkeit
                sphero.roll(int(sphero_heading), speed, 0.1)
                sphero.set_main_led(Color(r=0, g=255, b=0))
                last_move_time = time.time()
                is_stopped     = False

            elif state == "neutral":
                # Auto-Stopp nach STOP_TIME Sekunden Inaktivität
                if not is_stopped and time.time() - last_move_time > STOP_TIME:
                    sphero.stop_roll(int(sphero_heading))
                    sphero.set_main_led(Color(r=255, g=0, b=0))  # Rot
                    is_stopped = True
                    print("[STOPP] Auto-Stopp")

            time.sleep(0.05)

        # Sauber beenden
        print("\n[INFO] Trenne Sphero...")
        try:
            sphero.stop_roll(0)
            sphero.set_main_led(Color(r=0, g=0, b=0))
            time.sleep(0.5)
        except Exception:
            pass
        print("[OK] Getrennt.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        show_controller_ui()
    except KeyboardInterrupt:
        print("\n[STOPP] Beende...")
        stop_event.set()
