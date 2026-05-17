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
    print(f"gX={gx:+.2f} | gY={gy:+.2f} | gZ={gz:+.2f} -> {state}")

    with lock:
        latest_data["gx"] = gx
        latest_data["gy"] = gy
        latest_data["gz"] = gz

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


def show_controller_ui():
    root = tk.Tk()
    root.title("Sphero Controller")
    root.geometry("420x260")
    root.minsize(380, 240)

    status_var = tk.StringVar(value=last_status)
    sensor_var = tk.StringVar(value="gX=0.00 | gY=0.00 | gZ=0.00")

    main = ttk.Frame(root, padding=18)
    main.pack(fill="both", expand=True)

    title = ttk.Label(main, text="Sphero Steuerung", font=("Segoe UI", 16, "bold"))
    title.pack(anchor="w")

    status = ttk.Label(main, textvariable=status_var, wraplength=360)
    status.pack(anchor="w", pady=(8, 14))

    buttons = ttk.Frame(main)
    buttons.pack(fill="x")

    start_button = ttk.Button(buttons, text="Sphero starten")
    stop_button = ttk.Button(buttons, text="Sphero stoppen", command=stop_sphero_control)
    graph_button = ttk.Button(buttons, text="Live Graphen", state="disabled")
    camera_button = ttk.Button(buttons, text="Kamera", state="disabled")

    start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=4)
    stop_button.grid(row=0, column=1, sticky="ew", pady=4)
    graph_button.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=4)
    camera_button.grid(row=1, column=1, sticky="ew", pady=4)
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    ttk.Separator(main).pack(fill="x", pady=14)

    sensor_label = ttk.Label(main, textvariable=sensor_var)
    sensor_label.pack(anchor="w")

    hint = ttk.Label(
        main,
        text="Die Buttons fuer Live Graphen und Kamera sind vorbereitet und koennen spaeter verbunden werden.",
        wraplength=360,
        foreground="#555555",
    )
    hint.pack(anchor="w", pady=(10, 0))

    def start_clicked():
        started = start_sphero_control()
        if started:
            set_status("Sphero-Steuerung startet. Suche nach Sphero BOLT...")
        else:
            set_status("Sphero-Steuerung laeuft bereits.")

    def refresh_ui():
        with lock:
            gx = latest_data["gx"]
            gy = latest_data["gy"]
            gz = latest_data["gz"]
            status_text = last_status

        sensor_var.set(f"gX={gx:+.2f} | gY={gy:+.2f} | gZ={gz:+.2f}")
        status_var.set(status_text)

        is_running = control_thread is not None and control_thread.is_alive()
        start_button.config(state="disabled" if is_running else "normal")
        stop_button.config(state="normal" if is_running else "disabled")

        root.after(250, refresh_ui)

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
