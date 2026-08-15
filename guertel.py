"""
Ansteuerung des feelSpace naviBelt (Vibrationsgürtel).

Aufgabe in dieser Anwendung: Die Lenkrichtung am Körper spürbar machen.
    Handdrehung nach rechts  -> Vibration rechts   (90°)
    Handdrehung nach links   -> Vibration links   (270°)
    Rückwärtsfahrt           -> Vibration hinten  (180°)
    Geradeaus / Stillstand   -> keine Vibration

Entwurfsprinzipien – der Gürtel darf den Rest der Anwendung unter keinen
Umständen ausbremsen oder blockieren:

  1. Eigener Arbeits-Thread.
     Die Steuerung ruft nichts auf, was auf den Gürtel wartet. Der Thread holt
     sich die aktuelle Richtung selbst und schickt die Befehle in eigenem Takt.
     Selbst wenn ein Gürtelbefehl hängt, läuft der Sphero unbeirrt weiter.

  2. Kein Aufwand, solange der Gürtel nicht eingeschaltet ist.
     Ohne `start()` existiert weder Thread noch Verbindung. `pybelt` wird erst
     beim Verbindungsaufbau importiert – ist die Bibliothek gar nicht
     installiert, stört das die Anwendung nicht.

  3. Fehler bleiben lokal.
     Keine Ausnahme verlässt dieses Modul. Schlägt die Verbindung fehl oder
     bricht sie ab, wird das als Status gemeldet und der Gürtel abgeschaltet –
     die Sphero-Steuerung und die Aufzeichnung laufen unverändert weiter.

  4. Nur bei Änderung senden.
     Wie beim Sphero wird ein Befehl nur abgesetzt, wenn sich die Richtung
     wirklich ändert. Ein Dauerfeuer identischer Befehle würde die Verbindung
     unnötig belasten.

Verbindungsart: Standardmäßig wird zuerst USB versucht und erst danach
Bluetooth. Grund: Sphero und Gürtel würden sich sonst dasselbe 2,4-GHz-Band
teilen, in dem bereits die USB3-Kamera stört. Über USB ist der Gürtel davon
vollständig entkoppelt.
"""

import threading
import time

# ── Winkelkonvention des Gürtels ─────────────────────────────────────────────
# 0° = vorne am Bauch, im Uhrzeigersinn positiv (aus Sicht der tragenden Person).
ANGLE_FRONT = 0
ANGLE_RIGHT = 90
ANGLE_BACK  = 180
ANGLE_LEFT  = 270

# Zuordnung der Fahrzustände der Steuerung auf Gürtelwinkel.
# Zustände ohne Eintrag (forward, neutral) lösen keine Vibration aus.
RICHTUNG_WINKEL = {
    "right":    ANGLE_RIGHT,
    "left":     ANGLE_LEFT,
    "backward": ANGLE_BACK,
}

# "auto" = erst USB, dann Bluetooth | "usb" | "bluetooth"
VERBINDUNGSART = "auto"

# None = voreingestellte Stärke des Gürtels benutzen, sonst 0…100.
VIBRATION_INTENSITY = None


class VibrationBelt:
    """
    Nicht blockierende Ansteuerung des Gürtels.

    `richtung_quelle` ist eine Funktion ohne Argumente, die den aktuellen
    Fahrzustand als Zeichenkette liefert ("left", "right", "backward",
    "forward", "neutral"). Sie wird im Arbeits-Thread aufgerufen, damit die
    Steuerungsschleife nichts vom Gürtel weiß und nichts zusätzlich tun muss.
    """

    TAKT_S             = 0.05   # wie oft der Thread die Richtung prüft
    MIN_CMD_INTERVAL_S = 0.12   # Mindestabstand zwischen zwei Gürtelbefehlen
    MAX_FEHLER         = 5      # danach gilt die Verbindung als verloren

    def __init__(self, richtung_quelle, on_status=None, on_event=None):
        self._richtung_quelle = richtung_quelle
        self._on_status = on_status or (lambda text: None)
        self._on_event  = on_event  or (lambda name, detail: None)

        self._thread     = None
        self._stop       = threading.Event()
        self._verbunden  = threading.Event()
        self._controller = None
        self._schnittstelle = None      # Beschreibung für die Anzeige
        self._lock       = threading.Lock()
        self._status_text = "Gürtel aus"
        self._je_verbunden = False   # wurde in diesem Verbindungsversuch je CONNECTED erreicht?

    # ── Zustand ───────────────────────────────────────────────────────────────

    def ist_aktiv(self) -> bool:
        """True, solange der Arbeits-Thread läuft (auch während des Verbindens)."""
        return self._thread is not None and self._thread.is_alive()

    def ist_verbunden(self) -> bool:
        return self._verbunden.is_set()

    def status(self) -> str:
        with self._lock:
            return self._status_text

    def _set_status(self, text: str):
        with self._lock:
            self._status_text = text
        self._on_status(text)

    # ── Lebenszyklus ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Startet Verbindungsaufbau und Ansteuerung im Hintergrund."""
        if self.ist_aktiv():
            return False
        self._stop.clear()
        self._verbunden.clear()
        self._je_verbunden = False
        self._thread = threading.Thread(target=self._arbeite, daemon=True)
        self._thread.start()
        return True

    def stop(self, warten: bool = True):
        """Beendet die Ansteuerung und trennt die Verbindung."""
        self._stop.set()
        t = self._thread
        if warten and t is not None and t.is_alive() and t is not threading.current_thread():
            # Kurz warten, aber niemals unbegrenzt: Hängt der Gürtel beim
            # Trennen, darf das die Oberfläche nicht einfrieren.
            t.join(3.0)

    # ── Arbeits-Thread ────────────────────────────────────────────────────────

    def _arbeite(self):
        try:
            if not self._verbinde():
                self._verbunden.clear()
                return
            self._verbunden.set()
            self._schleife()
        except Exception as e:
            self._set_status(f"Gürtel-Fehler: {type(e).__name__}: {e}")
        finally:
            self._verbunden.clear()
            self._trenne()

    def _schleife(self):
        gesendet   = None    # zuletzt gesendeter Winkel (None = keine Vibration)
        letzter_cmd = 0.0
        fehler      = 0

        while not self._stop.is_set():
            try:
                richtung = self._richtung_quelle()
            except Exception:
                richtung = None
            ziel = RICHTUNG_WINKEL.get(richtung)

            if ziel != gesendet and (time.time() - letzter_cmd) >= self.MIN_CMD_INTERVAL_S:
                try:
                    if ziel is None:
                        self._stoppe_vibration()
                    else:
                        self._vibriere(ziel)
                    gesendet    = ziel
                    letzter_cmd = time.time()
                    fehler      = 0
                except Exception as e:
                    fehler += 1
                    if fehler >= self.MAX_FEHLER:
                        self._set_status(
                            f"Gürtel antwortet nicht ({type(e).__name__}) – abgeschaltet. "
                            "Sphero und Aufzeichnung laufen weiter.")
                        self._on_event("belt_lost", str(e)[:80])
                        return
                    # Kurz Luft lassen, bevor der nächste Versuch läuft.
                    letzter_cmd = time.time()

            self._stop.wait(self.TAKT_S)

    # ── Verbindungsaufbau ─────────────────────────────────────────────────────

    def _verbinde(self) -> bool:
        self._set_status("Gürtel: verbinde...")
        try:
            from pybelt.belt_controller import (
                BeltController, BeltControllerDelegate, BeltConnectionState, BeltMode)
        except ImportError:
            self._set_status("Gürtel: pybelt ist nicht installiert (pip install pybelt).")
            return False

        class _Delegate(BeltControllerDelegate):
            pass

        self._controller = BeltController(_Delegate())

        versuche = []
        if VERBINDUNGSART in ("auto", "usb"):
            versuche += [("USB", p) for p in self._serielle_ports()]
        if VERBINDUNGSART in ("auto", "bluetooth"):
            versuche += [("Bluetooth", None)]

        if not versuche:
            self._set_status("Gürtel: keine Schnittstelle gefunden.")
            return False

        for art, ziel in versuche:
            if self._stop.is_set():
                return False
            try:
                if art == "USB":
                    self._set_status(f"Gürtel: versuche USB {ziel}...")
                    self._controller.connect(ziel)
                else:
                    self._set_status("Gürtel: suche per Bluetooth...")
                    geraet = self._ble_suche()
                    if geraet is None:
                        continue
                    self._controller.connect(geraet)
            except Exception:
                continue

            if self._controller.get_connection_state() == BeltConnectionState.CONNECTED:
                self._schnittstelle = f"{art}" + (f" {ziel}" if ziel else "")
                # APP_MODE schaltet die Nordpfeil-Funktion ab. Die Winkelangabe
                # ist dann körperbezogen (0° = vorne am Bauch) und dreht sich
                # nicht mit der Blickrichtung – genau das brauchen wir hier.
                try:
                    self._controller.set_belt_mode(BeltMode.APP_MODE)
                except Exception:
                    pass
                self._set_status(f"Gürtel verbunden ({self._schnittstelle}).")
                self._on_event("belt_connected", self._schnittstelle)
                self._je_verbunden = True
                return True

        self._set_status("Gürtel nicht gefunden. Eingeschaltet? Kabel/Bluetooth prüfen.")
        return False

    @staticmethod
    def _serielle_ports():
        """
        Mögliche USB-Anschlüsse des Gürtels, den wahrscheinlichsten zuerst.

        Nur echte USB-Seriell-Geräte kommen in Frage. Insbesondere die
        virtuellen "Seriell über Bluetooth"-Ports, die Windows beim Pairing
        anlegt (BTHENUM), werden übersprungen: Sie zu öffnen blockiert
        ~10 Sekunden pro Port, weil Windows dabei eine klassische
        Bluetooth-Verbindung versucht, die der Gürtel nicht bedient. Der
        Gürtel spricht Bluetooth nur über BLE (siehe _ble_suche).
        """
        try:
            from serial.tools import list_ports
        except ImportError:
            return []
        try:
            ports = list(list_ports.comports())
        except Exception:
            return []

        guertel_ports, usb_ports = [], []
        for p in ports:
            text = f"{getattr(p, 'description', '')} {getattr(p, 'manufacturer', '')}".lower()
            hwid = (getattr(p, 'hwid', '') or '').upper()
            if "belt" in text or "feelspace" in text:
                guertel_ports.append(p.device)
            elif hwid.startswith("USB"):
                usb_ports.append(p.device)
        return guertel_ports + usb_ports

    def _ble_suche(self):
        try:
            import pybelt.belt_scanner
        except ImportError:
            return None
        try:
            with pybelt.belt_scanner.create() as scanner:
                gefunden = scanner.scan()
            return gefunden[0] if gefunden else None
        except Exception:
            return None

    # ── Gürtelbefehle ─────────────────────────────────────────────────────────

    def _vibriere(self, winkel: int):
        from pybelt.belt_controller import BeltOrientationType, BeltVibrationPattern
        self._controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=VIBRATION_INTENSITY,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=int(winkel) % 360,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=True,
            clear_other_channels=True,
        )

    def _stoppe_vibration(self):
        self._controller.stop_vibration(channel_index=1)

    def _trenne(self):
        ctrl, self._controller = self._controller, None
        if ctrl is not None:
            try:
                ctrl.stop_vibration(channel_index=1)
            except Exception:
                pass
            try:
                # Nicht blockierend trennen: Ein hängender Gürtel darf das
                # Schließen der Anwendung nicht aufhalten.
                ctrl.disconnect_belt(wait=False)
            except Exception:
                pass
            self._on_event("belt_disconnected", self._schnittstelle or "")
            self._schnittstelle = None

        # War in diesem Versuch nie eine Verbindung zustande gekommen, steht in
        # _status_text bereits der konkrete Grund (z.B. "nicht gefunden",
        # "pybelt ist nicht installiert") – den wollen wir hier nicht mit dem
        # nichtssagenden "Gürtel aus" überschreiben, sonst sieht der Nutzer nie,
        # woran die Verbindung gescheitert ist.
        if self._je_verbunden:
            self._set_status("Gürtel aus")
