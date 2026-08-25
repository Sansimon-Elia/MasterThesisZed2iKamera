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
# ToyUtil ist die Ebene UNTER SpheroEduAPI: ein Fahrbefehl, ein BLE-Paket.
# Gebraucht für _SpheroDrive (siehe dort) – SpheroEduAPI.roll() schickt pro
# Aufruf zwei Pakete und hält den Ball dazwischen an.
from spherov2.utils import ToyUtil

# Probandenverwaltung (pseudonymisierte Stammdaten + Eingabemaske)
import probanden
# Vibrationsgürtel (feelSpace naviBelt) – optional, wird nur bei Bedarf genutzt
import guertel

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

# ── Ausrichten der Nullrichtung ──────────────────────────────────────────────
# Beim Ausrichten schickt die Oberfläche selbst Befehle an den Sphero. Die
# Steuerungsschleife muss dafür stillstehen, sonst funken beide gleichzeitig auf
# dieselbe BLE-Verbindung und der Ball zuckt zwischen Ausricht- und Fahrbefehl.
#
# Übergabe in zwei Schritten, damit kein Befehl der Schleife mehr unterwegs ist,
# wenn die Oberfläche zu senden beginnt:
#   aim_request  – von der Oberfläche gesetzt: "bitte anhalten und abgeben"
#   aim_active   – von der Schleife gesetzt: "ich stehe, der Sphero gehört dir"
aim_request       = threading.Event()
aim_active        = threading.Event()
# Nach dem Ausrichten ist die neue Nullrichtung per Definition 0°; die Schleife
# muss ihren mitgeführten Kurs darauf zurücksetzen.
aim_heading_reset = threading.Event()


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
    # Gemessene Frequenz der Steuerungsschleife. 0.0 = noch nicht gemessen.
    # Steht hier, damit Anzeigen sie lesen können, ohne in die Steuerung
    # hineinzugreifen. Bis zur Umstellung auf die zeitbasierte Drehrate hing die
    # gefahrene Drehgeschwindigkeit unmittelbar an dieser Größe – sie war aber
    # nirgends sichtbar, weshalb die Ursache des schwankenden Kurvenverhaltens
    # lange nicht zu erkennen war.
    "loop_hz": 0.0,
    # Zuletzt empfangene Herzfrequenz. 0.0 = die Uhr hat noch keinen Puls
    # geliefert (siehe valid_hr). Liegt hier und nicht nur im Graph-Puffer,
    # damit die Ruhepulsmessung sie ohne Umweg über die Anzeige lesen kann.
    "heart_rate": 0.0,
}

# Live-Graph-Daten (Zeitachse = t_abs_s der gemeinsamen Referenzuhr)
MAX_POINTS        = 300
intensity_values  = deque(maxlen=MAX_POINTS)
heart_rate_values = deque(maxlen=MAX_POINTS)
graph_time_values = deque(maxlen=MAX_POINTS)

# ── Sphero-Konfiguration ──────────────────────────────────────────────────────
# Tempobereich der Vorwärtsfahrt. Der Sphero nimmt 0…255 entgegen; die hier
# eingetragene Obergrenze ist eine bewusst gewählte Reserve, keine Gerätegrenze.
# Beide Werte sind zur Laufzeit über "Fahrverhalten justieren" verstellbar,
# damit dieselbe Anwendung von vorsichtigen und von sportlichen Testpersonen
# gefahren werden kann, ohne dass am Code etwas geändert werden muss.
MIN_SPEED_DYN        = 30
MAX_SPEED_DYN        = 120
TURN_SPEED_FACTOR    = 0.5

# Handneigung, bei der das Höchsttempo erreicht ist (Hand weit nach unten
# gekippt). Zusammen mit GX_FORWARD_MAX spannt dieser Wert die Tempo-Rampe auf:
# bei GX_FORWARD_MAX fährt der Sphero MIN_SPEED_DYN, bei GX_FULL_SPEED
# MAX_SPEED_DYN, dazwischen linear (siehe calc_speed).
#
# Warum einstellbar: Der nutzbare Neigungsbereich ist die eigentliche
# Bewegungsaufgabe. Wer den Unterarm nur eingeschränkt senken kann, erreicht
# -0.95 nie und bliebe dauerhaft im unteren Tempodrittel. Wird der Wert auf die
# tatsächlich erreichte Neigung gesetzt, steht der volle Tempobereich wieder
# über den ganzen individuellen Bewegungsumfang zur Verfügung.
GX_FULL_SPEED        = -0.95

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
GY_RIGHT_THRESHOLD   = -0.50
GY_LEFT_THRESHOLD    = +0.50
GX_FORWARD_MAX       = +0.1
GX_NEUTRAL_THRESHOLD = +0.12
# Maximaler Drehwinkel je Lenkschritt, getrennt je Seite.
# Getrennt, weil die Schwellen unterschiedlich sind: Rechts wird ab 0.80
# gedreht, links schon ab 0.72. Der Winkel wird zwischen Schwelle und Vollaus-
# schlag (1.0) hochskaliert – bei der niedrigeren linken Schwelle ist dieser
# Bereich breiter (0.28 statt 0.20), sodass dieselbe Handdrehung links einen
# deutlich größeren Winkel ergäbe. Ein kleinerer Maximalwinkel links gleicht
# das wieder aus.
#
# "Je Lenkschritt" heißt seit der Umstellung auf die zeitbasierte Drehrate:
# je LENK_BEZUGSTAKT_S Sekunden (siehe dort), NICHT mehr je tatsächlichem
# Schleifendurchlauf.
MAX_TURN_ANGLE_RIGHT = 61
MAX_TURN_ANGLE_LEFT  = 62

# ── Zeitbasierte Drehrate ────────────────────────────────────────────────────
# Taktzeit, auf die sich MAX_TURN_ANGLE_* bezieht.
#
# Vorher wurde der Winkel einmal PRO SCHLEIFENDURCHLAUF aufaddiert. Die
# tatsächlich gefahrene Drehrate war damit MAX_TURN_ANGLE mal Schleifenfrequenz –
# und die hängt an Funkqualität und Rechenlast. Gemessen schwankte sie zwischen
# rund 2,5 und 7 Durchläufen je Sekunde, die Drehrate bei gleicher Handhaltung
# also um mehr als das Doppelte. Genau das war beim Testen als "erst dreht er
# langsam, dann plötzlich viel mehr" aufgefallen, und zwar bei allen drei
# Testpersonen.
#
# Jetzt wird mit der wirklich vergangenen Zeit skaliert:
#
#     Drehung = MAX_TURN_ANGLE * (dt / LENK_BEZUGSTAKT_S)
#
# Läuft die Schleife genau in diesem Takt, ist das Verhalten identisch zu
# vorher. Läuft sie schneller oder langsamer, bleibt die Drehrate gleich.
# Damit ist die Lenkung erstens für die Testperson vorhersagbar und zweitens
# zwischen Sitzungen und Personen überhaupt erst vergleichbar – bisher steckte
# die Tagesform des Laptops mit in der Messgröße.
#
# 0.22 s ist die gemessene mittlere Durchlaufzeit der ALTEN Schleife (zwei
# quittierte BLE-Pakete je Durchlauf plus deren Sicherheitsabstände). Der Wert
# ist deshalb so gewählt, dass die eingefahrene Abstimmung erhalten bleibt.
# Kleiner = dreht insgesamt schneller, größer = gemächlicher.
LENK_BEZUGSTAKT_S = 0.22

# Obergrenze des Zeitfaktors dt / LENK_BEZUGSTAKT_S. Hängt die Schleife einmal
# (Funkaussetzer, kurz blockierte Kamera), würde die "nachzuholende" Drehung
# sonst als Sprung herauskommen – genau die ruckartige Bewegung, die abgestellt
# werden soll. 2.0 heißt: höchstens die doppelte Drehung eines normalen Takts.
LENK_MAX_ZEITFAKTOR = 2.0

# Obergrenze der Kursänderung während EINER durchgehenden Handdrehung, in Grad.
# 0 schaltet die Begrenzung ab.
#
# VOREINSTELLUNG 0 = AUS. Das war zwischenzeitlich anders, und warum es wieder
# abgeschaltet wurde, ist der eigentlich lehrreiche Teil:
#
# Das Budget war als Gegenmittel gegen das Überdrehen eingeführt worden. Wer die
# Hand weit gedreht hielt und wartete, bis sichtbar etwas geschah, hatte den Ball
# gleich weit über das Ziel hinausgedreht – bei allen Testpersonen zu beobachten.
# Die Ursache dafür war aber gar nicht die Ratensteuerung selbst, sondern die
# schwankende Drehrate (siehe LENK_BEZUGSTAKT_S): Dieselbe Handhaltung drehte je
# nach Rechenlast unterschiedlich schnell, also konnte niemand die nötige Dauer
# abschätzen. Seit die Drehrate zeitbasiert und damit vorhersagbar ist, tritt das
# Überdrehen nicht mehr auf – das Budget behandelte ein Symptom, dessen Ursache
# inzwischen behoben ist.
#
# Es hatte dafür einen eigenen Preis: Bei einer Dauerkurve um ein Objekt hörte
# der Ball nach der eingestellten Gradzahl auf zu drehen und fuhr geradeaus
# weiter, obwohl die Hand noch gedreht war. Die Ratensteuerung verhielt sich in
# dem Moment wie eine Positionssteuerung, und das ist genau die Eigenschaft, die
# man bei der Wahl dieser Steuerungsart NICHT haben will.
#
# Der Regler bleibt erhalten: Für vorsichtige Testpersonen kann eine Begrenzung
# weiterhin sinnvoll sein, und als Vergleichsgröße zwischen den Altersgruppen ist
# sie interessant. Ist ein Budget gesetzt und aufgebraucht, wird die Kurvenfarbe
# dunkler – so ist der Zustand ohne Erklärung erkennbar.
LENK_MAX_DREHUNG = 0

# ── Art der Kursführung ───────────────────────────────────────────────────────
# "rate"     – Ratensteuerung (bisheriges und voreingestelltes Verhalten):
#              Die Handdrehung bestimmt, wie SCHNELL sich der Kurs ändert. Der
#              Winkel aus MAX_TURN_ANGLE_* wird pro Schleifendurchlauf
#              aufaddiert; gehaltene Drehung heißt also dauerndes Weiterdrehen.
#              Beliebig große Kursänderungen sind möglich, dafür muss man die
#              Hand im richtigen Moment zurücknehmen – Überschwingen ist der
#              typische Fehler.
#
# "position" – Positionssteuerung: Die Handdrehung bestimmt, WIE WEIT der Kurs
#              vom Bezugskurs abweicht. Halbe Drehung heißt dauerhaft halber
#              Versatz, Hand zurück heißt Kurs zurück. Der Zusammenhang ist
#              direkt und ohne Zeitverlauf, was das Zielen erleichtert.
#
#              Damit trotz begrenztem Versatz beliebige Kurse erreichbar
#              bleiben, wird der Bezugskurs bei jeder Geradeausfahrt neu
#              gesetzt: Eine Drehung wird "eingerastet", sobald die Hand in die
#              Neutrallage zurückkehrt, und kann von dort erneut beginnen.
#              Ohne dieses Nachführen wäre der Sphero auf POS_MAX_OFFSET_*
#              um die Nullrichtung eingesperrt und ließe sich nicht zurückholen.
#
# In der Literatur zu assistiven und teleoperierten Steuerungen ist die
# Positionssteuerung für Ungeübte meist leichter zu dosieren und erzeugt
# weniger Überschwingen, während die Ratensteuerung bei großen, fortlaufenden
# Kursänderungen im Vorteil ist. Welche Art besser passt, ist damit selbst eine
# sinnvolle Vergleichsgröße zwischen den Altersgruppen.
STEUERMODUS_RATE     = "rate"
STEUERMODUS_POSITION = "position"
STEUERMODUS          = STEUERMODUS_RATE

# Nur für "position": Kursversatz bei voller Handdrehung, je Seite.
POS_MAX_OFFSET_LEFT  = 60
POS_MAX_OFFSET_RIGHT = 60

# ── Feinfühligkeit der Lenkung ────────────────────────────────────────────────
# Exponent der Lenkkennlinie. 1.0 = linear (bisheriges Verhalten).
#
# Warum das nötig wurde: Die Drehrate reicht linear von 0 bis rund 600 °/s.
# Eine Kurve um ein Objekt braucht aber nur etwa 30 bis 90 °/s. Gemessen an der
# nutzbaren Handdrehung (gy 0,60 bis 1,00) liegt dieser ganze Bereich zwischen
# gy 0,62 und 0,66 – also in den ersten 15 % des Wegs. Die übrigen 85 % sind
# Drehen auf der Stelle. Kurvenfahren hieß deshalb, eine Handstellung auf wenige
# Hundertstel genau zu halten.
#
# Mit einem Exponenten > 1 wird der untere Bereich gespreizt: Der Anteil geht
# als anteil**LENK_EXPO ein, kleine Handdrehungen ergeben also deutlich kleinere
# Drehraten, während der volle Ausschlag unverändert die Höchstrate liefert.
# Dasselbe Mittel benutzen Fernsteuerungen im Modellbau ("Expo").
#
# Anhaltswerte: 2.0 verdoppelt die Feinfühligkeit im unteren Drittel spürbar,
# 3.0 macht Kurven sehr gut dosierbar, kostet aber Reaktionsschärfe beim
# schnellen Wenden.
#
# In den Testfahrten hat sich 1.0 zusammen mit voller Tempo-Kopplung als gut
# dosierbar erwiesen und ist deshalb die Vorgabe geblieben.
LENK_EXPO = 1.0

# Kopplung der Drehrate an das Fahrtempo, 0.0 = aus (bisheriges Verhalten).
#
# Bisher ist die Drehrate unabhängig vom Tempo. Wer langsamer fährt, dreht sich
# deshalb enger, wer schneller fährt, macht einen weiten Bogen – und um ein
# Objekt herumzukommen, muss man abbremsen. Genau das ist beim Testen
# aufgefallen.
#
# Bei voller Kopplung wird die Drehrate proportional zum Tempo. Eine gehaltene
# Handstellung ergibt dann einen Kreis mit FESTEM RADIUS, unabhängig davon, wie
# schnell gefahren wird – das Verhalten eines Lenkrads. Um ein Objekt zu fahren
# genügt es, die Hand ruhig zu halten; Tempo und Kurve passen von selbst
# zusammen.
#
# Zwischenwerte mischen beides. 0.7 lässt noch etwas Drehen im Stand zu, was
# beim Rangieren auf engem Raum hilft.
#
# VOREINSTELLUNG 1.0: In den Testfahrten war die volle Kopplung die Einstellung,
# mit der sich Kurven am leichtesten treffen ließen – Tempo und Kurvenradius
# passen dann von selbst zusammen, statt dass zum Kurvenfahren abgebremst
# werden muss.
LENK_TEMPO_KOPPLUNG = 1.0

# ── Glättung der Handhaltung ─────────────────────────────────────────────────
# Zeitkonstante eines Tiefpasses erster Ordnung auf gx und gy. 0 schaltet die
# Glättung ab und entspricht dem Verhalten vor dieser Änderung.
#
# Warum überhaupt: Die Uhr liefert die Schwerkraftrichtung roh. Darin steckt
# neben der gewollten Handhaltung auch alles Unwillkürliche – Halte- und
# Ruhetremor, das Zucken beim Anspannen, Erschütterungen beim Gehen. Liegt die
# Hand nahe einer Kippschwelle, springt der Fahrzustand dadurch mehrmals pro
# Sekunde hin und her, obwohl die Person die Hand ruhig zu halten glaubt.
#
# Für die Studie ist das besonders heikel: Halte- und Ruhetremor nehmen mit dem
# Alter zu. Ungefiltert misst man in den älteren Gruppen also teilweise die
# Empfindlichkeit der Schwellenlogik statt der Steuerleistung – genau in dem
# Vergleich, um den es geht.
#
# VOREINSTELLUNG 0.15 s: In den Testfahrten hat sich gezeigt, dass ohne
# Glättung vor allem die Kurven unruhig werden. 0.15 s dämpft das Zittern
# spürbar, ohne dass die Steuerung merklich hinterherhinkt, und ist deshalb der
# neue Ausgangswert.
#
# Als Anhaltspunkt für das Einstellen: 0,15 s bedeutet, dass eine sprunghafte
# Änderung der Handhaltung nach etwa einer Zeitkonstante zu 63 % übernommen
# ist, nach drei zu 95 %. Bei rund 100 ms Zykluszeit bleibt die Steuerung damit
# deutlich schneller als die menschliche Reaktion, während Zittern im Bereich
# mehrerer Hertz stark gedämpft wird (gemessen: auf 22 % der Schwankungsbreite).
GLAETTUNG_TAU_S = 0.0

# ── Hysterese der Kippschwellen ──────────────────────────────────────────────
# Zusatzweg, den die Hand zurücklegen muss, um einen Dreh- oder Fahrzustand
# wieder zu VERLASSEN. Einschalten geschieht weiterhin genau an der Schwelle.
#
# Ohne diesen Zusatz gibt es je Seite nur einen einzigen Umschaltpunkt. Wer
# knapp daran zittert, löst in schneller Folge links/neutral/links aus; der
# Sphero ruckt, und in den Daten entstehen Dutzende Zustandswechsel, die keine
# Absicht abbilden. Mit getrennter Ein- und Ausstiegsschwelle entsteht ein
# Halteband (Schmitt-Trigger) – dieselbe Lösung wie in jedem Thermostat.
#
# VOREINSTELLUNG: 0,12 für die Drehung, 0 fürs Fahren – in dieser Kombination
# ließen sich die Kurven in den Testfahrten am besten halten. Zusammen mit einer
# Glättung von 0,15 s sanken die Zustandswechsel bei ruhig an der Schwelle
# gehaltener Hand im Test von 121 auf 2 in zehn Sekunden.
#
# Das Halteband beim Fahren bleibt bei 0: Ein Anheben der Hand soll den Ball
# ohne Zusatzweg anhalten – das ist die Notbremse der Testperson.
GY_HYSTERESE = 0.12
GX_HYSTERESE = 0.0

# ── Reaktionsgeschwindigkeit der Steuerung ────────────────────────────────────
# ROLL_COMMAND_DURATION + CONTROL_LOOP_SLEEP ist der ANGESTREBTE Takt eines
# Schleifendurchlaufs. Die Schleife misst, wie lange ihre Arbeit gedauert hat,
# und schläft nur den Rest ab; braucht ein Funkbefehl ausnahmsweise länger,
# entfällt die Pause ganz.
#
# Früher stand hier etwas anderes: sphero.roll() schlief ROLL_COMMAND_DURATION
# selbst ab UND schickte danach von sich aus einen Stopp-Befehl. Ein Durchlauf
# kostete damit zwei quittierte BLE-Pakete samt der 75 ms Sicherheitsabstand,
# die spherov2 nach jedem Paket einlegt – gemessen 220 bis 330 ms statt der
# erhofften 100 ms. Der Ball fuhr außerdem 60 ms und stand dann wieder still,
# was das Ruckeln erzeugt hat.
#
# Seit _SpheroDrive geht je Durchlauf höchstens EIN Paket raus, und nur wenn
# sich Kurs oder Tempo wirklich geändert haben. Der Takt unten ist deshalb
# tatsächlich erreichbar.
#
# ERSTE MASSNAHME, falls die Verbindung instabil wird: hier zurück auf 0.10 und
# 0.05 und SPHERO_CMD_INTERVAL_S wieder auf 0.075. Das kostet
# Reaktionsschnelligkeit, aber nichts an der Funktion.

# Sicherheitsabstand zwischen zwei BLE-Paketen. spherov2 legt hier für den BOLT
# fest 0.075 s an (types.py, ToyType.cmd_safe_interval) und schläft diese Zeit
# im Sendethread nach JEDEM Paket ab – bei zwei Paketen je Durchlauf waren das
# allein 150 ms Grundlast. Da jetzt deutlich weniger Pakete anfallen, ist ein
# kürzerer Abstand vertretbar und macht die Steuerung spürbar direkter.
# Wird beim Verbinden auf die Toy-Instanz gesetzt (siehe control_sphero).
SPHERO_CMD_INTERVAL_S = 0.04

# ── Richtungspfeil auf der LED-Matrix ────────────────────────────────────────
#
# Rueckmeldung aus den Testfahrten P05-P07: Man sieht dem leuchtenden Ball
# nicht an, wo bei ihm "vorne" ist. Wer die Hand hebt, weiss deshalb nicht, in
# welche Richtung er gleich losfaehrt - und korrigiert erst, nachdem der Ball
# schon in die falsche Richtung gerollt ist.
#
# Der BOLT hat eine 8x8-LED-Matrix, die fest auf dem inneren Chassis sitzt.
# Dreht die Steuerung den Kurs, dreht sich das Chassis und damit die Matrix
# mit. Ein Pfeil, der auf der Matrix immer nach "oben" zeigt, zeigt am Ball
# also immer in die aktuelle Fahrtrichtung, ohne dass ein einziges zusaetzliches
# Paket noetig waere, wenn sich der Kurs aendert.
MATRIX_PFEIL_AN = True

# Der Pfeil traegt die Zustandsfarbe selbst, der Rest der Matrix bleibt
# dunkel. Frueher wurde die GANZE Matrix in der Zustandsfarbe gefuellt und der
# Pfeil weiss darueber gelegt - das kostete bei jedem Zustandswechsel ein
# Fuellpaket zusaetzlich, und weil das Fuellen den Pfeil jedes Mal loeschte,
# war er nach jedem Wechsel kurz gar nicht da.
#
# Links und rechts sind bewusst BLAU und GELB und nicht rot/gruen: Die
# Rot-Gruen-Schwaeche ist die mit Abstand haeufigste Farbsehschwaeche, Blau
# gegen Gelb dagegen fuer praktisch jeden unterscheidbar. Bei der Richtung
# darf es keinen Zweifel geben - Fahren und Halt sind zusaetzlich ueber die
# Bewegung erkennbar, links und rechts nicht.
MATRIX_PFEIL_FARBEN = {
    "forward":   (0, 255, 0),      # gruen  - geradeaus
    "right":     (255, 210, 0),    # gelb   - Rechtskurve
    "left":      (0, 90, 255),     # blau   - Linkskurve
    "stop":      (255, 0, 0),      # rot    - steht
    "backward":  (160, 0, 255),    # violett- Rueckwaertsfahrt
    "bereit":    (255, 255, 255),  # weiss  - verbunden, wartet
}

# Vierteldrehungen im Uhrzeigersinn, mit denen der Pfeil auf die tatsaechliche
# Einbaulage der Matrix gedreht wird.
#
# 180 Grad ist kein Vorschlagswert, sondern am Ball ausgemessen: Bei 0 Grad
# zeigte der Pfeil nach HINTEN. Die Zeile y=0 der Matrix liegt beim BOLT also
# auf der Rueckseite, nicht in Fahrtrichtung.
#
# Nachpruefen, falls je ein anderer Ball verwendet wird: Ball hinlegen, kurz
# vorwaerts fahren lassen und schauen, wohin der Pfeil zeigt. Umstellen laesst
# sich das zur Laufzeit im Fenster "Fahrverhalten", Abschnitt 7.
MATRIX_PFEIL_DREHUNG = 180

# Wird gesetzt, wenn die Oberflaeche Pfeilfarbe, -drehung oder -zustand
# geaendert hat. Die Steuerungsschleife baut das Bild dann im naechsten
# Durchlauf neu auf. Ueber ein Signal und nicht durch direktes Zeichnen aus
# der Oberflaeche heraus, weil auf der BLE-Verbindung immer nur EIN Thread
# senden darf - sonst zuckt der Ball zwischen zwei Befehlsstroemen.
pfeil_neu_zeichnen = threading.Event()

# ── Fahrbefehle nur bei Änderung senden ──────────────────────────────────────
# Ein wiederholtes "fahre Kurs 90 mit Tempo 60" ändert am Ball nichts, kostet
# aber ein Paket samt Antwort und Sicherheitsabstand. Bei Geradeausfahrt mit
# ruhiger Hand ist das jeder einzelne Durchlauf. Unterhalb dieser Schwellen
# gilt ein Befehl als unverändert und wird übersprungen.
#
# WICHTIG: Das bremst nichts aus. Sobald sich die Handhaltung ändert – tiefer
# gesenkt heißt anderes Tempo, gedreht heißt anderer Kurs – liegt die Änderung
# über der Schwelle und das Paket geht sofort raus. Übersprungen wird nur, was
# ohnehin keine Wirkung hätte. Der freie Funkkanal kommt dann genau den
# Befehlen zugute, die etwas ändern.
DRIVE_MIN_HEADING_DELTA = 2      # Grad
DRIVE_MIN_SPEED_DELTA   = 3      # Sphero-Tempoeinheiten

# Spätestens nach dieser Zeit wird derselbe Fahrbefehl trotzdem wiederholt.
# Absicherung gegen ein einzelnes verlorenes Paket: Ginge ausgerechnet der
# letzte Befehl vor einer langen ruhigen Geradeausfahrt verloren, führe der Ball
# sonst bis zur nächsten Handbewegung mit dem alten Kurs weiter.
DRIVE_KEEPALIVE_S = 0.5

# ── Tragearm der Apple Watch ─────────────────────────────────────────────────
# Wird die Uhr auf den anderen Arm gewechselt und dabei gedreht (Krone zeigt
# dann in die andere Richtung entlang des Unterarms), liegt das Gerät um 180°
# um die Displayachse verdreht am Handgelenk. In den Rohdaten kehren sich
# dadurch zwei der drei Schwerkraftachsen im Vorzeichen um:
#
#     gx -> -gx     (Hand oben/unten vertauscht: Stopp und Fahren verwechselt)
#     gy -> -gy     (Handdrehung vertauscht: links und rechts verwechselt)
#     gz ->  gz     (Displaynormale bleibt, wird von der Steuerung nicht genutzt)
#
# Genau dieses Bild wurde beim Tragen am rechten Arm beobachtet. Die Korrektur
# rechnet die Werte auf die Bezugslage "linker Arm" um, sodass Steuerung,
# Schwellen und Aufzeichnung für beide Arme identisch bleiben und die Daten
# verschiedener Testpersonen ohne Umrechnung vergleichbar sind.
#
# Die Transformation ist eine reine Vorzeichenspiegelung und damit verlustfrei
# umkehrbar: Aus den aufgezeichneten Werten lassen sich die Rohwerte jederzeit
# zurückrechnen, wenn der in metadata.json vermerkte Tragearm bekannt ist.
WATCH_ARM_LEFT  = "links"
WATCH_ARM_RIGHT = "rechts"
watch_arm = WATCH_ARM_LEFT        # zur Laufzeit über die Oberfläche umschaltbar

ROLL_COMMAND_DURATION = 0.06   # Sekunden – Hauptanteil des angestrebten Takts
CONTROL_LOOP_SLEEP    = 0.04   # Sekunden – Restanteil des angestrebten Takts
BACKWARD_DURATION = 2.0   # Sekunden Rückwärtsfahrt pro Double Tap
BACKWARD_SPEED    = 80    # Geschwindigkeit während der Rückwärtsfahrt
DATA_TIMEOUT      = 1.0   # Sekunden ohne Sensor-POST → Sphero gilt als "keine Daten", bleibt stehen

# ── Fahrprofile ───────────────────────────────────────────────────────────────
# Reproduzierbare Ausgangspunkte für eine Sitzung, damit nicht jede Testperson
# mit zufällig stehengebliebenen Reglern startet. "Standard" entspricht exakt
# den oben eingetragenen Werten.
#
# Die Kippschwellen GY_*_THRESHOLD und GX_FULL_SPEED bleiben unangetastet: Sie
# sind die Anpassung an den Bewegungsumfang der jeweiligen Person und damit
# unabhängig davon, ob jemand langsam oder sportlich fahren möchte. Ein
# Profilwechsel darf eine einmal eingestellte Kalibrierung nicht überschreiben.
#
# Alles Übrige setzen die Profile dagegen vollständig – auch Steuerungsart,
# Glättung und Hysterese. Nur so ist "Standard" ein verlässlicher Rückweg zum
# bekannten Ausgangsverhalten: Wer sich verstellt hat, kommt mit einem Klick
# exakt dorthin zurück, statt einzelne Regler von Hand zurücksuchen zu müssen.
#
# "Standard" bildet den über die Testfahrten eingefahrenen Stand 1:1 ab und ist
# damit die Bezugsgröße, gegen die alles Neue verglichen wird. Der Inhalt hat
# sich mit den Testfahrten geändert: Glättung, Halteband der Drehung und volle
# Tempo-Kopplung gehören inzwischen dazu, weil sich damit die Kurven deutlich
# besser dosieren ließen.
#
# Die Zahlen sind aus Testfahrten abgeleitet und keine normierten Stufen. Für
# die Auswertung zählt nicht der Profilname, sondern die tatsächlich gefahrene
# Einstellung: Sie steht in metadata.json ("config"), jede Änderung während der
# Fahrt zusätzlich in events.csv.
FAHRPROFILE = {
    "Sanft": {
        "MIN_SPEED_DYN": 20, "MAX_SPEED_DYN": 70,
        "TURN_SPEED_FACTOR": 0.5,
        "MAX_TURN_ANGLE_LEFT": 40, "MAX_TURN_ANGLE_RIGHT": 40,
        "ROLL_COMMAND_DURATION": 0.08, "STOP_TIME": 0.8,
        "BACKWARD_SPEED": 50,
        "STEUERMODUS": "rate",
        "POS_MAX_OFFSET_LEFT": 45, "POS_MAX_OFFSET_RIGHT": 45,
        "GLAETTUNG_TAU_S": 0.20, "GY_HYSTERESE": 0.06, "GX_HYSTERESE": 0.04,
        "LENK_EXPO": 2.5, "LENK_TEMPO_KOPPLUNG": 0.7,
        # Auch hier kein Budget: Es hat sich beim Fahren als hinderlicher
        # erwiesen als hilfreich (Begründung bei LENK_MAX_DREHUNG). Wer es
        # ausprobieren will, stellt es am Regler ein.
        "LENK_BEZUGSTAKT_S": 0.22, "LENK_MAX_DREHUNG": 0,
    },
    # Der in den Testfahrten mit P05-P07 eingefahrene Stand.
    #
    # Kurven-Tempo 0.5 statt 0.8: Die Rueckmeldung aus allen drei Testfahrten
    # war, dass er die Kurve schlecht nimmt. Ursache war nicht zu langsames
    # Drehen, sondern zu hohes Tempo IN der Kurve - er fuhr einen weiten Bogen,
    # wo ein enger gebraucht wurde. Der Ball ist damit in Kurven sichtbar
    # langsamer; das ist gewollt und nicht mit einer traegeren Steuerung zu
    # verwechseln.
    #
    # Glaettung 0.0 statt 0.15: Die 0.15 s waren als Zitterausgleich gedacht,
    # haben sich aber als spuerbare Verzoegerung zwischen Hand und Ball
    # bemerkbar gemacht. Gegen Zittern bleiben die beiden Haltebänder.
    #
    # Kippschwellen bei 0.50 (statt 0.60/0.72): Die Kurve beginnt frueher, das
    # Handgelenk muss weniger weit drehen. In den Testfahrten war die weite
    # Drehung schmerzhaft - bei der Zielgruppe ist das ein Ausschlusskriterium,
    # kein Komfortthema. Die Schwellen sind bewusst symmetrisch, obwohl Aus-
    # und Einwaertsdrehung des Unterarms es nicht sind; wer sie je Person
    # anpasst, verstellt sie im Fahrverhalten-Fenster und sichert sie dort.
    "Standard": {
        "MIN_SPEED_DYN": 30, "MAX_SPEED_DYN": 120,
        "TURN_SPEED_FACTOR": 0.5,
        "MAX_TURN_ANGLE_LEFT": 62, "MAX_TURN_ANGLE_RIGHT": 61,
        "ROLL_COMMAND_DURATION": 0.06, "STOP_TIME": 0.6,
        "BACKWARD_SPEED": 80,
        "STEUERMODUS": "rate",
        "POS_MAX_OFFSET_LEFT": 60, "POS_MAX_OFFSET_RIGHT": 60,
        "GLAETTUNG_TAU_S": 0.0, "GY_HYSTERESE": 0.12, "GX_HYSTERESE": 0.0,
        "LENK_EXPO": 1.0, "LENK_TEMPO_KOPPLUNG": 1.0,
        "LENK_BEZUGSTAKT_S": 0.22, "LENK_MAX_DREHUNG": 0,
    },
    "Sportlich": {
        "MIN_SPEED_DYN": 45, "MAX_SPEED_DYN": 200,
        "TURN_SPEED_FACTOR": 0.85,
        "MAX_TURN_ANGLE_LEFT": 75, "MAX_TURN_ANGLE_RIGHT": 75,
        "ROLL_COMMAND_DURATION": 0.05, "STOP_TIME": 0.4,
        "BACKWARD_SPEED": 110,
        "STEUERMODUS": "rate",
        "POS_MAX_OFFSET_LEFT": 75, "POS_MAX_OFFSET_RIGHT": 75,
        "GLAETTUNG_TAU_S": 0.05, "GY_HYSTERESE": 0.02, "GX_HYSTERESE": 0.01,
        "LENK_EXPO": 1.8, "LENK_TEMPO_KOPPLUNG": 0.5,
        "LENK_BEZUGSTAKT_S": 0.22, "LENK_MAX_DREHUNG": 0,
    },
}

# Welches Profil zuletzt gewählt wurde. PROFIL_MANUELL bedeutet: von Hand
# verstellt, also keinem Profil mehr zuzuordnen. Steht in metadata.json und in
# jedem Änderungsereignis, damit in der Auswertung nach Fahrstil gruppiert
# werden kann.
PROFIL_MANUELL  = "manuell"
fahrprofil_name = "Standard"


def fahrverhalten_werte() -> dict:
    """
    Momentaufnahme aller zur Laufzeit einstellbaren Fahrgrößen.

    Eine gemeinsame Quelle für die Aufzeichnung (Ausgangsstand beim Start der
    Sitzung) und für das Einstellfenster (Erkennen von Änderungen). Käme jede
    Stelle mit einer eigenen Liste, würde ein neu hinzugefügter Regler früher
    oder später an einer davon fehlen und in den Daten unsichtbar bleiben.
    """
    return {
        "MIN_SPEED_DYN": MIN_SPEED_DYN, "MAX_SPEED_DYN": MAX_SPEED_DYN,
        "GX_FULL_SPEED": GX_FULL_SPEED,
        "GY_LEFT_THRESHOLD": GY_LEFT_THRESHOLD,
        "MAX_TURN_ANGLE_LEFT": MAX_TURN_ANGLE_LEFT,
        "GY_RIGHT_THRESHOLD": GY_RIGHT_THRESHOLD,
        "MAX_TURN_ANGLE_RIGHT": MAX_TURN_ANGLE_RIGHT,
        "TURN_SPEED_FACTOR": TURN_SPEED_FACTOR,
        "ROLL_COMMAND_DURATION": ROLL_COMMAND_DURATION,
        "STOP_TIME": STOP_TIME,
        "BACKWARD_SPEED": BACKWARD_SPEED,
        "BACKWARD_DURATION": BACKWARD_DURATION,
        "GLAETTUNG_TAU_S": GLAETTUNG_TAU_S,
        "GY_HYSTERESE": GY_HYSTERESE,
        "GX_HYSTERESE": GX_HYSTERESE,
        "STEUERMODUS": STEUERMODUS,
        "POS_MAX_OFFSET_LEFT": POS_MAX_OFFSET_LEFT,
        "POS_MAX_OFFSET_RIGHT": POS_MAX_OFFSET_RIGHT,
        "LENK_EXPO": LENK_EXPO,
        "LENK_TEMPO_KOPPLUNG": LENK_TEMPO_KOPPLUNG,
        "LENK_BEZUGSTAKT_S": LENK_BEZUGSTAKT_S,
        "LENK_MAX_DREHUNG": LENK_MAX_DREHUNG,
    }


# Wertebereiche der einstellbaren Größen. Dieselben Grenzen wie die Regler im
# Einstellfenster – hier zusätzlich als Prüfung beim Laden gespeicherter Werte:
# Eine von Hand bearbeitete oder aus einer älteren Programmfassung stammende
# participants.json darf dem Sphero keine unsinnigen Befehle einbringen.
FAHRWERT_GRENZEN = {
    "MIN_SPEED_DYN":         (0, 150),
    "MAX_SPEED_DYN":         (20, 255),
    "GX_FULL_SPEED":         (-0.95, -0.30),
    "GY_LEFT_THRESHOLD":     (0.40, 0.95),
    "MAX_TURN_ANGLE_LEFT":   (10, 90),
    "GY_RIGHT_THRESHOLD":    (-0.95, -0.40),
    "MAX_TURN_ANGLE_RIGHT":  (10, 90),
    "TURN_SPEED_FACTOR":     (0.20, 1.00),
    "ROLL_COMMAND_DURATION": (0.04, 0.15),
    "STOP_TIME":             (0.2, 2.0),
    "BACKWARD_SPEED":        (20, 200),
    "BACKWARD_DURATION":     (0.5, 5.0),
    "GLAETTUNG_TAU_S":       (0.0, 0.60),
    "GY_HYSTERESE":          (0.0, 0.15),
    "GX_HYSTERESE":          (0.0, 0.10),
    "POS_MAX_OFFSET_LEFT":   (10, 180),
    "POS_MAX_OFFSET_RIGHT":  (10, 180),
    "LENK_EXPO":             (1.0, 4.0),
    "LENK_TEMPO_KOPPLUNG":   (0.0, 1.0),
    "LENK_BEZUGSTAKT_S":     (0.05, 0.40),
    "LENK_MAX_DREHUNG":      (0, 360),
}

# STEUERMODUS ist keine Zahl und wird deshalb getrennt geprüft.
STEUERMODI = (STEUERMODUS_RATE, STEUERMODUS_POSITION)

_FAHRWERT_GANZZAHLIG = {
    "MIN_SPEED_DYN", "MAX_SPEED_DYN", "MAX_TURN_ANGLE_LEFT",
    "MAX_TURN_ANGLE_RIGHT", "BACKWARD_SPEED",
    "POS_MAX_OFFSET_LEFT", "POS_MAX_OFFSET_RIGHT",
    "LENK_MAX_DREHUNG",
}


def fahrverhalten_anwenden(werte: dict, profil: str = None) -> list:
    """
    Gespeicherte Fahrwerte übernehmen (Gegenstück zu fahrverhalten_werte).

    Unbekannte Schlüssel werden übergangen, fehlende bleiben auf ihrem
    bisherigen Stand, und jeder Wert wird auf seinen zulässigen Bereich
    begrenzt. Damit kann eine unvollständige oder veraltete Datei die Steuerung
    nicht in einen unfahrbaren Zustand bringen – im schlimmsten Fall gilt
    weiterhin die Voreinstellung.

    Rückgabe: Namen der tatsächlich geänderten Größen.
    """
    global MIN_SPEED_DYN, MAX_SPEED_DYN, GX_FULL_SPEED
    global GY_LEFT_THRESHOLD, MAX_TURN_ANGLE_LEFT
    global GY_RIGHT_THRESHOLD, MAX_TURN_ANGLE_RIGHT
    global TURN_SPEED_FACTOR, ROLL_COMMAND_DURATION
    global STOP_TIME, BACKWARD_SPEED, BACKWARD_DURATION
    global GLAETTUNG_TAU_S, GY_HYSTERESE, GX_HYSTERESE
    global STEUERMODUS, POS_MAX_OFFSET_LEFT, POS_MAX_OFFSET_RIGHT
    global LENK_EXPO, LENK_TEMPO_KOPPLUNG
    global LENK_BEZUGSTAKT_S, LENK_MAX_DREHUNG
    global fahrprofil_name

    if not isinstance(werte, dict):
        return []

    vorher   = fahrverhalten_werte()
    geprueft = {}
    for name, grenzen in FAHRWERT_GRENZEN.items():
        if name not in werte:
            continue
        try:
            wert = float(werte[name])
        except (TypeError, ValueError):
            continue
        untere, obere = grenzen
        wert = max(untere, min(obere, wert))
        geprueft[name] = int(round(wert)) if name in _FAHRWERT_GANZZAHLIG else wert

    # Das Grundtempo darf das Höchsttempo nicht überschreiten, sonst kehrt sich
    # die Rampe in calc_speed() um (vgl. DrivingTuneWindow._anwenden).
    neu_min = geprueft.get("MIN_SPEED_DYN", MIN_SPEED_DYN)
    neu_max = geprueft.get("MAX_SPEED_DYN", MAX_SPEED_DYN)
    if neu_min > neu_max:
        geprueft["MIN_SPEED_DYN"] = neu_max

    MIN_SPEED_DYN         = geprueft.get("MIN_SPEED_DYN", MIN_SPEED_DYN)
    MAX_SPEED_DYN         = geprueft.get("MAX_SPEED_DYN", MAX_SPEED_DYN)
    GX_FULL_SPEED         = round(geprueft.get("GX_FULL_SPEED", GX_FULL_SPEED), 2)
    GY_LEFT_THRESHOLD     = round(geprueft.get("GY_LEFT_THRESHOLD", GY_LEFT_THRESHOLD), 3)
    MAX_TURN_ANGLE_LEFT   = geprueft.get("MAX_TURN_ANGLE_LEFT", MAX_TURN_ANGLE_LEFT)
    GY_RIGHT_THRESHOLD    = round(geprueft.get("GY_RIGHT_THRESHOLD", GY_RIGHT_THRESHOLD), 3)
    MAX_TURN_ANGLE_RIGHT  = geprueft.get("MAX_TURN_ANGLE_RIGHT", MAX_TURN_ANGLE_RIGHT)
    TURN_SPEED_FACTOR     = round(geprueft.get("TURN_SPEED_FACTOR", TURN_SPEED_FACTOR), 2)
    ROLL_COMMAND_DURATION = round(geprueft.get("ROLL_COMMAND_DURATION", ROLL_COMMAND_DURATION), 3)
    STOP_TIME             = round(geprueft.get("STOP_TIME", STOP_TIME), 2)
    BACKWARD_SPEED        = geprueft.get("BACKWARD_SPEED", BACKWARD_SPEED)
    BACKWARD_DURATION     = round(geprueft.get("BACKWARD_DURATION", BACKWARD_DURATION), 1)
    GLAETTUNG_TAU_S       = round(geprueft.get("GLAETTUNG_TAU_S", GLAETTUNG_TAU_S), 2)
    GY_HYSTERESE          = round(geprueft.get("GY_HYSTERESE", GY_HYSTERESE), 2)
    GX_HYSTERESE          = round(geprueft.get("GX_HYSTERESE", GX_HYSTERESE), 2)
    POS_MAX_OFFSET_LEFT   = geprueft.get("POS_MAX_OFFSET_LEFT", POS_MAX_OFFSET_LEFT)
    POS_MAX_OFFSET_RIGHT  = geprueft.get("POS_MAX_OFFSET_RIGHT", POS_MAX_OFFSET_RIGHT)
    LENK_EXPO             = round(geprueft.get("LENK_EXPO", LENK_EXPO), 2)
    LENK_TEMPO_KOPPLUNG   = round(geprueft.get("LENK_TEMPO_KOPPLUNG", LENK_TEMPO_KOPPLUNG), 2)
    LENK_BEZUGSTAKT_S     = round(geprueft.get("LENK_BEZUGSTAKT_S", LENK_BEZUGSTAKT_S), 3)
    LENK_MAX_DREHUNG      = geprueft.get("LENK_MAX_DREHUNG", LENK_MAX_DREHUNG)

    # Unbekannte Bezeichnung übergehen statt übernehmen: Ein Tippfehler in der
    # Datei darf nicht dazu führen, dass die Steuerung in keinem der beiden
    # Modi läuft.
    if werte.get("STEUERMODUS") in STEUERMODI:
        STEUERMODUS = werte["STEUERMODUS"]

    if profil:
        fahrprofil_name = profil

    nachher = fahrverhalten_werte()
    return [name for name in nachher if vorher[name] != nachher[name]]

# ── Herzfrequenzzonen ─────────────────────────────────────────────────────────
# Früher standen hier zwei feste Werte (100 / 120 BPM). Für einen Vergleich über
# Altersdekaden ist das nicht haltbar: Die maximale Herzfrequenz sinkt mit dem
# Alter um grob 0,7 Schläge pro Lebensjahr. 120 BPM sind für eine 30-jährige
# Person eine lockere Aufwärmintensität und für eine 80-jährige bereits ein
# deutlich anstrengender Bereich – dieselbe Zahl bedeutet also je nach Gruppe
# etwas völlig anderes, und genau das macht die Gruppen unvergleichbar.
#
# Die Zonen werden deshalb bei der Auswahl einer Testperson aus deren Alter und
# – sofern gemessen – deren Ruhepuls berechnet (siehe hr_zonen).
#
# WICHTIG: Das sind Intensitätsgrenzen aus der Trainingswissenschaft, keine
# medizinischen Grenzwerte. Die gesundheitliche Eignung klärt der PAR-Q und die
# Studienleitung; diese Anwendung ist kein Medizinprodukt.

# Solange keine Testperson gewählt ist (freies Testen), bleibt es bei den alten
# absoluten Werten – dann ist kein Alter bekannt, aus dem sich rechnen ließe.
HR_WARN_DEFAULT   = 100
HR_DANGER_DEFAULT = 120

# Maximale Herzfrequenz nach Tanaka et al. (2001), Meta-Analyse über 351
# Studien: HFmax = 208 - 0,7 x Alter. Deutlich besser belegt als die verbreitete
# Faustformel 220 - Alter, die bei jungen Menschen zu hoch und bei älteren zu
# niedrig liegt – also ausgerechnet den Altersvergleich verzerren würde.
HR_MAX_INTERCEPT = 208.0
HR_MAX_SLOPE     = 0.7

# Zonengrenzen nach der Intensitätseinteilung des ACSM.
# Mit Ruhepuls wird über die Herzfrequenzreserve gerechnet (Karvonen), ohne
# Ruhepuls ersatzweise über den Prozentsatz der maximalen Herzfrequenz. Die
# beiden Prozentsätze sind NICHT austauschbar – sie beziehen sich auf
# verschiedene Skalen und sind deshalb getrennt hinterlegt.
HR_WARN_PCT_HRR    = 0.60   # Übergang moderat -> anstrengend
HR_DANGER_PCT_HRR  = 0.85   # Übergang anstrengend -> nahezu maximal
HR_WARN_PCT_MAX    = 0.77
HR_DANGER_PCT_MAX  = 0.94

HR_WARN    = HR_WARN_DEFAULT
HR_DANGER  = HR_DANGER_DEFAULT

# Klartext, wie die aktuellen Zonen zustande kamen – für Anzeige, metadata.json
# und Ereignisprotokoll. Ohne diese Angabe wäre später nicht mehr erkennbar, ob
# eine Sitzung mit Karvonen-Zonen oder mit Ersatzwerten gefahren wurde.
hr_zonen_herkunft = "Standardwerte (keine Testperson gewählt)"

# ── Live-Graph-Konfiguration ─────────────────────────────────────────────────
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


class Tiefpass:
    """
    Tiefpass erster Ordnung (exponentiell gleitender Mittelwert).

        geglättet += alpha * (roh - geglättet)

    Der Gewichtungsfaktor alpha wird aus der tatsächlich vergangenen Zeit
    berechnet: alpha = 1 - exp(-dt / tau). Das ist wichtig, weil die Uhr ihre
    Werte NICHT in exakt gleichen Abständen schickt – bei einem festen alpha
    würde die Glättung mit der Übertragungsrate schwanken und wäre zwischen
    Sitzungen nicht vergleichbar. So wirkt die Zeitkonstante tau immer gleich,
    egal ob gerade 20 oder 60 Werte pro Sekunde ankommen.

    Aussetzer werden erkannt: Nach einer längeren Lücke (Uhr kurz weg) beginnt
    der Filter neu, statt langsam von einem veralteten Wert herzukriechen.
    """

    NEUSTART_NACH_S = 1.0

    def __init__(self):
        self._wert = None
        self._t    = 0.0

    def reset(self):
        self._wert = None

    def __call__(self, roh: float, jetzt: float, tau: float) -> float:
        if tau <= 0.0:
            self._wert, self._t = roh, jetzt
            return roh
        dt = jetzt - self._t
        self._t = jetzt
        if self._wert is None or dt <= 0.0 or dt > self.NEUSTART_NACH_S:
            self._wert = roh
            return roh
        alpha = 1.0 - math.exp(-dt / tau)
        self._wert += alpha * (roh - self._wert)
        return self._wert


# Je eine Filterinstanz für die beiden steuernden Achsen. gz wird von der
# Steuerung nicht ausgewertet und bleibt deshalb ungefiltert.
_filter_gx = Tiefpass()
_filter_gy = Tiefpass()


def get_state(gx, gy, gz, vorher: str = None) -> str:
    """
    Fahrzustand aus der Handhaltung.

    `vorher` ist der zuletzt gültige Zustand. Ist er angegeben, gilt eine
    Hysterese: Ein Zustand wird an seiner Schwelle betreten, aber erst wieder
    verlassen, wenn die Hand um GY_HYSTERESE bzw. GX_HYSTERESE darüber hinaus
    zurückgeht. Dadurch entsteht ein Halteband, in dem Zittern den Zustand nicht
    mehr umschalten kann.

    Ohne `vorher` verhält sich die Funktion wie zuvor (eine Schwelle je Seite).
    """
    # Beim Verlassen liegen die Schwellen um die Hysterese näher an der Mitte,
    # sind also schwerer zu unterschreiten.
    rechts_schwelle = GY_RIGHT_THRESHOLD + (GY_HYSTERESE if vorher == "right" else 0.0)
    links_schwelle  = GY_LEFT_THRESHOLD  - (GY_HYSTERESE if vorher == "left"  else 0.0)

    if gy < rechts_schwelle:  return "right"
    if gy > links_schwelle:   return "left"

    # Fahren/Stoppen: Der Bereich zwischen GX_FORWARD_MAX und
    # GX_NEUTRAL_THRESHOLD war schon bisher ein Totband. Mit `vorher` wird
    # daraus eine echte Hysterese – wer bereits fährt, darf die Hand etwas
    # weiter anheben, ohne dass die Fahrt abreißt.
    fahr_schwelle = GX_FORWARD_MAX + (GX_HYSTERESE if vorher == "forward" else 0.0)
    if gx > max(GX_NEUTRAL_THRESHOLD, fahr_schwelle): return "neutral"
    if gx < fahr_schwelle:                            return "forward"
    return "neutral"


def korrigiere_tragearm(gx, gy, gz):
    """
    Rechnet die Schwerkraftwerte auf die Bezugslage "linker Arm" um.

    Am rechten Arm liegt die Uhr um 180° um die Displayachse gedreht; gx und gy
    kehren dadurch ihr Vorzeichen um (Herleitung siehe watch_arm oben). Die
    Korrektur geschieht bewusst so früh wie möglich – direkt beim Empfang der
    Sensordaten. Dadurch gelten dieselben Schwellen, dieselbe Lenkung und
    dieselbe Auswertung für beide Arme, und in der Steuerung selbst muss an
    keiner Stelle zwischen links und rechts unterschieden werden.
    """
    if watch_arm == WATCH_ARM_RIGHT:
        return -gx, -gy, gz
    return gx, gy, gz


def calc_speed(gx) -> int:
    """
    Fahrtempo aus der Handneigung.

    Lineare Rampe zwischen den beiden Enden des nutzbaren Neigungsbereichs:
    bei GX_FORWARD_MAX (Hand knapp unter der Fahrschwelle) fährt der Sphero
    MIN_SPEED_DYN, bei GX_FULL_SPEED (Hand voll gesenkt) MAX_SPEED_DYN.

    Beide Enden standen früher als Zahlen in dieser Formel. Sie sind jetzt
    dieselben Konstanten, die auch get_state() und die Oberfläche verwenden –
    sonst würde ein Verstellen der Fahrschwelle die Tempo-Rampe stillschweigend
    gegen die Zustandserkennung verschieben.
    """
    span      = GX_FORWARD_MAX - GX_FULL_SPEED
    intensity = (GX_FORWARD_MAX - gx) / span if span > 1e-6 else 0.0
    intensity = max(0.0, min(1.0, intensity))
    return int(MIN_SPEED_DYN + intensity * (MAX_SPEED_DYN - MIN_SPEED_DYN))


def _drehanteil(gy_value) -> float:
    """
    Wie weit die Hand über die Kippschwelle hinaus gedreht ist: 0 an der
    Schwelle, 1 bei vollem Ausschlag.

    Der Nullpunkt ist die Schwelle DERSELBEN Seite. Vorher stand hier eine
    gemeinsame Konstante (TURN_DEADZONE = 0.80). Solange beide Schwellen bei
    0.80 lagen, war das gleichwertig – sobald die linke Schwelle aber tiefer
    liegt, entstünde dazwischen ein toter Bereich: Der Zustand wäre "links",
    der Drehwinkel aber 0. Der Sphero würde also in den Kurvenmodus wechseln
    (langsamer werden, Farbe umschalten), ohne sich zu drehen.

    Gemeinsame Grundlage beider Steuerungsarten: Die Ratensteuerung macht daraus
    Grad je Durchlauf, die Positionssteuerung Grad Kursversatz. Damit sich
    beide gleich anfühlen, was das Ansprechen angeht, teilen sie diese Kennlinie.
    """
    links     = gy_value > 0
    threshold = abs(GY_LEFT_THRESHOLD) if links else abs(GY_RIGHT_THRESHOLD)
    span      = max(1.0 - threshold, 1e-6)
    intensity = (abs(gy_value) - threshold) / span
    anteil    = max(0.0, min(1.0, intensity))
    # Kennlinie spreizen: Bei LENK_EXPO > 1 wirken kleine Handdrehungen
    # deutlich schwächer, der Vollausschlag bleibt unverändert bei 1.
    return anteil if LENK_EXPO == 1.0 else anteil ** LENK_EXPO


def calc_turn(gy_value, tempo=None, dt=None) -> float:
    """
    Ratensteuerung: Drehwinkel je Lenkschritt, 0° an der Kippschwelle bis
    MAX_TURN_ANGLE bei voller Kippung. Wird fortlaufend auf den Kurs aufaddiert,
    solange die Hand gedreht bleibt.

    `tempo` ist das gerade gefahrene Tempo. Es wird nur gebraucht, wenn die
    Drehrate ans Tempo gekoppelt ist (LENK_TEMPO_KOPPLUNG > 0): Dann ergibt eine
    gehaltene Handstellung einen Kreis mit festem Radius statt einer festen
    Drehgeschwindigkeit. Ohne Angabe bleibt die Kopplung wirkungslos.

    `dt` ist die seit dem letzten Lenkschritt tatsächlich vergangene Zeit. Damit
    wird MAX_TURN_ANGLE von "je Durchlauf" auf "je LENK_BEZUGSTAKT_S Sekunden"
    umgerechnet – siehe die ausführliche Begründung bei LENK_BEZUGSTAKT_S.
    Ohne Angabe (Vorschau in der Oberfläche) wird der Bezugstakt selbst
    eingesetzt, die Funktion liefert dann den unskalierten Winkel je Takt.

    Der Zeitfaktor ist nach oben begrenzt: Bleibt die Schleife einmal länger
    hängen (Funkaussetzer, Kamera blockiert kurz), soll die Nachholdrehung nicht
    als Sprung herauskommen. Lieber einmal etwas zu wenig gedreht als der Ball
    plötzlich quer.
    """
    max_winkel = MAX_TURN_ANGLE_LEFT if gy_value > 0 else MAX_TURN_ANGLE_RIGHT
    winkel     = _drehanteil(gy_value) * max_winkel

    if LENK_TEMPO_KOPPLUNG > 0.0 and tempo is not None:
        # Bezug ist das höchste in der Kurve erreichbare Tempo, damit der Faktor
        # bei Vollgas tatsächlich 1 wird und die Kopplung die Drehrate im
        # oberen Bereich nicht künstlich beschneidet.
        bezug  = max(1.0, MAX_SPEED_DYN * TURN_SPEED_FACTOR)
        anteil = max(0.0, min(1.0, float(tempo) / bezug))
        winkel *= (1.0 - LENK_TEMPO_KOPPLUNG) + LENK_TEMPO_KOPPLUNG * anteil

    takt = LENK_BEZUGSTAKT_S if LENK_BEZUGSTAKT_S > 1e-6 else 0.22
    if dt is not None:
        winkel *= max(0.0, min(float(dt) / takt, LENK_MAX_ZEITFAKTOR))
    return winkel


def calc_kursversatz(gy_value) -> float:
    """
    Positionssteuerung: Kursversatz gegenüber dem Bezugskurs, mit Vorzeichen.

    Positiv = nach rechts, negativ = nach links – dieselbe Zählrichtung wie beim
    Sphero-Kurs (im Uhrzeigersinn positiv). Anders als bei calc_turn wird hier
    NICHT aufaddiert: Der Wert ist die vollständige Abweichung, die sich aus der
    aktuellen Handhaltung ergibt. Hand zurück heißt Kurs zurück.
    """
    if gy_value > 0:   # Hand nach links gedreht
        return -_drehanteil(gy_value) * POS_MAX_OFFSET_LEFT
    return +_drehanteil(gy_value) * POS_MAX_OFFSET_RIGHT


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


def hr_max_tanaka(age_years) -> float:
    """Maximale Herzfrequenz nach Tanaka et al. (2001): 208 - 0,7 x Alter."""
    return HR_MAX_INTERCEPT - HR_MAX_SLOPE * float(age_years)


def hr_zonen(age_years, ruhepuls=None) -> dict:
    """
    Individuelle Warn- und Gefahrenschwelle für die Herzfrequenz.

    Mit Ruhepuls wird nach Karvonen über die Herzfrequenzreserve gerechnet:
        Zielpuls = Ruhepuls + Anteil x (HFmax - Ruhepuls)
    Das ist die genauere Variante, weil sie die individuelle Leistungsfähigkeit
    einbezieht – zwei gleichaltrige Personen mit Ruhepuls 50 und 80 haben bei
    derselben absoluten Herzfrequenz eine deutlich verschiedene Beanspruchung.

    Ohne Ruhepuls bleibt nur der Prozentsatz der maximalen Herzfrequenz. Das
    Ergebnis ist gröber, aber immer noch altersnormiert und damit über die
    Gruppen hinweg vergleichbar.

    Rückgabe enthält neben den Schwellen auch, wie sie zustande kamen – diese
    Angabe wandert in die Sitzungs-Metadaten.
    """
    hrmax = hr_max_tanaka(age_years)
    if valid_hr(ruhepuls):
        ruhe    = float(ruhepuls)
        reserve = hrmax - ruhe
        warn    = ruhe + HR_WARN_PCT_HRR   * reserve
        gefahr  = ruhe + HR_DANGER_PCT_HRR * reserve
        verfahren = "Karvonen (Herzfrequenzreserve)"
        herkunft  = (f"Alter {int(age_years)} J. -> HFmax {hrmax:.0f}, "
                     f"Ruhepuls {ruhe:.0f} -> Reserve {reserve:.0f} BPM "
                     f"({int(HR_WARN_PCT_HRR*100)} % / "
                     f"{int(HR_DANGER_PCT_HRR*100)} % der Reserve)")
    else:
        ruhe      = None
        warn      = HR_WARN_PCT_MAX   * hrmax
        gefahr    = HR_DANGER_PCT_MAX * hrmax
        verfahren = "Prozent der maximalen Herzfrequenz (kein Ruhepuls gemessen)"
        herkunft  = (f"Alter {int(age_years)} J. -> HFmax {hrmax:.0f} "
                     f"({int(HR_WARN_PCT_MAX*100)} % / "
                     f"{int(HR_DANGER_PCT_MAX*100)} % von HFmax)")

    return {
        "warn":       int(round(warn)),
        "gefahr":     int(round(gefahr)),
        "hr_max":     round(hrmax, 1),
        "ruhepuls":   ruhe,
        "verfahren":  verfahren,
        "herkunft":   herkunft,
    }


def setze_hr_zonen(record) -> str:
    """
    Herzfrequenzzonen auf die gewählte Testperson umstellen.

    Ohne Testperson (freies Testen) wird auf die absoluten Vorgabewerte
    zurückgeschaltet, damit nicht versehentlich die Zonen der zuvor gewählten
    Person weitergelten.
    """
    global HR_WARN, HR_DANGER, hr_zonen_herkunft

    if not record or not record.get("age_years"):
        HR_WARN, HR_DANGER = HR_WARN_DEFAULT, HR_DANGER_DEFAULT
        hr_zonen_herkunft  = "Standardwerte (keine Testperson gewählt)"
        return hr_zonen_herkunft

    zonen = hr_zonen(record["age_years"], record.get("resting_hr_bpm"))
    HR_WARN, HR_DANGER = zonen["warn"], zonen["gefahr"]
    hr_zonen_herkunft  = f"{zonen['verfahren']}: {zonen['herkunft']}"
    return hr_zonen_herkunft


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
    # gx/gy/gz sind die unveränderten Messwerte der Uhr (auf den Tragearm
    # normiert). gx_filt/gy_filt sind dieselben Größen nach der Glättung –
    # also genau das, worauf die Steuerung reagiert hat. Bei abgeschalteter
    # Glättung (GLAETTUNG_TAU_S = 0) sind beide Paare identisch.
    "sensor":       ["t_rel_s", "t_abs_s", "timestamp", "gx", "gy", "gz",
                      "accel_x", "accel_y", "accel_z", "heart_rate", "intensity",
                      "gx_filt", "gy_filt"],
    "control":      ["t_rel_s", "t_abs_s", "timestamp", "gx", "gy", "gz",
                      "state", "heading_deg", "speed_cmd", "is_stopped"],
    # Schulterwinkel: Elevation = Hebung des Oberarms gegenüber dem Rumpf,
    # Ebene = Richtung der Hebung (0° seitlich, 90° nach vorne).
    # Ellbogenwinkel bleibt als Qualitätskontrolle (Arm gestreckt?) erhalten.
    "tracking":     ["t_rel_s", "t_abs_s", "timestamp", "person_id", "distance_m",
                      "shoulder_elev_left_deg", "shoulder_elev_right_deg",
                      "shoulder_plane_left_deg", "shoulder_plane_right_deg",
                      "elbow_left_deg", "elbow_right_deg"],
    "events":       ["t_rel_s", "t_abs_s", "timestamp", "event", "detail"],
    "video_frames": ["frame_index", "t_rel_s", "t_abs_s", "timestamp"],
    # Ein Datensatz je gefahrenem Aufgabendurchgang – die auswertbare
    # Zielgröße der Studie. Bewusst als eigene, flache Tabelle: So lässt sich
    # ohne Vorverarbeitung über alle Sitzungen hinweg vergleichen, während
    # sensor/control die Rohdaten für die Detailanalyse bereithalten.
    #
    # gueltig=0 heißt: Durchgang wurde als ungültig markiert (Abbruch, Störung).
    # Solche Zeilen werden NICHT gelöscht, sondern bleiben mit Begründung
    # stehen – nachträgliches Entfernen misslungener Versuche wäre eine
    # Verzerrung der Stichprobe.
    # abstand_m: Abstand der beiden Bodenmarkierungen. Steht in JEDER Zeile,
    # damit Zeiten aus verschiedenen Aufbauten nie stillschweigend in einen
    # Topf geraten – eine Sekunde auf 1 m Strecke ist etwas anderes als auf 2 m.
    # rw_strecke_m: Länge der geraden Rückwärtsstrecke (Markierung 1 → 3).
    #
    # backward_dauer_s / backward_speed: die zum Zeitpunkt des Durchgangs
    # gültigen Rückwärtsparameter. Sie sind zur Laufzeit verstellbar
    # (Fahrverhalten justieren) und bestimmen unmittelbar, wie weit ein
    # Doppeltipp trägt – ohne sie wäre die Rückwärtszeit zwischen Personen
    # nicht vergleichbar, und die Abweichung fiele erst bei der Auswertung auf.
    #
    # Zeiten je Abschnitt statt nur Gesamtzeit: Vorwärts- und Rückwärtsanteil
    # sind die eigentlich interessante Gegenüberstellung. t_standard_s ist die
    # Summe der Vorwärtsabschnitte und entspricht der bisherigen Gesamtzeit –
    # damit bleiben früher erhobene Durchgänge vergleichbar.
    #
    # Verglichen wird über die Tempi, nicht über die rohen Zeiten: Hinweg und
    # Rückwärtsstrecke sind unterschiedlich lang, ein Zeitverhältnis wäre also
    # von der Streckenwahl abhängig. ratio_rw_vw = Tempo rückwärts / Tempo
    # vorwärts ist dagegen streckenunabhängig (Werte < 1 = langsamer rückwärts).
    #
    # teilnehmer_id steht in JEDER Zeile: Die Sitzungsordner sind zwar nach
    # Person sortiert, eine herausgeloeste CSV-Zeile waere ohne die Kennung
    # aber nicht mehr zuzuordnen.
    #
    # ZWEI Zaehlungen, weil eine Person ihre Durchgaenge ueber mehrere
    # Aufzeichnungen verteilt (Pause zwischen den Durchgaengen, die
    # Aufzeichnung wird dazwischen beendet):
    #   versuch            - fortlaufend ueber ALLE Sitzungen dieser Person.
    #                        Das ist die Nummer aus dem Studienprotokoll:
    #                        Durchgang 1 = UEbungsdurchgang, 2..n = Wertung.
    #   versuch_in_session - Position innerhalb dieses Sitzungsordners.
    #                        Nur zur Nachvollziehbarkeit, nie zur Auswertung.
    "aufgaben":     ["teilnehmer_id", "versuch", "versuch_in_session",
                      "abstand_m", "rw_strecke_m",
                      "backward_dauer_s", "backward_speed",
                      "t_rel_start_s", "t_rel_ende_s", "dauer_s",
                      "t_hinweg_s", "t_rundkurs_s", "t_standard_s",
                      "t_rueckwaerts_s",
                      "v_hinweg_ms", "v_rueckwaerts_ms", "ratio_rw_vw",
                      "zwischenzeiten_s", "halte", "lenkwechsel",
                      "rueckwaerts", "rueckwaerts_korrektur",
                      "rueckwaerts_strecke",
                      "hr_mittel", "hr_max", "gueltig", "notiz"],
}

# Sammeltabelle je Testperson: sessions/<ID>/aufgaben_alle.csv
#
# Warum zusaetzlich zu den Sitzungs-CSVs: Die vier Durchgaenge einer Person
# entstehen in der Regel in vier getrennten Aufzeichnungen, weil zwischen den
# Durchgaengen Pause gemacht wird und die Aufzeichnung dabei beendet ist. Fuer
# die Auswertung ("Durchgang 1 verwerfen, 2 bis 4 mitteln") muessen sie aber in
# EINER Tabelle stehen. Die Datei ist reine Ableitung - sie enthaelt keine
# Angabe, die nicht auch in den Sitzungsordnern steht, und laesst sich daraus
# jederzeit neu erzeugen (siehe sammel_csv_aufbauen).
AUFGABEN_SAMMEL_DATEI = "aufgaben_alle.csv"
_AUFGABEN_SAMMEL_SCHEMA = (["teilnehmer_id", "session_id"]
                           + [spalte for spalte in _CSV_SCHEMAS["aufgaben"]
                              if spalte != "teilnehmer_id"])


def teilnehmer_ordner(participant_id: str) -> str:
    """Ordner, in dem alle Sitzungen einer Testperson liegen."""
    return os.path.join(SESSIONS_DIR, participant_id)


def sammel_csv_pfad(participant_id: str) -> str:
    return os.path.join(teilnehmer_ordner(participant_id), AUFGABEN_SAMMEL_DATEI)


def _sitzungsordner(participant_id: str):
    """
    Sitzungsordner einer Person in zeitlicher Reihenfolge.

    Der Ordnername endet auf den Zeitstempel des Aufnahmestarts; alphabetisch
    sortiert ist damit zugleich chronologisch sortiert.
    """
    basis = teilnehmer_ordner(participant_id)
    if not os.path.isdir(basis):
        return []
    return sorted(
        os.path.join(basis, name) for name in os.listdir(basis)
        if os.path.isdir(os.path.join(basis, name))
        and os.path.isfile(os.path.join(basis, name, "aufgaben.csv")))


def sammel_csv_aufbauen(participant_id: str) -> str:
    """
    Die Sammeltabelle einer Person aus ihren Sitzungsordnern neu erzeugen.

    Zwei Faelle, die unterschiedlich behandelt werden muessen:

    Neue Sitzungen fuehren die Spalte teilnehmer_id und darin bereits die
    personenweite Durchgangsnummer - die wird unveraendert uebernommen.

    Aeltere Sitzungen (vor dieser Aenderung) haben die Spalte nicht. Dort war
    "versuch" ein Zaehler des Aufgabenfensters, der bei jedem Neuoeffnen wieder
    bei 1 begann; dieselbe Person hat dadurch mehrere Zeilen "Durchgang 1".
    Fuer solche Zeilen wird die Nummer aus der zeitlichen Reihenfolge der
    Sitzungsordner neu vergeben - das ist die einzige Information, die den
    tatsaechlichen Ablauf noch wiedergibt. Die Sitzungs-CSVs bleiben dabei
    unangetastet, der urspruengliche Zaehlerstand ist also nicht verloren.
    """
    ordner = teilnehmer_ordner(participant_id)
    if not os.path.isdir(ordner):
        return ""

    zeilen = []
    laufend = 0
    for sitzung in _sitzungsordner(participant_id):
        session_id = os.path.basename(sitzung)
        try:
            with open(os.path.join(sitzung, "aufgaben.csv"),
                      newline="", encoding="utf-8") as f:
                daten = list(csv.DictReader(f))
        except OSError:
            continue
        for i, satz in enumerate(daten, start=1):
            laufend += 1
            # Neues Format erkennt man an teilnehmer_id; nur dann ist die
            # gespeicherte Nummer schon personenweit vergeben.
            gespeichert = (satz.get("versuch") or "").strip()
            if satz.get("teilnehmer_id") and gespeichert.isdigit():
                nummer  = int(gespeichert)
                laufend = max(laufend, nummer)
            else:
                nummer = laufend
            satz["teilnehmer_id"]      = participant_id
            satz["session_id"]         = session_id
            satz["versuch"]            = nummer
            satz["versuch_in_session"] = satz.get("versuch_in_session") or i
            zeilen.append(satz)

    pfad = sammel_csv_pfad(participant_id)
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        schreiber = csv.writer(f)
        schreiber.writerow(_AUFGABEN_SAMMEL_SCHEMA)
        for satz in zeilen:
            schreiber.writerow([satz.get(spalte, "")
                                for spalte in _AUFGABEN_SAMMEL_SCHEMA])
    return pfad


def sammel_csv_sicherstellen(participant_id: str) -> str:
    """
    Pfad der Sammeltabelle liefern und sie anlegen, falls sie noch fehlt.

    Damit werden auch Personen erfasst, deren Durchgaenge vor dieser Aenderung
    aufgezeichnet wurden: Beim ersten Zugriff wird die Tabelle einmalig aus
    den vorhandenen Sitzungsordnern nachgebaut.
    """
    if not participant_id:
        return ""
    pfad = sammel_csv_pfad(participant_id)
    if os.path.isfile(pfad):
        return pfad
    if not os.path.isdir(teilnehmer_ordner(participant_id)):
        return pfad
    return sammel_csv_aufbauen(participant_id) or pfad


def durchgaenge_der_person(participant_id: str):
    """
    Alle bisher gespeicherten Durchgaenge einer Person, nach Nummer sortiert.

    Grundlage fuer die Weiterzaehlung und fuer die Ergebnistabelle im
    Aufgabenfenster - beide muessen ueber Sitzungsgrenzen hinweg dasselbe Bild
    zeigen wie die spaetere Auswertung.
    """
    if not participant_id:
        return []
    pfad = sammel_csv_sicherstellen(participant_id)
    if not pfad or not os.path.isfile(pfad):
        return []
    try:
        with open(pfad, newline="", encoding="utf-8") as f:
            zeilen = list(csv.DictReader(f))
    except OSError:
        return []

    def nummer(satz):
        wert = (satz.get("versuch") or "").strip()
        return int(wert) if wert.isdigit() else 0

    return sorted(zeilen, key=nummer)


def naechste_versuchsnummer(participant_id: str) -> int:
    """
    Nummer, die der naechste Durchgang dieser Person bekommt.

    Gezaehlt wird ueber alle Sitzungen hinweg, damit die Nummerierung dem
    Studienprotokoll entspricht (Durchgang 1 = UEbung, 2..n = Wertung) und
    nicht davon abhaengt, wie oft die Aufzeichnung zwischendurch beendet
    wurde. Verworfene Durchgaenge verbrauchen ihre Nummer mit: Sie bleiben als
    gueltig=0 in den Daten stehen, und eine spaeter neu vergebene Nummer wuerde
    zwei verschiedene Fahrten gleich benennen.
    """
    hoechste = 0
    for satz in durchgaenge_der_person(participant_id):
        wert = (satz.get("versuch") or "").strip()
        if wert.isdigit():
            hoechste = max(hoechste, int(wert))
    return hoechste + 1


STUDIE_SAMMEL_DATEI = "aufgaben_alle_personen.csv"


def studien_csv_aufbauen() -> str:
    """
    Alle Durchgaenge aller Testpersonen in EINE Tabelle schreiben.

    sessions/aufgaben_alle_personen.csv ist die Datei, die in der Auswertung
    tatsaechlich geoeffnet wird: eine Zeile je gefahrenem Durchgang, mit
    teilnehmer_id und personenweiter Durchgangsnummer, also unmittelbar
    gruppierbar. Sie wird bei jedem Ende einer Aufzeichnung vollstaendig neu
    erzeugt statt fortgeschrieben - so kann sie nicht auseinanderlaufen, und
    von Hand geloeschte oder nachtraeglich korrigierte Sitzungen schlagen
    sofort durch.
    """
    if not os.path.isdir(SESSIONS_DIR):
        return ""
    personen = sorted(
        name for name in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, name)))

    pfad = os.path.join(SESSIONS_DIR, STUDIE_SAMMEL_DATEI)
    try:
        with open(pfad, "w", newline="", encoding="utf-8") as f:
            schreiber = csv.writer(f)
            schreiber.writerow(_AUFGABEN_SAMMEL_SCHEMA)
            for pid in personen:
                for satz in durchgaenge_der_person(pid):
                    schreiber.writerow([satz.get(spalte, "")
                                        for spalte in _AUFGABEN_SAMMEL_SCHEMA])
    except OSError as e:
        print(f"[WARN] Studientabelle {pfad} nicht schreibbar: {e}")
        return ""
    return pfad


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
        self._watch_arm     = WATCH_ARM_LEFT
        self._belt_active   = False
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
            self._watch_arm     = watch_arm
            self._belt_active   = vibrationsguertel.ist_verbunden()

            # Sitzungen liegen unter sessions/<Teilnehmer-ID>/, damit in der
            # Auswertung alle Aufnahmen einer Person beisammen liegen.
            participant_dir = teilnehmer_ordner(self.participant_id)
            os.makedirs(participant_dir, exist_ok=True)
            # Sammeltabelle der Person JETZT bereitstellen, nicht erst beim
            # ersten Durchgang: Fehlt sie, wird sie aus den vorhandenen
            # Sitzungsordnern nachgebaut - danach kann jeder Durchgang einfach
            # angehaengt werden, ohne dass eine bereits erfasste Zeile ein
            # zweites Mal hineingeraet.
            sammel_csv_sicherstellen(self.participant_id)
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
            self._log_event_locked("watch_arm", self._watch_arm)
            # Ausgangsstand des Fahrverhaltens festhalten. metadata.json enthält
            # am Ende nur den Schlussstand; erst zusammen mit diesem Eintrag und
            # den "fahrverhalten_geaendert"-Ereignissen lässt sich für jeden
            # Zeitpunkt der Aufzeichnung rekonstruieren, womit gefahren wurde.
            self._log_event_locked(
                "fahrverhalten_start",
                f"profil={fahrprofil_name}; "
                + "; ".join(f"{name}: {wert}"
                            for name, wert in fahrverhalten_werte().items()))
            self._log_event_locked(
                "herzfrequenzzonen",
                f"Warnung {HR_WARN} BPM; Gefahr {HR_DANGER} BPM; {hr_zonen_herkunft}")
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
            # Erst jetzt, nicht nach jedem Durchgang: Die Datei wird komplett
            # neu geschrieben, und zum Ende der Aufzeichnung ist der Stand
            # dieser Sitzung vollstaendig.
            studien_csv_aufbauen()
            return finished_dir

    # ── Logging-Methoden (threadsicher) ──────────────────────────────────────

    def t_rel(self, t_abs=None) -> float:
        """
        Sitzungszeit (0 = Aufnahmebeginn) zu einem absoluten Zeitpunkt.

        Für Auswertungen, die ihre Zeitpunkte selbst nehmen – etwa der
        Aufgaben-Timer – und sie trotzdem auf dieselbe Achse legen müssen wie
        alle aufgezeichneten Zeilen.
        """
        if t_abs is None:
            t_abs = clock.t_abs()
        return t_abs - self._t_offset

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

    def log_sensor(self, gx, gy, gz, ax, ay, az, hr, intensity,
                   gx_filt=None, gy_filt=None):
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
                 hr_out, intensity,
                 "" if gx_filt is None else f"{gx_filt:.5f}",
                 "" if gy_filt is None else f"{gy_filt:.5f}"])
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

    def log_tracking(self, person_id, distance, elev_left, elev_right,
                     plane_left=None, plane_right=None,
                     elbow_left=None, elbow_right=None):
        if not self.active:
            return
        with self._lock:
            if not self.active:
                return
            self._subsystems_seen["camera"] = True
            t_rel, t_abs, ts = self._now()
            # Fehlende Werte als leeres Feld = NaN in pandas, nicht als 0.
            def cell(v):
                return "" if v is None else v
            self._writers["tracking"].writerow(
                [f"{t_rel:.3f}", f"{t_abs:.3f}", ts, person_id, cell(distance),
                 cell(elev_left), cell(elev_right),
                 cell(plane_left), cell(plane_right),
                 cell(elbow_left), cell(elbow_right)])
            self._counts["tracking"] += 1
            self._maybe_flush()
            if distance is not None:
                self._plot["dist_t"].append(t_rel)
                self._plot["distance"].append(distance)
            if elev_left is not None or elev_right is not None:
                self._plot["angle_t"].append(t_rel)
                self._plot["angle_left"].append(
                    elev_left if elev_left is not None else float("nan"))
                self._plot["angle_right"].append(
                    elev_right if elev_right is not None else float("nan"))

    def log_aufgabe(self, zeile: dict):
        """
        Einen abgeschlossenen Aufgabendurchgang festhalten.

        Erwartet die Spalten aus _CSV_SCHEMAS["aufgaben"]; fehlende bleiben leer.
        Die Kennung der Testperson und die Position innerhalb der Sitzung
        ergaenzt der Recorder selbst - er kennt beides sicher, das
        Aufgabenfenster muesste es sich zusammensuchen.

        Geschrieben wird an ZWEI Stellen: in die CSV dieser Sitzung und in die
        Sammeltabelle der Person. Beides sofort geflusht, damit ein Absturz
        hoechstens den laufenden, nicht aber einen bereits beendeten Durchgang
        kostet.
        """
        if not self.active:
            return
        with self._lock:
            if not self.active or "aufgaben" not in self._writers:
                return

            satz = dict(zeile)
            satz["teilnehmer_id"]      = self.participant_id
            satz["versuch_in_session"] = self._counts["aufgaben"] + 1

            self._writers["aufgaben"].writerow(
                [satz.get(spalte, "") for spalte in _CSV_SCHEMAS["aufgaben"]])
            self._files["aufgaben"].flush()
            self._counts["aufgaben"] += 1

            self._sammel_zeile_anhaengen(satz)

    def _sammel_zeile_anhaengen(self, satz: dict):
        """
        Einen Durchgang an die Sammeltabelle der Person anhaengen.

        Die Datei wird je Durchgang geoeffnet und wieder geschlossen statt
        offen gehalten: Es sind wenige Zeilen je Sitzung, dafuer liegt der
        Stand nach jedem Durchgang vollstaendig auf der Platte - auch wenn die
        Aufzeichnung danach nicht sauber beendet wird.
        """
        if not self.participant_id:
            return
        satz = dict(satz)
        satz["session_id"] = os.path.basename(self.session_dir or "")
        pfad = sammel_csv_pfad(self.participant_id)
        try:
            neu_anlegen = not os.path.isfile(pfad)
            with open(pfad, "a", newline="", encoding="utf-8") as f:
                schreiber = csv.writer(f)
                if neu_anlegen:
                    schreiber.writerow(_AUFGABEN_SAMMEL_SCHEMA)
                schreiber.writerow([satz.get(spalte, "")
                                    for spalte in _AUFGABEN_SAMMEL_SCHEMA])
        except OSError as e:
            # Die Sitzungs-CSV ist zu diesem Zeitpunkt bereits geschrieben, der
            # Durchgang also nicht verloren - die Sammeltabelle laesst sich
            # daraus jederzeit mit sammel_csv_aufbauen() neu erzeugen.
            print(f"[WARN] Sammeltabelle {pfad} nicht beschreibbar: {e}")

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
            # Tragearm der Uhr beim Start der Aufzeichnung. Wichtig für die
            # Postanalyse: gx/gy in sensor.csv sind auf die Bezugslage
            # "linker Arm" normiert (siehe korrigiere_tragearm). Bei "rechts"
            # ergeben sich die Rohwerte der Uhr durch erneutes Umkehren der
            # Vorzeichen von gx und gy.
            "watch_arm": self._watch_arm,
            "gravity_frame": "normiert auf linken Arm",
            # Vibrationsgürtel beim Start der Aufzeichnung aktiv? Verbindungs-
            # aufbau und -verlust stehen zusätzlich als Ereignis in events.csv.
            "belt_active": self._belt_active,
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
            # ACHTUNG bei der Auswertung: Hier steht der Stand zum Zeitpunkt des
            # Schreibens, am Ende der Sitzung also der SCHLUSSSTAND. Wurde das
            # Fahrverhalten während der Aufzeichnung verstellt, steht der
            # Ausgangsstand im Ereignis "fahrverhalten_start" und jede einzelne
            # Änderung als "fahrverhalten_geaendert" in events.csv.
            "config": {
                "fahrprofil": fahrprofil_name,
                "MIN_SPEED_DYN": MIN_SPEED_DYN, "MAX_SPEED_DYN": MAX_SPEED_DYN,
                "GX_FULL_SPEED": GX_FULL_SPEED,
                "TURN_SPEED_FACTOR": TURN_SPEED_FACTOR, "STOP_TIME": STOP_TIME,
                "GY_RIGHT_THRESHOLD": GY_RIGHT_THRESHOLD, "GY_LEFT_THRESHOLD": GY_LEFT_THRESHOLD,
                "GX_FORWARD_MAX": GX_FORWARD_MAX, "GX_NEUTRAL_THRESHOLD": GX_NEUTRAL_THRESHOLD,
                "MAX_TURN_ANGLE_LEFT": MAX_TURN_ANGLE_LEFT,
                "MAX_TURN_ANGLE_RIGHT": MAX_TURN_ANGLE_RIGHT,
                "ROLL_COMMAND_DURATION": ROLL_COMMAND_DURATION,
                "CONTROL_LOOP_SLEEP": CONTROL_LOOP_SLEEP,
                "BACKWARD_DURATION": BACKWARD_DURATION, "BACKWARD_SPEED": BACKWARD_SPEED,
                "DATA_TIMEOUT": DATA_TIMEOUT,
                # Vorverarbeitung der Handhaltung. Bei GLAETTUNG_TAU_S = 0 und
                # beiden Hysteresen = 0 entspricht die Steuerung dem Stand vor
                # Einführung dieser Größen.
                "GLAETTUNG_TAU_S": GLAETTUNG_TAU_S,
                "GY_HYSTERESE": GY_HYSTERESE, "GX_HYSTERESE": GX_HYSTERESE,
                # "rate" = Handdrehung bestimmt die Drehgeschwindigkeit,
                # "position" = Handdrehung bestimmt den Kursversatz.
                "STEUERMODUS": STEUERMODUS,
                "POS_MAX_OFFSET_LEFT": POS_MAX_OFFSET_LEFT,
                "POS_MAX_OFFSET_RIGHT": POS_MAX_OFFSET_RIGHT,
                "LENK_EXPO": LENK_EXPO,
                "LENK_TEMPO_KOPPLUNG": LENK_TEMPO_KOPPLUNG,
                # Zeitbezug der Drehrate. Unverzichtbar für die Auswertung:
                # MAX_TURN_ANGLE_* allein sagt nichts mehr über die gefahrene
                # Drehgeschwindigkeit aus, erst zusammen mit dieser Zeitspanne.
                "LENK_BEZUGSTAKT_S": LENK_BEZUGSTAKT_S,
                "LENK_MAX_DREHUNG": LENK_MAX_DREHUNG,
                # Funkverhalten der Fahrbefehle – bestimmt mit, wie direkt die
                # Steuerung in dieser Sitzung reagiert hat.
                "SPHERO_CMD_INTERVAL_S": SPHERO_CMD_INTERVAL_S,
                "DRIVE_MIN_HEADING_DELTA": DRIVE_MIN_HEADING_DELTA,
                "DRIVE_MIN_SPEED_DELTA": DRIVE_MIN_SPEED_DELTA,
            },
            # Die Herzfrequenzzonen sind personenabhängig und werden aus Alter
            # und (falls gemessen) Ruhepuls berechnet. Ohne diese Angaben wäre
            # in der Auswertung nicht rekonstruierbar, worauf eine Warnung in
            # dieser Sitzung beruhte – und Sitzungen verschiedener Altersgruppen
            # wären nicht vergleichbar.
            "herzfrequenz": {
                "warn_bpm":       HR_WARN,
                "gefahr_bpm":     HR_DANGER,
                "herkunft":       hr_zonen_herkunft,
                "resting_hr_bpm": (self._participant or {}).get("resting_hr_bpm"),
                "hinweis": ("Intensitätsgrenzen nach der Einteilung des ACSM, "
                            "HFmax nach Tanaka et al. (2001). Keine medizinischen "
                            "Grenzwerte."),
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
            ax.plot(self._plot["angle_t"], self._plot["angle_left"],
                    label="Links", color="tab:blue")
            ax.plot(self._plot["angle_t"], self._plot["angle_right"],
                    label="Rechts", color="tab:orange")
            ax.axhline(SHOULDER_TARGET_DEG, color="green", linestyle="--", linewidth=1.0,
                       label=f"waagerecht ({SHOULDER_TARGET_DEG:.0f})")
            ax.set_ylim(0, 185)
            ax.set_title("Schulterwinkel – Hebung des Oberarms gegenueber dem Rumpf")
            ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Elevation (Grad)")
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

    # Sphero-Steuerung (Schwerkraft). Erst auf den Tragearm normieren, danach
    # arbeiten Steuerung, Anzeige und Aufzeichnung durchgehend mit denselben
    # Bezugsachsen – unabhängig davon, an welchem Arm die Uhr sitzt.
    gx_roh, gy_roh, gz = korrigiere_tragearm(float(data.get("gravityX", 0)),
                                             float(data.get("gravityY", 0)),
                                             float(data.get("gravityZ", 0)))

    # Zwei Uhren mit klarer Aufgabenteilung:
    #   time.time()     -> Steuerung (die Schleife vergleicht damit, s.o.)
    #   clock.t_abs()   -> Graphen und CSV (monotone Referenzuhr, siehe MasterClock)
    t_abs = clock.t_abs()

    # Glättung und Zustandsbestimmung geschehen NUR hier, an einer einzigen
    # Stelle. Steuerungsschleife, Gürtel und Anzeige lesen das Ergebnis aus
    # latest_data, statt es jeweils neu zu berechnen – sonst müsste dieselbe
    # Schwellenlogik an mehreren Stellen gepflegt werden und liefe früher oder
    # später auseinander.
    gx = _filter_gx(gx_roh, t_abs, GLAETTUNG_TAU_S)
    gy = _filter_gy(gy_roh, t_abs, GLAETTUNG_TAU_S)

    # Live-Graph (Beschleunigung + Herzfrequenz)
    ax = float(data.get("motionUserAccelerationX", 0))
    ay = float(data.get("motionUserAccelerationY", 0))
    az = float(data.get("motionUserAccelerationZ", 0))
    hr = float(data.get("heartRate", 0))
    intensity = math.sqrt(ax**2 + ay**2 + az**2)

    with data_lock:
        state = get_state(gx, gy, gz, vorher=latest_data["state"])
        latest_data["gx"]          = gx
        latest_data["gy"]          = gy
        latest_data["gz"]          = gz
        latest_data["state"]       = state
        latest_data["last_update"] = time.time()
        latest_data["heart_rate"]  = hr
        intensity_values.append(intensity)
        heart_rate_values.append(hr)
        graph_time_values.append(t_abs)

    # Aufgezeichnet wird beides: die Rohwerte als unveränderte Messung und die
    # geglätteten als das, worauf die Steuerung tatsächlich reagiert hat. Nur
    # die geglätteten zu speichern hieße, die Messung durch eine Einstellung zu
    # ersetzen – die Rohdaten wären dann unwiederbringlich weg.
    recorder.log_sensor(gx_roh, gy_roh, gz, ax, ay, az, hr, intensity, gx, gy)
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


class _BefehlUeberlastet(Exception):
    """
    Ein Befehl kam nicht in SPHERO_CMD_TIMEOUT durch – die Verbindung selbst ist
    aber nicht als tot erwiesen.

    Warum das eine eigene Ausnahme braucht: Bisher warf der Wächter
    concurrent.futures.TimeoutError, und _is_connection_error() wertet genau die
    als Verbindungsabbruch. Ein langsamer Durchlauf – Laptop wird warm und
    drosselt, USB3 der Kamera stört ins 2,4-GHz-Band, Funkkanal kurz voll –
    sah damit exakt aus wie ein Abriss. Nach drei solchen Durchläufen hat die
    Steuerung eine völlig intakte Verbindung abgebaut und einen Reconnect-Zyklus
    von mehreren Sekunden begonnen; der Sphero stand in dieser Zeit still.

    Getrennt geführt, kann die Schleife den Durchlauf einfach überspringen und
    weiterarbeiten. Erst wenn es viele Male hintereinander passiert, ist es doch
    ein Abriss (siehe MAX_CONSECUTIVE_OVERLOAD).
    """


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
            raise _BefehlUeberlastet(
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
            # Eigenes Zeitlimit gerissen. Das ist eine Aussage über UNSERE
            # Wartebereitschaft, nicht über den Zustand der Verbindung – deshalb
            # _BefehlUeberlastet und nicht TimeoutError (siehe dort).
            raise _BefehlUeberlastet(
                f"Sphero-Befehl {getattr(func, '__name__', func)} antwortet nicht innerhalb {timeout}s"
            )
        if "error" in result:
            raise result["error"]
        return result.get("value")


class _MatrixPfeil:
    """
    Zeichnet einen Richtungspfeil auf die 8x8-Matrix des BOLT.

    Das Problem, das diese Klasse loest, ist nicht das Zeichnen, sondern der
    Funkverkehr. set_main_led() faerbt beim BOLT die GESAMTE Matrix ein
    (ToyUtil.set_main_led -> set_led_matrix_one_colour) - jeder Farbwechsel des
    Fahrzustands loescht den Pfeil also wieder. Er muss danach neu gezeichnet
    werden, und dafuer sind mehrere Pakete noetig: eine Linie je Zeile der
    Pfeilspitze plus ein Rechteck fuer den Schaft.

    Wuerden die alle sofort hinterher geschickt, haenge die Steuerung bei jedem
    Zustandswechsel fuer die Dauer von fuenf Paketen - genau die Verzoegerung
    zwischen Handbewegung und Reaktion, die in den Tests ohnehin schon
    bemerkt wurde.

    Deshalb liegen die Zeichenbefehle in einer Warteschlange und es geht je
    Schleifendurchlauf hoechstens EINER davon raus. Der erste wird beim
    Farbwechsel sofort gesendet - so viel hat der Wechsel auch gekostet, als
    noch die ganze Matrix gefuellt wurde -, die restlichen vier verteilen sich
    auf die naechsten Durchlaeufe.

    Fuer die Testperson ist das kein Notbehelf, sondern die Anzeige selbst: Die
    neue Farbe laeuft vom Schaft zur Spitze durch den Pfeil. Ein Farbwechsel
    ist dadurch auch aus dem Augenwinkel als Bewegung wahrnehmbar, waehrend ein
    schlagartiger Wechsel leicht uebersehen wird.

    Warum in JEDEM Durchlauf und nicht nur in Sendepausen: Genau das war die
    erste Fassung, und sie ist im Fahrversuch durchgefallen. Solange
    ununterbrochen gelenkt wird, geht jedes Paket ans Fahren - es entsteht
    keine Pause, die Warteschlange verhungert, und der Pfeil bleibt auf halbem
    Weg stehen: Schaft schon in der neuen Farbe, Kopf noch in der alten, die
    Spitze oft gar nicht gezeichnet. Die Warteschlange ist auf fuenf Befehle
    begrenzt und fuellt sich nur bei einem Zustandswechsel; der Aufschlag
    beschraenkt sich also auf die vier Durchlaeufe nach dem Wechsel.
    """

    # Der Pfeil in Matrixkoordinaten, y=0 ist die Fahrtrichtung. Bewusst
    # grossflaechig: Durch die Kugelschale gesehen verschwimmen die einzelnen
    # Leuchtpunkte, eine duenne Linie waere aus zwei Metern nicht mehr als
    # Richtung erkennbar.
    #
    #   . . . # # . . .      Spitze
    #   . . # # # # . .
    #   . # # # # # # .
    #   # # # # # # # #      Widerhaken
    #   . . . # # . . .      Schaft
    #   . . . # # . . .
    #   . . . # # . . .
    #   . . . # # . . .
    #
    # Der Kopf laeuft ueber vier Baender von 8 auf 2 Pixel zu. Zwei und nicht
    # eines, weil die Matrix acht Spalten hat: Eine mittige Reihe gibt es auf
    # geradzahliger Breite nicht, ein einzelnes Pixel saesse sichtbar
    # ausserhalb der Achse. Durch die Kugelschale verlaufen zwei Punkte
    # ohnehin zu einer Spitze.
    #
    # Reihenfolge ist Absicht: Der Schaft steht zuerst, weil er die groesste
    # zusammenhaengende Flaeche ist. Von einem Farbwechsel geht nur das erste
    # Paket sofort raus, der Rest verteilt sich auf die naechsten Durchlaeufe -
    # so traegt schon das erste Paket den groessten Teil der neuen Farbe, und
    # die Farbe laeuft sichtbar bis in die Spitze durch.
    # Ausschliesslich gefuellte Rechtecke, auch fuer die einzeiligen Baender
    # des Pfeilkopfs. set_matrix_line erwartet seine Punkte in aufsteigender
    # Reihenfolge; nach einer Drehung um 180 Grad standen sie absteigend da,
    # und die betroffenen Zeilen wurden schlicht nicht gezeichnet - deshalb
    # fehlte am Ball die Pfeilspitze. Ein Rechteck laesst sich dagegen
    # zuverlaessig normalisieren (min/max in _senden), egal wie gedreht wurde.
    _FORM = (
        ("fuell", 3, 4, 4, 7),   # Schaft
        ("fuell", 0, 3, 7, 3),   # Widerhaken, volle Breite
        ("fuell", 1, 2, 6, 2),   # Kopf, 6 breit
        ("fuell", 2, 1, 5, 1),   # Kopf, 4 breit
        ("fuell", 3, 0, 4, 0),   # Spitze, 2 breit
    )

    # Wie oft ein einzelner Zeichenbefehl wiederholt wird, bevor er aufgegeben
    # wird. Ein belegter Funkkanal ist der Normalfall und darf nicht dazu
    # fuehren, dass eine Linie des Pfeils fehlt; ein Befehl, den dieser Ball
    # gar nicht kennt, darf die Warteschlange aber auch nicht ewig blockieren.
    MAX_VERSUCHE = 3

    def __init__(self, api, guard):
        self._api      = api
        self._guard    = guard
        self._offen    = []    # noch zu sendende Zeichenbefehle
        self._versuche = 0     # Fehlversuche am vordersten Befehl
        self._farbe    = MATRIX_PFEIL_FARBEN["bereit"]
        self.pakete    = 0

    @staticmethod
    def _drehen(x, y, viertel):
        """Einen Punkt um Vielfache von 90 Grad im Uhrzeigersinn drehen."""
        for _ in range(viertel):
            x, y = 7 - y, x
        return x, y

    def faerben(self, r, g, b):
        """
        Den Pfeil in einer neuen Farbe anfordern.

        Die Pixel bleiben dieselben, nur ihre Farbe aendert sich - es muss also
        nichts geloescht werden. Faellt ein Paket aus, steht der Pfeil kurz
        zweifarbig da, verschwindet aber nie.
        """
        if not MATRIX_PFEIL_AN:
            self.aus()
            return
        self._farbe    = (int(r), int(g), int(b))
        self._offen    = list(self._FORM)
        self._versuche = 0

    def loeschen(self):
        """Matrix dunkel schalten - beim Abschalten der Steuerung."""
        self._offen    = [("leeren", 0, 0, 0, 0)]
        self._versuche = 0

    def aus(self):
        """Nichts mehr zeichnen - beim Abschalten und beim Ausrichten."""
        self._offen    = []
        self._versuche = 0

    @property
    def offen(self) -> int:
        return len(self._offen)

    def schritt(self) -> bool:
        """
        Hoechstens einen Zeichenbefehl senden.

        Rueckgabe: True, wenn ein Paket rausging. Fehler werden bewusst
        geschluckt - ein nicht gezeichneter Pfeil ist eine fehlende Hilfe,
        kein Grund, eine laufende Fahrt zu stoeren.
        """
        if not self._offen or not MATRIX_PFEIL_AN:
            return False
        art, x1, y1, x2, y2 = self._offen[0]
        viertel = (int(MATRIX_PFEIL_DREHUNG) // 90) % 4
        x1, y1 = self._drehen(x1, y1, viertel)
        x2, y2 = self._drehen(x2, y2, viertel)
        farbe = Color(r=self._farbe[0], g=self._farbe[1], b=self._farbe[2])
        try:
            if art == "leeren":
                self._guard.call(self._api.clear_matrix)
            else:
                # Nach dem Drehen koennen die Ecken vertauscht sein; die API
                # erwartet die linke obere zuerst.
                self._guard.call(self._api.set_matrix_fill,
                                 min(x1, x2), min(y1, y2),
                                 max(x1, x2), max(y1, y2), farbe)
        except Exception:
            # Meist nur ein belegter Funkkanal. Denselben Befehl im naechsten
            # Durchlauf noch einmal versuchen, sonst fehlte dem Pfeil dauerhaft
            # ein Band. Erst nach mehreren Fehlversuchen aufgeben - sonst
            # blieben Baelle ohne LED-Matrix in dieser Warteschlange haengen.
            self._versuche += 1
            if self._versuche >= self.MAX_VERSUCHE:
                self._offen.pop(0)
                self._versuche = 0
            return False
        self._offen.pop(0)
        self._versuche = 0
        self.pakete += 1
        return True


class _SpheroDrive:
    """
    Fahrbefehle an den Sphero – ein Befehl, ein BLE-Paket, und nur wenn er etwas
    ändert.

    Warum diese Schicht überhaupt existiert. SpheroEduAPI.roll(kurs, tempo,
    dauer) ist bequem, aber für eine laufende Steuerung falsch gebaut
    (spherov2/sphero_edu.py):

        roll_start(...)  ->  time.sleep(dauer)  ->  stop_roll()

    Jeder Aufruf hält den Ball am Ende also wieder an. Bei einer Schleife, die
    zehnmal je Sekunde einen Fahrbefehl schickt, heißt das: fahren, stoppen,
    fahren, stoppen. Genau daher kam das Ruckeln – es war kein Funkproblem,
    sondern der Ball hat getan, was ihm gesagt wurde.

    Dazu kommt der Preis in Zeit. spherov2 wartet nach JEDEM Paket
    cmd_safe_interval ab (BOLT: 75 ms) und blockiert zusätzlich, bis der Ball
    das Paket quittiert hat. Zwei Pakete je Durchlauf ergaben gemessen 220 bis
    330 ms – die Steuerung lief also mit 3 bis 4,5 Hz statt der eingestellten
    10 Hz, und weil MAX_TURN_ANGLE früher je DURCHLAUF galt, hing die
    Drehgeschwindigkeit unmittelbar an dieser schwankenden Rate.

    Diese Klasse geht deshalb eine Ebene tiefer, auf ToyUtil.roll_start: ein
    einziges drive_with_heading-Paket, das Kurs und Tempo gemeinsam setzt und
    den Ball weiterfahren lässt, bis etwas anderes gesagt wird. Angehalten wird
    nur noch, wenn wirklich angehalten werden soll.

    Zusätzlich wird ein Befehl übersprungen, der nichts ändert (gleicher Kurs,
    gleiches Tempo). Das kostet keine Reaktionsschnelligkeit – ändert sich die
    Handhaltung, ändert sich sofort Kurs oder Tempo und das Paket geht raus –,
    hält den Funkkanal aber frei für die Befehle, die tatsächlich etwas tun.

    Der interne Zustand von SpheroEduAPI (__heading/__speed) wird dabei
    mitgeführt. Sonst hätten die Stellen, die weiterhin die normale API
    benutzen – Ausrichten, Rückwärtsfahrt, Abschalten – einen veralteten Stand
    und würden mit einem falschen Kurs oder Tempo weiterarbeiten.
    """

    def __init__(self, api, guard):
        self._api      = api
        self._guard    = guard
        self._toy      = getattr(api, "_SpheroEduAPI__toy", None)
        self._last     = None    # zuletzt gesendetes (kurs, tempo)
        self._last_t   = 0.0
        self.pakete    = 0       # gesendete Fahrpakete (für die Taktanzeige)
        self.uebersprungen = 0

    def _roh_senden(self, kurs: int, tempo: int):
        """Ein Fahrpaket, ohne Änderungsprüfung."""
        if self._toy is None:
            # Ältere oder abweichende spherov2-Fassung: kein Zugriff auf die
            # Toy-Instanz. Dann über die öffentliche API – das sind zwei Pakete
            # statt einem, funktioniert aber. Lieber langsamer als gar nicht.
            self._guard.call(self._api.set_heading, kurs)
            self._guard.call(self._api.set_speed, tempo)
            return
        # Den mitgeführten Zustand der API gleich mitschreiben, damit
        # stop_roll() und die Rückwärtsfahrt denselben Kurs sehen.
        self._api._SpheroEduAPI__heading = kurs
        self._api._SpheroEduAPI__speed   = tempo
        self._guard.call(ToyUtil.roll_start, self._toy, kurs, tempo)

    def fahre(self, kurs, tempo) -> bool:
        """
        Fahrbefehl absetzen, sofern er etwas ändert.

        Rückgabe: True, wenn tatsächlich gesendet wurde.
        """
        kurs  = int(kurs) % 360
        tempo = int(tempo)
        jetzt = time.time()

        if self._last is not None:
            alt_kurs, alt_tempo = self._last
            # Kursdifferenz über den kürzeren Weg, damit 359 -> 1 als 2 Grad
            # zählt und nicht als 358.
            d_kurs  = abs((kurs - alt_kurs + 180) % 360 - 180)
            d_tempo = abs(tempo - alt_tempo)
            frisch  = (jetzt - self._last_t) < DRIVE_KEEPALIVE_S
            if (d_kurs < DRIVE_MIN_HEADING_DELTA
                    and d_tempo < DRIVE_MIN_SPEED_DELTA and frisch):
                self.uebersprungen += 1
                return False

        self._roh_senden(kurs, tempo)
        self._last   = (kurs, tempo)
        self._last_t = jetzt
        self.pakete += 1
        return True

    def halte(self, kurs):
        """
        Anhalten. Anders als fahre() immer gesendet, nie übersprungen: Ein
        nicht ausgeführter Stopp ist der eine Fehler, den diese Steuerung sich
        nicht leisten darf.
        """
        kurs = int(kurs) % 360
        self._roh_senden(kurs, 0)
        self._last   = (kurs, 0)
        self._last_t = time.time()
        self.pakete += 1

    def vergiss(self):
        """
        Zwischenspeicher verwerfen – nach dem Ausrichten oder der
        Rückwärtsfahrt. Dort haben andere Stellen gefunkt, der Ball steht also
        möglicherweise nicht mehr auf dem hier vermerkten Stand. Ohne dieses
        Verwerfen könnte der nächste Fahrbefehl als "unverändert" durchfallen
        und der Ball bliebe stehen.
        """
        self._last = None


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


def aktuelle_fahrtrichtung() -> str:
    """
    Aktuelle Fahrtrichtung, wie sie die Steuerungsschleife gerade umsetzt.

    Gedacht für Anzeigen und Zusatzgeräte (Vibrationsgürtel), die dieselbe
    Richtung brauchen, ohne die Steuerungsschleife anzufassen. Die Funktion
    liest nur und sendet nichts – sie ist damit gefahrlos aus fremden Threads
    aufrufbar.

    Der Fahrzustand wird NICHT neu berechnet, sondern derselbe gelesen, den
    auch die Steuerungsschleife benutzt (beim Empfang der Sensordaten einmal
    bestimmt, siehe sensorlog). Früher rechneten beide Seiten getrennt – mit der
    Hysterese wäre das nicht mehr haltbar, weil jede Seite ihren eigenen
    Vorzustand mitführte und der Gürtel dann zeitweise eine andere Richtung
    meldete, als der Sphero tatsächlich fuhr.

    Die Reihenfolge der Sonderfälle bildet weiterhin die Schleife nach:
    Rückwärtsfahrt hat Vorrang, danach die Prüfung auf veraltete Sensordaten.
    """
    with data_lock:
        zustand        = latest_data["state"]
        backward_until = latest_data["backward_until"]
        last_update    = latest_data["last_update"]

    now = time.time()
    if now < backward_until:
        return "backward"
    if last_update == 0.0 or now - last_update > DATA_TIMEOUT:
        return "neutral"
    return zustand


# Der Gürtel wird erst durch das Häkchen in der Oberfläche gestartet; bis dahin
# existiert weder ein Thread noch eine Verbindung.
vibrationsguertel = guertel.VibrationBelt(
    richtung_quelle=aktuelle_fahrtrichtung,
    on_status=lambda text: set_status(text),
    on_event=lambda name, detail: recorder.log_event(name, detail),
)


def _takt_warten(schritt_start: float):
    """
    Den angestrebten Takt eines Schleifendurchlaufs einhalten.

    Gewartet wird nur die RESTZEIT bis ROLL_COMMAND_DURATION +
    CONTROL_LOOP_SLEEP, nicht diese Zeit zusätzlich zur geleisteten Arbeit.
    Der Unterschied ist wesentlich: Früher schlief die Schleife eine feste
    Pause ab, obwohl der Fahrbefehl davor schon 200 ms gedauert hatte – der
    Takt war damit weder eingestellt noch gemessen, sondern das zufällige
    Ergebnis aus Funklaufzeit plus Pause.

    Dauerte die Arbeit länger als der Takt, wird gar nicht gewartet: Der
    nächste Durchlauf soll dann sofort beginnen, damit die Steuerung der Hand
    so schnell wie möglich wieder folgt.
    """
    ziel = ROLL_COMMAND_DURATION + CONTROL_LOOP_SLEEP
    rest = ziel - (time.time() - schritt_start)
    if rest > 0:
        time.sleep(rest)


def control_sphero():
    global sphero_api

    MAX_RECONNECTS   = 5
    RECONNECT_DELAY  = 3.0   # Sekunden zwischen Reconnect-Versuchen
    reconnect_count  = 0
    sphero_heading   = 0     # Heading über Reconnects hinweg behalten
    # Bezugskurs der Positionssteuerung: der Kurs, von dem aus der Versatz aus
    # der Handhaltung gerechnet wird. Wird bei Geradeausfahrt und im Stillstand
    # nachgeführt. In der Ratensteuerung ohne Bedeutung.
    kurs_referenz    = 0

    # Ein einzelner fehlgeschlagener Befehl bedeutet nicht zwangsläufig einen
    # echten Verbindungsabbruch – z.B. kann ein Zusammenstoß (Wand, Hindernis)
    # die Sphero-Firmware kurz blockieren, sodass ein einzelner Befehl einen
    # Timeout wirft, obwohl die BLE-Verbindung Sekundenbruchteile später
    # wieder normal reagiert. Erst nach mehreren Fehlversuchen in Folge gilt
    # die Verbindung als wirklich verloren (→ voller Reconnect-Zyklus).
    MAX_CONSECUTIVE_FAILURES = 3

    # Für Überlastfälle (_BefehlUeberlastet) gilt eine eigene, deutlich höhere
    # Grenze. Ein zu langsamer Durchlauf ist kein Verbindungsabbruch: Der Ball
    # fährt mit dem letzten Befehl weiter, und der nächste Durchlauf kommt in
    # 100 ms. Bei acht in Folge steht die Verbindung aber offensichtlich doch
    # nicht mehr – dann greift der normale Reconnect.
    MAX_CONSECUTIVE_OVERLOAD = 8

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

        # Sicherheitsabstand zwischen zwei BLE-Paketen herabsetzen. spherov2
        # schläft diese Zeit im Sendethread nach jedem Paket ab; der
        # Vorgabewert des BOLT (75 ms) war der größte Einzelposten in der
        # Durchlaufzeit. Gesetzt wird nur auf DIESER Toy-Instanz – die
        # Klassenvorgabe der Bibliothek bleibt unangetastet.
        try:
            toy.toy_type = toy.toy_type._replace(
                cmd_safe_interval=SPHERO_CMD_INTERVAL_S)
        except Exception as e:
            print(f"[WARN] Paketabstand nicht setzbar, Vorgabe bleibt: {e}")

        # ── Steuerungs-Loop ───────────────────────────────────────────────────
        connection_lost = False
        try:
            with _SpheroAPIFastClose(toy) as sphero:
                sphero_api           = sphero
                last_move_time       = time.time()
                is_stopped           = True
                consecutive_failures = 0
                consecutive_overload = 0
                was_backward         = False
                guard                = _SpheroCommandGuard()
                drive                = _SpheroDrive(sphero, guard)
                pfeil                = _MatrixPfeil(sphero, guard)
                current_led          = None
                # Zeitpunkt des vorigen Lenkschritts – Grundlage der
                # zeitbasierten Drehrate (siehe LENK_BEZUGSTAKT_S).
                letzter_schritt      = time.time()
                # Bereits verbrauchtes Dreh-Budget der laufenden Handdrehung,
                # mit Vorzeichen (siehe LENK_MAX_DREHUNG). Wird zurückgesetzt,
                # sobald die Hand die Drehstellung verlässt.
                dreh_budget          = 0.0
                budget_erschoepft    = False
                # Messung der tatsächlich erreichten Schleifenfrequenz. Sie war
                # bisher unbekannt und ist genau die Größe, an der die Lenkung
                # hing – deshalb wird sie angezeigt und protokolliert.
                takt_mittel          = 0.0
                takt_letzte_meldung  = time.time()

                def set_led(r, g, b):
                    # Nur bei Farbwechsel senden – vorher wurde dieselbe Farbe
                    # bei jedem Schleifendurchlauf erneut gefunkt und hat den
                    # BLE-Verkehr unnötig aufgebläht.
                    nonlocal current_led
                    if current_led == (r, g, b):
                        return
                    current_led = (r, g, b)

                    if not MATRIX_PFEIL_AN:
                        # Pfeil abgeschaltet: das frühere Verhalten – die ganze
                        # Matrix leuchtet in der Zustandsfarbe. Bleibt als
                        # Vergleichsmöglichkeit erhalten, damit sich beide
                        # Darstellungen im Fahrversuch gegeneinander stellen
                        # lassen, ohne den Code zu ändern.
                        guard.call(sphero.set_main_led, Color(r=r, g=g, b=b))
                        pfeil.aus()
                        return

                    if (r, g, b) == (0, 0, 0):
                        pfeil.loeschen()
                    else:
                        pfeil.faerben(r, g, b)
                    # Genau EIN Paket sofort – so viel hat der Farbwechsel auch
                    # gekostet, als noch die ganze Matrix gefüllt wurde. Der
                    # Rest des Pfeils füllt Sendepausen und hält die Steuerung
                    # damit nicht auf.
                    try:
                        pfeil.schritt()
                    except Exception:
                        pass

                guard.call(sphero.set_heading, sphero_heading)
                set_led(*MATRIX_PFEIL_FARBEN["bereit"])

                if reconnect_count == 0:
                    set_status("Sphero verbunden. Sensor-App kann Daten senden.")
                    recorder.log_event("sphero_connected", "")
                else:
                    set_status(f"Sphero wieder verbunden (Versuch {reconnect_count}).")
                    recorder.log_event("sphero_reconnected", f"Versuch {reconnect_count}")
                reconnect_count = 0  # bei Erfolg zurücksetzen

                while not stop_sphero.is_set():
                    schritt_start = time.time()
                    # Pfeil wurde im Fenster "Fahrverhalten" verstellt.
                    # Bei einer geaenderten Drehung liegt der alte Pfeil noch
                    # an seiner alten Stelle - er muss geloescht werden, sonst
                    # stehen zwei Pfeile gleichzeitig auf der Matrix. Ein
                    # blosses Neufaerben genuegt hier also nicht.
                    if pfeil_neu_zeichnen.is_set():
                        pfeil_neu_zeichnen.clear()
                        farbe_jetzt = current_led or MATRIX_PFEIL_FARBEN["bereit"]
                        current_led = None      # erzwingt echtes Neusenden
                        try:
                            pfeil.loeschen()
                            pfeil.schritt()     # Loeschen sofort ausfuehren
                        except Exception:
                            pass
                        try:
                            set_led(*farbe_jetzt)
                        except Exception:
                            pass

                    # ── Ausrichten: Sphero an die Oberfläche abgeben ──────────
                    # Solange ausgerichtet wird, darf diese Schleife nichts
                    # senden. Zwei gleichzeitige Befehlsströme auf derselben
                    # BLE-Verbindung lassen den Ball zwischen Ausricht- und
                    # Fahrbefehl hin- und herzucken.
                    if aim_request.is_set():
                        try:
                            guard.call(sphero.stop_roll, int(sphero_heading))
                            guard.call(sphero.set_back_led, 255)
                        except Exception:
                            pass
                        is_stopped  = True
                        current_led = None      # Farbe nach dem Ausrichten neu setzen
                        # Der halb gezeichnete Pfeil gehoert zum alten Bild und
                        # wird nach dem Ausrichten ohnehin neu angefordert.
                        pfeil.aus()
                        aim_active.set()
                        while aim_request.is_set() and not stop_sphero.is_set():
                            time.sleep(0.05)
                        aim_active.clear()
                        if aim_heading_reset.is_set():
                            aim_heading_reset.clear()
                            # Die eben festgelegte Richtung IST jetzt 0°.
                            sphero_heading = 0
                            kurs_referenz  = 0
                        try:
                            guard.call(sphero.set_back_led, 0)
                            guard.call(sphero.set_heading, sphero_heading)
                        except Exception:
                            pass
                        # Während des Ausrichtens hat die Oberfläche gefunkt –
                        # der zwischengespeicherte Fahrbefehl gilt nicht mehr.
                        drive.vergiss()
                        last_move_time  = time.time()
                        letzter_schritt = time.time()
                        dreh_budget     = 0.0
                        continue

                    with data_lock:
                        gx             = latest_data["gx"]
                        gy             = latest_data["gy"]
                        gz             = latest_data["gz"]
                        gemessener_zustand = latest_data["state"]
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
                                drive.halte(sphero_heading)
                                set_led(*MATRIX_PFEIL_FARBEN["backward"])
                                time.sleep(0.3)
                                was_backward = True
                            backward_heading = (sphero_heading + 180) % 360
                            # Auch rückwärts über drive(): ein Paket, kein
                            # Stopp am Ende. Der Ball rollt gleichmäßig zurück,
                            # statt in der Rückwärtsfahrt zu stottern.
                            drive.fahre(backward_heading, BACKWARD_SPEED)
                            set_led(*MATRIX_PFEIL_FARBEN["backward"])
                            last_move_time = time.time()
                            is_stopped     = False
                            consecutive_failures = 0
                            consecutive_overload = 0
                        except _BefehlUeberlastet as e:
                            consecutive_overload += 1
                            if consecutive_overload >= MAX_CONSECUTIVE_OVERLOAD:
                                connection_lost = True
                                break
                            print(f"[WARN] Rückwärts-Befehl zu langsam ({consecutive_overload}/{MAX_CONSECUTIVE_OVERLOAD}): {e}")
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
                        # Nach der Rückwärtsfahrt beginnt die Handsteuerung neu:
                        # weder ein altes Dreh-Budget noch die Zeit seit dem
                        # letzten Lenkschritt dürfen mitgeschleppt werden.
                        letzter_schritt = time.time()
                        dreh_budget     = 0.0
                        _takt_warten(schritt_start)
                        continue

                    if was_backward:
                        # Der Ball fuhr eben noch rückwärts; der gemerkte
                        # Fahrbefehl passt nicht mehr zur neuen Richtung.
                        drive.vergiss()
                    was_backward = False

                    # ── Ohne frische Sensordaten von Handy/Watch nichts fahren ──────
                    # (gx/gy/gz stehen sonst auf ihrem 0.0-Default, was get_state()
                    #  fälschlich als "forward" auswertet)
                    # Der Zustand wird beim Empfang der Sensordaten EINMAL
                    # bestimmt (mit Glättung und Hysterese, siehe sensorlog) und
                    # hier nur noch gelesen. Früher rechnete diese Schleife ihn
                    # neu – mit Hysterese ginge das nicht mehr auf, weil beide
                    # Seiten unterschiedliche Vorzustände hätten und der Sphero
                    # zeitweise anders führe, als Anzeige und Gürtel meldeten.
                    data_is_stale = (last_update == 0.0) or (time.time() - last_update > DATA_TIMEOUT)
                    state         = "neutral" if data_is_stale else gemessener_zustand
                    applied_speed = 0

                    # Tatsächlich vergangene Zeit seit dem letzten Lenkschritt.
                    # Sie ersetzt die frühere stillschweigende Annahme, jeder
                    # Durchlauf sei gleich lang – siehe LENK_BEZUGSTAKT_S.
                    dt              = schritt_start - letzter_schritt
                    letzter_schritt = schritt_start

                    try:
                        if state in ("right", "left"):
                            applied_speed = int(calc_speed(gx) * TURN_SPEED_FACTOR)
                            if STEUERMODUS == STEUERMODUS_POSITION:
                                # Kurs = Bezugskurs + Versatz aus der Handhaltung.
                                # Nichts wird aufaddiert: Dieselbe Handhaltung
                                # ergibt immer denselben Kurs, Hand zurück heißt
                                # Kurs zurück.
                                sphero_heading = (kurs_referenz
                                                  + calc_kursversatz(gy)) % 360
                                budget_erschoepft = False
                            else:
                                # Ratensteuerung: Drehung aufaddieren, skaliert
                                # mit der wirklich vergangenen Zeit.
                                # Tempo mitgeben, damit die Kurve bei aktiver
                                # Kopplung ihren Radius behält, statt bei
                                # langsamer Fahrt enger zu werden.
                                turn = calc_turn(gy, tempo=applied_speed, dt=dt)
                                if state == "left":
                                    turn = -turn

                                # Dreh-Budget: Der Stand der laufenden
                                # Handdrehung wird auf +/- LENK_MAX_DREHUNG
                                # geklemmt, und nur die tatsächlich erlaubte
                                # Differenz geht in den Kurs.
                                #
                                # Beidseitig geklemmt und nicht über den Betrag:
                                # Ist das Budget nach rechts aufgebraucht, muss
                                # eine Drehung nach LINKS trotzdem sofort
                                # wirken. Über den Betrag gerechnet wäre sie
                                # ebenfalls blockiert – die Testperson säße
                                # fest, obwohl sie gerade zurücklenken will.
                                if LENK_MAX_DREHUNG > 0:
                                    neu = max(-LENK_MAX_DREHUNG,
                                              min(LENK_MAX_DREHUNG, dreh_budget + turn))
                                    turn        = neu - dreh_budget
                                    dreh_budget = neu
                                    budget_erschoepft = (
                                        abs(dreh_budget) >= LENK_MAX_DREHUNG - 1e-6)
                                else:
                                    budget_erschoepft = False

                                sphero_heading = (sphero_heading + turn) % 360

                            drive.fahre(sphero_heading, applied_speed)
                            # Aufgebrauchtes Budget wird durch eine dunklere
                            # Kurvenfarbe angezeigt: Die Testperson sieht ohne
                            # Erklärung, dass weiteres Halten der Hand nichts
                            # mehr bewirkt und sie zurückdrehen muss.
                            # Aufgebrauchtes Budget wird gedimmt gezeigt:
                            # dieselbe Farbe, nur dunkler – die Richtung bleibt
                            # damit ablesbar, während die Testperson sieht, dass
                            # weiteres Halten nichts mehr bewirkt.
                            grund = MATRIX_PFEIL_FARBEN[
                                "right" if state == "right" else "left"]
                            if budget_erschoepft:
                                set_led(*(k // 3 for k in grund))
                            else:
                                set_led(*grund)
                            last_move_time = time.time()
                            is_stopped     = False
                            with data_lock:
                                latest_data["heading"] = sphero_heading

                        elif state == "forward":
                            # Geradeausfahrt rastet den erreichten Kurs ein: Er
                            # wird zum neuen Bezug für die nächste Drehung. Ohne
                            # dieses Nachführen bliebe die Positionssteuerung auf
                            # POS_MAX_OFFSET_* um die Nullrichtung eingesperrt,
                            # und der Sphero ließe sich nicht zurückholen.
                            kurs_referenz = sphero_heading
                            applied_speed = calc_speed(gx)
                            # Hand aus der Drehstellung zurück = neue Drehung,
                            # also volles Budget. Damit ist eine Wende in
                            # mehreren Zügen jederzeit möglich.
                            dreh_budget       = 0.0
                            budget_erschoepft = False
                            drive.fahre(sphero_heading, applied_speed)
                            set_led(*MATRIX_PFEIL_FARBEN["forward"])
                            last_move_time = time.time()
                            is_stopped     = False

                        elif state == "neutral":
                            # Auch das Anhalten rastet den Kurs ein, damit die
                            # nächste Drehung von der tatsächlichen Blickrichtung
                            # aus beginnt und nicht von einem alten Bezug.
                            kurs_referenz     = sphero_heading
                            dreh_budget       = 0.0
                            budget_erschoepft = False

                            # Tempo SOFORT auf 0, nicht erst nach STOP_TIME.
                            # Das ist wichtiger, als es aussieht: Bisher hat
                            # sphero.roll() am Ende jedes Aufrufs von sich aus
                            # einen Stopp geschickt, der Ball stand also schon,
                            # bevor STOP_TIME überhaupt zu laufen begann. Ohne
                            # diese Zeile würde er nach dem Wegfall des
                            # Auto-Stopps bis zu STOP_TIME (0,6 s) einfach
                            # weiterfahren – und genau dieser Zweig fängt auch
                            # den Fall ab, dass die Uhr keine Daten mehr liefert
                            # (data_is_stale). Anhalten muss dann sofort
                            # geschehen, nicht in einer halben Sekunde.
                            # Wiederholungen kostet das nichts: Derselbe Befehl
                            # wird von drive.fahre() übersprungen.
                            drive.fahre(sphero_heading, 0)

                            if not is_stopped and time.time() - last_move_time > STOP_TIME:
                                # Endgültiger Halt: erst jetzt gilt die Fahrt
                                # als beendet (rote Anzeige). STOP_TIME ist
                                # damit weiterhin das Halteband gegen Zittern
                                # um die Fahrschwelle, ohne die Bremswirkung
                                # zu verzögern.
                                drive.halte(sphero_heading)
                                is_stopped = True
                                set_led(*MATRIX_PFEIL_FARBEN["stop"])

                        consecutive_failures = 0
                        consecutive_overload = 0

                    except _BefehlUeberlastet as e:
                        # Nur dieser Durchlauf ist ausgefallen. Der Ball fährt
                        # mit dem letzten Befehl weiter, der nächste Versuch
                        # kommt in einem Takt – kein Grund, eine funktionierende
                        # Verbindung abzubauen (siehe _BefehlUeberlastet).
                        consecutive_overload += 1
                        if consecutive_overload >= MAX_CONSECUTIVE_OVERLOAD:
                            recorder.log_event(
                                "sphero_ueberlast",
                                f"{consecutive_overload} zu langsame Befehle in Folge")
                            connection_lost = True
                            break
                        print(f"[WARN] Sphero-Befehl zu langsam ({consecutive_overload}/{MAX_CONSECUTIVE_OVERLOAD}): {e}")

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

                    # ── Richtungspfeil in der Sendepause nachziehen ───────────
                    # Ein Zeichenbefehl je Durchlauf, bis der Pfeil fertig ist -
                    # bewusst auch dann, wenn in diesem Durchlauf schon ein
                    # Fahrpaket rausging. Die frühere Fassung zeichnete nur in
                    # Sendepausen und ließ den Pfeil deshalb genau dann
                    # unfertig, wenn ununterbrochen gelenkt wurde: Schaft neu
                    # eingefärbt, Kopf noch in der alten Farbe.
                    #
                    # Die Warteschlange ist fünf Befehle lang und füllt sich nur
                    # bei einem Zustandswechsel. Teurer wird es also nur in den
                    # vier Durchläufen nach einem Wechsel – und genau die zeigen
                    # den Farbverlauf durch den Pfeil.
                    #
                    # Ausnahme: Läuft die Steuerung ohnehin schon deutlich zu
                    # langsam, hat das Fahren Vorrang und die Anzeige wartet.
                    # Die Rückmeldung darf die Ursache nicht verschlimmern.
                    schleife_gesund = (takt_mittel <= 0.0
                                       or takt_mittel < 2.5 * ROLL_COMMAND_DURATION)
                    if pfeil.offen and schleife_gesund:
                        try:
                            pfeil.schritt()
                        except _BefehlUeberlastet:
                            pass

                    recorder.log_control(gx, gy, gz, state, sphero_heading, applied_speed, is_stopped)

                    # ── Takt messen und einhalten ─────────────────────────────
                    # Die erreichte Frequenz war bisher unbekannt und schwankte
                    # zwischen 2,5 und 7 Hz, obwohl 10 Hz eingestellt waren.
                    # Genau daran hing die Drehgeschwindigkeit. Jetzt wird sie
                    # gemessen, angezeigt und protokolliert – so ist belegbar,
                    # unter welchen Bedingungen eine Sitzung gefahren wurde.
                    dauer = time.time() - schritt_start
                    takt_mittel = dauer if takt_mittel <= 0 else takt_mittel + 0.2 * (dauer - takt_mittel)
                    with data_lock:
                        latest_data["loop_hz"] = (1.0 / takt_mittel) if takt_mittel > 1e-6 else 0.0
                    if time.time() - takt_letzte_meldung >= 10.0:
                        takt_letzte_meldung = time.time()
                        recorder.log_event(
                            "steuertakt",
                            f"{1.0/takt_mittel:.1f} Hz; {takt_mittel*1000:.0f} ms je Durchlauf; "
                            f"gesendet={drive.pakete}; uebersprungen={drive.uebersprungen}")

                    _takt_warten(schritt_start)

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

# ── Keypoint-Indizes des ZED-Skeletts BODY_34 ────────────────────────────────
# Namen statt Zahlen, weil die Zuordnung nicht offensichtlich ist und ein
# Zahlendreher hier unbemerkt seitenverkehrte Messwerte erzeugt. Die Werte
# stammen aus sl.BODY_34_PARTS des installierten SDK.
KP_PELVIS         = 0
KP_CHEST_SPINE    = 2
KP_NECK           = 3
KP_LEFT_SHOULDER  = 5
KP_LEFT_ELBOW     = 6
KP_LEFT_WRIST     = 7
KP_LEFT_HAND      = 8
KP_RIGHT_SHOULDER = 12
KP_RIGHT_ELBOW    = 13
KP_RIGHT_WRIST    = 14
KP_RIGHT_HAND     = 15
KP_LEFT_HIP       = 18
KP_RIGHT_HIP      = 22
KP_NOSE           = 27

_WICHTIGE_PUNKTE = {
    KP_PELVIS, KP_CHEST_SPINE, KP_NECK, KP_NOSE,
    KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST, KP_LEFT_HAND,
    KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST, KP_RIGHT_HAND,
    KP_LEFT_HIP, KP_RIGHT_HIP,
}
_HAND_KP     = {KP_LEFT_HAND, KP_RIGHT_HAND}
_ARM_KP      = {KP_LEFT_ELBOW, KP_RIGHT_ELBOW, KP_LEFT_WRIST, KP_RIGHT_WRIST}
_SHOULDER_KP = {KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER}
_TRUNK_KP    = {KP_PELVIS, KP_CHEST_SPINE, KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP}

_COLOR_HAND    = (0, 255, 0)
_COLOR_ARM     = (255, 165, 0)
_COLOR_BODY    = (0, 180, 255)
_COLOR_HEAD    = (255, 255, 0)
_COLOR_LINE    = (200, 200, 200)
_COLOR_GOOD    = (0, 255, 0)
_COLOR_WARNING = (0, 165, 255)
_COLOR_BAD     = (0, 0, 255)

# Anatomisch korrektes Teilskelett. Vorher liefen beide Schultern auf Punkt 11
# (RIGHT_CLAVICLE) zusammen, wodurch die linke Schulter am rechten Schlüsselbein
# hing; als "Hals" diente ebenfalls 11 statt NECK.
_BODY_CONNECTIONS = [
    (KP_NOSE, KP_NECK), (KP_NECK, KP_CHEST_SPINE),
    (KP_CHEST_SPINE, KP_PELVIS),
    (KP_NECK, KP_LEFT_SHOULDER), (KP_NECK, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW), (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_LEFT_WRIST, KP_LEFT_HAND),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_RIGHT_WRIST, KP_RIGHT_HAND),
    (KP_PELVIS, KP_LEFT_HIP), (KP_PELVIS, KP_RIGHT_HIP),
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


# ── Winkelmessung am Schultergelenk ──────────────────────────────────────────
# Gemessen wird der Elevationswinkel des Oberarms gegenüber dem RUMPF
# (thorakohumeraler Winkel im Sinne der ISB-Empfehlung, Wu et al. 2005):
#
#   Elevation   0° = Arm hängt am Körper
#              90° = Oberarm waagerecht
#             180° = Arm senkrecht über dem Kopf
#
#   Elevationsebene   0° = seitlich (Abduktion, Frontalebene)
#                    90° = nach vorne (Flexion, Sagittalebene)
#                   < 0° = nach hinten (Extension)
#
# Warum gegenüber dem Rumpf und nicht gegenüber der Senkrechten der Kamera:
# Beugt sich die Person vor oder lehnt sie sich zurück, ändert sich der Winkel
# zur Weltsenkrechten, ohne dass sich im Schultergelenk etwas bewegt hätte.
# Genau diese Rumpfausweichbewegung ist in der Schulterrehabilitation die
# typische Kompensation. Ein rumpfbezogener Winkel ist dagegen invariant
# dagegen und damit zwischen Personen und Sitzungen vergleichbar.
SHOULDER_TARGET_DEG = 90.0   # ab hier gilt die Hebung als vollständig (waagerecht)
SHOULDER_MID_DEG    = 45.0   # Zwischenstufe für die Farbrückmeldung


def _kp_valid(p) -> bool:
    """
    Prüft einen 3D-Keypoint auf Brauchbarkeit.

    Die ZED liefert für nicht erkannte Gelenke entweder exakt (0,0,0) oder NaN.
    Beides muss aussortiert werden – sonst entstehen Winkel aus Phantompunkten,
    die in der Auswertung wie echte Messwerte aussehen.
    """
    arr = np.asarray(p, dtype=float)[:3]
    return bool(np.all(np.isfinite(arr))) and not bool(np.allclose(arr, 0.0))


def _p3(kps_3d, idx):
    """Keypoint als 3D-Vektor, oder None wenn unbrauchbar."""
    if idx >= len(kps_3d):
        return None
    p = np.asarray(kps_3d[idx], dtype=float)[:3]
    return p if _kp_valid(p) else None


def _normiere(v):
    n = np.linalg.norm(v)
    return None if n < 1e-6 else v / n


def _berechne_winkel_3d(p1, p2, p3):
    """Winkel im Punkt p2 zwischen den Strecken p2→p1 und p2→p3 (Grad)."""
    a, b, c = (np.asarray(p, dtype=float)[:3] for p in (p1, p2, p3))
    if not (_kp_valid(a) and _kp_valid(b) and _kp_valid(c)):
        return None
    ba, bc = _normiere(a - b), _normiere(c - b)
    if ba is None or bc is None:
        return None
    return round(float(np.degrees(np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0)))), 1)


def _rumpf_koordinatensystem(kps_3d):
    """
    Baut ein rechtwinkliges Koordinatensystem aus dem Rumpf der Person.

    Rückgabe (up, right, forward) als Einheitsvektoren, oder None.
      up      Becken → Hals   (Längsachse des Rumpfes)
      right   zur rechten Körperseite der Person
      forward aus der Brust heraus nach vorne

    Die Achsen werden per Gram-Schmidt orthogonalisiert, weil die Schulterlinie
    nicht exakt senkrecht auf der Rumpfachse steht.

    `forward` wird aus der Nase abgeleitet statt über ein Kreuzprodukt: Das
    Vorzeichen eines Kreuzprodukts hängt an der Händigkeit des
    Kamerakoordinatensystems, und diese Anwendung setzt sl.COORDINATE_SYSTEM
    nicht explizit. Die Nase liegt anatomisch immer vor dem Hals – daraus ergibt
    sich die Blickrichtung eindeutig und unabhängig vom Koordinatensystem.
    """
    becken = _p3(kps_3d, KP_PELVIS)
    hals   = _p3(kps_3d, KP_NECK)
    ls     = _p3(kps_3d, KP_LEFT_SHOULDER)
    rs     = _p3(kps_3d, KP_RIGHT_SHOULDER)
    if becken is None or hals is None or ls is None or rs is None:
        return None

    up = _normiere(hals - becken)
    if up is None:
        return None

    seitlich = rs - ls                      # zeigt zur rechten Körperseite
    right = _normiere(seitlich - np.dot(seitlich, up) * up)
    if right is None:
        return None

    nase = _p3(kps_3d, KP_NOSE)
    if nase is not None:
        vorne = nase - hals
        forward = _normiere(vorne - np.dot(vorne, up) * up
                                  - np.dot(vorne, right) * right)
        if forward is not None:
            return up, right, forward

    # Ohne Nase: Kreuzprodukt als Rückfallebene. Das Vorzeichen ist dann
    # koordinatensystemabhängig, deshalb ist nur die Elevation verlässlich,
    # nicht das Vorzeichen der Elevationsebene.
    forward = _normiere(np.cross(right, up))
    return (up, right, forward) if forward is not None else None


def berechne_schulterwinkel(kps_3d, seite: str):
    """
    Elevationswinkel und Elevationsebene der Schulter (Grad).

    seite: "links" oder "rechts" – anatomische Seite der Person.
    Rückgabe (elevation, ebene) oder (None, None), wenn Punkte fehlen.
    """
    links = seite == "links"
    schulter_idx = KP_LEFT_SHOULDER if links else KP_RIGHT_SHOULDER
    ellbogen_idx = KP_LEFT_ELBOW    if links else KP_RIGHT_ELBOW

    schulter = _p3(kps_3d, schulter_idx)
    ellbogen = _p3(kps_3d, ellbogen_idx)
    system   = _rumpf_koordinatensystem(kps_3d)
    if schulter is None or ellbogen is None or system is None:
        return None, None

    up, right, forward = system
    oberarm = _normiere(ellbogen - schulter)
    if oberarm is None:
        return None, None

    # Elevation gegen die nach UNTEN gerichtete Rumpfachse: hängender Arm = 0°.
    elevation = float(np.degrees(np.arccos(np.clip(np.dot(oberarm, -up), -1.0, 1.0))))

    # Elevationsebene: Anteile des Oberarms quer zur Rumpfachse.
    # Für den linken Arm wird die Seitwärtsachse gespiegelt, damit "vom Körper
    # weg" auf beiden Seiten positiv ist und die Werte direkt vergleichbar sind.
    seit_vorzeichen = -1.0 if links else 1.0
    seitwaerts = float(np.dot(oberarm, right)) * seit_vorzeichen
    vorwaerts  = float(np.dot(oberarm, forward))
    ebene = float(np.degrees(np.arctan2(vorwaerts, seitwaerts)))

    return round(elevation, 1), round(ebene, 1)


def _winkel_farbe(w):
    if w >= SHOULDER_TARGET_DEG: return _COLOR_GOOD
    if w >= SHOULDER_MID_DEG:    return _COLOR_WARNING
    return _COLOR_BAD


def _winkel_text(w):
    if w >= SHOULDER_TARGET_DEG: return "Gut angehoben!"
    if w >= SHOULDER_MID_DEG:    return "Weiter anheben..."
    return "Arm anheben!"


def _ebene_text(ebene):
    """Benennt die Bewegungsebene für die Anzeige."""
    if ebene is None:
        return ""
    if -30 <= ebene <= 30:   return "seitlich"
    if 30 < ebene < 60:      return "schraeg vorne"
    if ebene >= 60:          return "vorne"
    return "nach hinten"


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


def _draw_schulterwinkel(frame, kps_2d, kps_3d, seite, cv2):
    """
    Zeichnet den Schulterwinkel ein und gibt (elevation, ebene, ellbogen) zurück.

    `seite` ist die anatomische Seite der Person ("links"/"rechts"), nicht die
    Bildseite. Eine im Bild rechts erscheinende Person hebt ihren LINKEN Arm.
    """
    h, w = frame.shape[:2]
    links = seite == "links"
    zeile = 80 if links else 120
    label = "Links " if links else "Rechts"

    schulter_idx = KP_LEFT_SHOULDER if links else KP_RIGHT_SHOULDER
    ellbogen_idx = KP_LEFT_ELBOW    if links else KP_RIGHT_ELBOW
    handgel_idx  = KP_LEFT_WRIST    if links else KP_RIGHT_WRIST

    elevation, ebene = berechne_schulterwinkel(kps_3d, seite)
    # Ellbogenwinkel weiterhin mitgemessen: er zeigt, ob der Arm beim Anheben
    # gestreckt bleibt, und ist damit die Qualitätskontrolle zur Elevation.
    ellbogen = _berechne_winkel_3d(kps_3d[schulter_idx], kps_3d[ellbogen_idx],
                                   kps_3d[handgel_idx]) \
        if max(schulter_idx, ellbogen_idx, handgel_idx) < len(kps_3d) else None

    if elevation is None:
        cv2.putText(frame, f"{label}: Schulter nicht sichtbar", (20, zeile),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return None, None, ellbogen

    farbe = _winkel_farbe(elevation)

    # Markierung am Schultergelenk – dort wird der Winkel gemessen.
    if schulter_idx < len(kps_2d):
        sx, sy = int(kps_2d[schulter_idx][0]), int(kps_2d[schulter_idx][1])
        if 0 < sx < w and 0 < sy < h:
            cv2.circle(frame, (sx, sy), 14, farbe, -1)
            cv2.circle(frame, (sx, sy), 14, (255, 255, 255), 2)
            cv2.putText(frame, f"{elevation:.0f}", (sx - 20, sy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    text = (f"{label}: {elevation:5.1f} Grad ({_ebene_text(ebene)}) _ "
            f"{_winkel_text(elevation)}")
    if ellbogen is not None:
        text += f"  [Ellbogen {ellbogen:.0f}]"
    cv2.putText(frame, text, (20, zeile),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)
    return elevation, ebene, ellbogen


def _draw_abstand(frame, kps_3d, cv2):
    global _last_distance_condition
    if KP_CHEST_SPINE >= len(kps_3d):
        return None
    p = kps_3d[KP_CHEST_SPINE]
    if not _kp_valid(p) or p[2] <= 0:
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

                    elev_l, plane_l, elb_l = _draw_schulterwinkel(
                        frame, kps_2d, kps_3d, "links", cv2)
                    elev_r, plane_r, elb_r = _draw_schulterwinkel(
                        frame, kps_2d, kps_3d, "rechts", cv2)

                    recorder.log_tracking(body.id, distance,
                                          elev_l, elev_r, plane_l, plane_r,
                                          elb_l, elb_r)

                    head = kps_2d[KP_NOSE]
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
# Standardaufgabe: Parcours mit Rückwärtsstrecke
# ─────────────────────────────────────────────────────────────────────────────

# Wortlaut, der jeder Testperson VORGELESEN wird. Bewusst als fester Text und
# nicht als freie Erklärung: Unterschiedlich formulierte Anweisungen sind eine
# der häufigsten Fehlerquellen in Bewegungsstudien, weil schon ein "möglichst
# schnell" gegenüber "in Ruhe" die Strategie und damit die Messgröße verändert.
#
# Die Formulierung vermeidet deshalb eine Aufforderung zur Eile: Gemessen wird
# die Zeit, aber die Person soll nicht auf Tempo optimieren, sondern die Aufgabe
# sauber lösen. Sonst misst man Risikobereitschaft statt Steuerfähigkeit – und
# die unterscheidet sich zwischen Altersgruppen systematisch.
AUFGABE_ANLEITUNG = (
    "Vor Ihnen liegen drei Markierungen auf dem Boden.\n"
    "\n"
    "Ihre Aufgabe:\n"
    "  1.  Fahren Sie den Ball von Markierung 1 zu Markierung 2.\n"
    "  2.  Umrunden Sie Markierung 2 einmal vollständig.\n"
    "  3.  Fahren Sie zurück zu Markierung 1.\n"
    "  4.  Umrunden Sie auch diese einmal vollständig.\n"
    "  5.  Fahren Sie den Ball zum Schluss rückwärts geradeaus von\n"
    "      Markierung 1 zu Markierung 3. Rückwärts fahren Sie mit einem\n"
    "      Doppeltipp auf die Uhr. Lenken können Sie dabei nicht – richten\n"
    "      Sie den Ball also vorher aus.\n"
    "\n"
    "Danach ist der Durchgang beendet.\n"
    "\n"
    "Nehmen Sie sich die Zeit, die Sie brauchen. Es geht nicht darum, schnell "
    "zu sein, sondern darum, den Ball sicher zu führen. Wenn etwas nicht "
    "klappt, fahren Sie einfach weiter."
)

AUFGABE_NAME = "Parcours mit Rückwärtsstrecke"

# Der Durchgang wird in Abschnitte zerlegt, deren Grenzen die betreuende Person
# per Knopfdruck setzt. Grund: Die Gesamtzeit allein vermischt Vorwärts- und
# Rückwärtsanteil, und genau deren Gegenüberstellung ist die Zielgröße.
#
# Warum die Rückwärtsstrecke gerade verläuft und nicht um eine Markierung
# herumführt: Während der Rückwärtsfahrt ist die Lenkung gesperrt
# (Rückwärtsfahrt hat Vorrang vor der Gravity-Steuerung). Eine Kurve rückwärts
# wäre damit nicht lösbar, sondern nur frustrierend. Eine gerade Strecke
# entspricht zudem dem etablierten Backward-Walk-Test, der ebenfalls geradlinig
# und ohne Drehung gemessen wird.
#
# Aufbau: Markierung 3 liegt auf der Verlängerung der Linie 2–1 jenseits von
# Markierung 1. So steht Markierung 2 der Rückwärtsstrecke nicht im Weg.
#
#   3 <···· rueckwaerts ····  1  ──── hinweg ────>  2
#
# (schluessel, ueberschrift, was die betreuende Person tut)
AUFGABE_ABSCHNITTE = (
    ("hinweg",
     "① Hinweg vorwärts – Markierung 1 → 2",
     "„Abschnitt beendet“ drücken, sobald der Ball Markierung 2 erreicht."),
    ("rundkurs",
     "②–④ Umrunden, zurückfahren, umrunden",
     "„Abschnitt beendet“ drücken, sobald Markierung 1 vollständig umrundet ist."),
    ("rueckwaerts",
     "⑤ Rückwärtsstrecke – Markierung 1 → 3",
     "„Durchgang beenden“ drücken, sobald der Ball Markierung 3 erreicht."),
)

# Feste Eingewöhnungsdauer vor dem ersten Durchgang.
#
# Bewusst eine Zeitvorgabe und kein "bis Sie sich sicher fühlen": Wie lange
# jemand von sich aus übt, hängt mit Alter, Technikaffinität und Selbstvertrauen
# zusammen – also genau mit den Merkmalen, deren Wirkung untersucht werden soll.
# Selbstbestimmte Übungsdauer wäre damit keine Randbedingung, sondern eine
# unkontrollierte Störvariable im Zentrum der Fragestellung.
EINGEWOEHNUNG_S = 180

# Vorgabe für die Länge der Rückwärtsstrecke in Metern.
#
# Deutlich länger als der Markierungsabstand, und das aus einem messtechnischen
# Grund: Ein Doppeltipp fährt BACKWARD_DURATION Sekunden mit BACKWARD_SPEED –
# bei den Vorgabewerten rund einen Meter. Auf einer Ein-Meter-Strecke wäre der
# Abschnitt mit einem einzigen Tipp erledigt und die gemessene Zeit bestünde
# fast nur aus Reaktions- und Ausrichtzeit. Drei Meter verlangen mehrere Tipps
# mit Nachkorrektur und entsprechen der Streckenlänge des Backward-Walk-Tests.
RUECKWAERTS_STRECKE_M = 3.0


class AufgabeFenster:
    """
    Durchführung und Zeitnahme der Standardaufgabe.

    Warum es die Aufgabe überhaupt braucht: Freies Fahren liefert reichhaltige
    Rohdaten, aber keine Zielgröße. Erst eine für alle gleiche Aufgabe macht
    Sitzungen und Altersgruppen vergleichbar. Die Zeit ist dabei die
    Hauptmessgröße; die übrigen Kennzahlen entstehen nebenbei aus dem
    Fahrzustand und kosten die betreuende Person keinen zusätzlichen Handgriff.

    Mehrere Durchgänge je Sitzung sind vorgesehen und ausdrücklich erwünscht:
    Ein einzelner Versuch ist stark von Zufall und Eingewöhnung geprägt. Üblich
    ist, den ersten Durchgang als Übungsversuch zu werten und erst die
    folgenden auszuwerten – deshalb wird jeder Durchgang nummeriert, statt nur
    eine Bestzeit zu behalten.

    Der Durchgang enthält am Ende eine gerade RÜCKWÄRTSSTRECKE. Grund: Eine
    Aufgabe, die alle Altersgruppen mühelos lösen, zeigt keine Unterschiede –
    Altersunterschiede in der Bewegungssteuerung treten erst mit steigender
    Aufgabenkomplexität hervor. Rückwärtsbewegung ist dafür der belegt
    empfindlichste Zusatz: Sie verlangt eine Umkehrung der Steuerzuordnung, und
    hier zusätzlich Vorausplanung, weil während der Rückwärtsfahrt nicht
    gelenkt werden kann.

    Die Abschnitte werden getrennt gestoppt (AUFGABE_ABSCHNITTE), damit sich
    Vorwärts- und Rückwärtsanteil gegenüberstellen lassen. Die Summe der
    Vorwärtsabschnitte entspricht der früheren Gesamtzeit des Parcours, sodass
    bereits erhobene Durchgänge vergleichbar bleiben.

    Misslungene Durchgänge werden als ungültig MARKIERT, nicht gelöscht. Wer
    Fehlversuche verschwinden lässt, verzerrt die Stichprobe.
    """

    TAKT_MS = 100    # Abtastung des Fahrzustands für die Nebenkennzahlen

    def __init__(self, parent, teilnehmer_text):
        self.win = tk.Toplevel(parent)
        self.win.title(f"Standardaufgabe – {AUFGABE_NAME}")
        self.win.geometry("640x820")
        self.win.minsize(560, 560)
        self.win.transient(parent)

        self._teilnehmer_text = teilnehmer_text
        self._laeuft      = False
        self._job         = None
        # Die Nummer des zuletzt gefahrenen Durchgangs. Die eigentliche
        # Zaehlung liegt NICHT hier, sondern in den gespeicherten Daten der
        # Testperson (naechste_versuchsnummer): Zwischen zwei Durchgaengen wird
        # Pause gemacht und die Aufzeichnung beendet, ein Zaehler im Fenster
        # wuerde dabei jedes Mal wieder bei 1 anfangen.
        self._versuch     = 0
        self._ergebnisse  = []
        # Durchgaenge, die ohne laufende Aufzeichnung gefahren wurden. Sie
        # stehen in keiner CSV und wuerden beim Neuaufbau der Tabelle aus den
        # gespeicherten Daten verschwinden - fuer die betreuende Person sollen
        # sie bis zum Schliessen des Fensters aber sichtbar bleiben.
        self._ungespeichert = []
        self._ungespeichert_pid = None

        # Abschnittssteuerung innerhalb eines laufenden Durchgangs.
        self._abschnitt_i     = 0
        self._abschnitt_t0    = 0.0
        self._abschnitt_zeit  = {}   # schluessel -> Dauer in Sekunden
        self._abschnitt_rueck = {}   # schluessel -> Doppeltipps in diesem Abschnitt

        # Eingewöhnung
        self._eing_laeuft     = False
        self._eing_job        = None
        self._eing_ende_t     = 0.0
        self._eing_absolviert = False

        scroll = probanden._ScrollFrame(self.win, height=620)
        scroll.pack(fill="both", expand=True, padx=14, pady=(12, 0))
        self._scroll = scroll
        main = scroll.inner

        ttk.Label(main, text=AUFGABE_NAME,
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.person_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.person_var, foreground="#555",
                  wraplength=560, justify="left").pack(anchor="w", pady=(2, 8))

        # ── Skizze des Parcours ───────────────────────────────────────────────
        self._zeichne_parcours(main)

        # ── Aufbau ────────────────────────────────────────────────────────────
        # Der Abstand wandert in jede Ergebniszeile: Zeiten sind nur innerhalb
        # desselben Aufbaus vergleichbar, und "wir hatten doch immer 1 m" ist
        # nach Monaten keine belastbare Auskunft mehr.
        aufbau = ttk.Frame(main)
        aufbau.pack(fill="x", pady=(8, 0))
        ttk.Label(aufbau, text="Abstand Markierung 1–2 (m):",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        self.var_abstand = tk.StringVar(value="1.0")
        ttk.Entry(aufbau, textvariable=self.var_abstand,
                  width=6, justify="center").pack(side="left", padx=(8, 16))
        ttk.Label(aufbau, text="Rückwärtsstrecke 1–3 (m):",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        self.var_rw_strecke = tk.StringVar(value=f"{RUECKWAERTS_STRECKE_M:.1f}")
        ttk.Entry(aufbau, textvariable=self.var_rw_strecke,
                  width=6, justify="center").pack(side="left", padx=(8, 0))
        ttk.Label(main,
                  text="Beide Maße müssen über die GESAMTE Studie gleich bleiben – sonst "
                       "sind die Zeiten zwischen den Testpersonen nicht vergleichbar. Sie "
                       "werden mit jedem Durchgang gespeichert.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 0))
        ttk.Label(main,
                  text="Markierung 3 liegt auf der Verlängerung der Linie 2–1, hinter "
                       "Markierung 1 – damit Markierung 2 der Rückwärtsstrecke nicht im "
                       "Weg steht. Kürzer als 2 m sollte sie nicht sein: Ein Doppeltipp "
                       "trägt bei den Vorgabewerten schon rund einen Meter weit.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 0))
        self.rueck_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.rueck_var, foreground="#777",
                  wraplength=560, justify="left").pack(anchor="w", pady=(2, 0))

        # ── Anleitung zum Vorlesen ────────────────────────────────────────────
        ttk.Separator(main).pack(fill="x", pady=(12, 8))
        ttk.Label(main, text="Diesen Text der Testperson vorlesen:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="Wortlaut bitte nicht abwandeln – unterschiedlich "
                       "formulierte Anweisungen verändern die Strategie und damit "
                       "die gemessene Zeit.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))
        anleitung = tk.Text(main, height=13, wrap="word", relief="solid",
                            borderwidth=1, font=("Segoe UI", 10))
        anleitung.insert("1.0", AUFGABE_ANLEITUNG)
        anleitung.config(state="disabled")
        anleitung.pack(fill="x")

        # ── Eingewöhnung ──────────────────────────────────────────────────────
        # Steht vor der Zeitnahme, weil sie im Ablauf davor liegt und weil sie
        # sonst übersprungen wird.
        ttk.Separator(main).pack(fill="x", pady=(12, 8))
        ttk.Label(main, text=f"Eingewöhnung ({EINGEWOEHNUNG_S // 60}:"
                             f"{EINGEWOEHNUNG_S % 60:02d} min, vor dem ersten Durchgang)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="Freies Fahren zum Kennenlernen der Steuerung, ohne Aufgabe. Die "
                       "Dauer ist fest vorgegeben und für alle Testpersonen gleich: Wie "
                       "lange jemand von sich aus übt, hängt mit Alter und Selbstvertrauen "
                       "zusammen – bei freier Übungsdauer wüsste man am Ende nicht, ob "
                       "Unterschiede aus der Steuerfähigkeit oder aus ungleicher Übung "
                       "stammen.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 4))

        eing = ttk.Frame(main)
        eing.pack(fill="x", pady=(2, 0))
        self.eing_button = ttk.Button(eing, text="🕒  Eingewöhnung starten",
                                      command=self._eing_umschalten)
        self.eing_button.pack(side="left")
        self.eing_var = tk.StringVar(value=self._mmss(EINGEWOEHNUNG_S))
        ttk.Label(eing, textvariable=self.eing_var,
                  font=("Consolas", 20, "bold"),
                  foreground="#0a7").pack(side="left", padx=(12, 0))
        self.eing_status_var = tk.StringVar(value="noch nicht durchgeführt")
        ttk.Label(main, textvariable=self.eing_status_var, foreground="#777",
                  wraplength=560, justify="left").pack(anchor="w", pady=(2, 0))

        # ── Zeitnahme ─────────────────────────────────────────────────────────
        ttk.Separator(main).pack(fill="x", pady=(12, 8))
        ttk.Label(main, text="Zeitnahme",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="Vorgesehener Ablauf: 4 Durchgänge, mit derselben vorgelesenen "
                       "Anweisung. Durchgang 1 ist der Übungsdurchgang und wird bei der "
                       "Auswertung ausgeschlossen (gespeichert wird er trotzdem); "
                       "gewertet wird der Mittelwert der Durchgänge 2 bis 4. Mehr als "
                       "4 Durchgänge sind möglich – sie werden fortlaufend "
                       "weitergezählt und alle mitgewertet.\n"
                       "Die Zählung läuft über die gesamte Testperson, nicht über die "
                       "einzelne Aufzeichnung: Zwischen den Durchgängen darf die "
                       "Aufzeichnung für eine Pause beendet und später neu gestartet "
                       "werden.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 4))

        self.zeit_var = tk.StringVar(value="0.0 s")
        ttk.Label(main, textvariable=self.zeit_var,
                  font=("Consolas", 34, "bold"),
                  foreground="#06c").pack(anchor="w", pady=(4, 2))

        # Welcher Abschnitt gerade läuft und was ihn beendet. Ohne diese Anzeige
        # müsste die betreuende Person die Abschnittsfolge auswendig kennen –
        # ein verpasster Knopfdruck macht die Aufteilung des Durchgangs kaputt.
        self.abschnitt_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.abschnitt_var,
                  font=("Segoe UI", 11, "bold"), foreground="#c60",
                  wraplength=560, justify="left").pack(anchor="w")
        self.abschnitt_hinweis_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.abschnitt_hinweis_var, foreground="#555",
                  wraplength=560, justify="left").pack(anchor="w")

        self.status_var = tk.StringVar(value="Bereit für Durchgang 1.")
        ttk.Label(main, textvariable=self.status_var, wraplength=560,
                  justify="left").pack(anchor="w", pady=(4, 0))

        self.zwischen_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.zwischen_var, font=("Consolas", 9),
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 0))

        knoepfe = ttk.Frame(main)
        knoepfe.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(knoepfe, text="▶  Aufgabe starten",
                                       command=self._umschalten)
        self.start_button.pack(side="left")
        self.zwischen_button = ttk.Button(knoepfe, text="⏭  Abschnitt beendet",
                                          command=self._abschnitt_weiter,
                                          state="disabled")
        self.zwischen_button.pack(side="left", padx=(8, 0))
        self.verwerfen_button = ttk.Button(knoepfe, text="✖  Durchgang verwerfen",
                                           command=self._verwerfen,
                                           state="disabled")
        self.verwerfen_button.pack(side="left", padx=(8, 0))

        ttk.Label(main,
                  text="„Abschnitt beendet“ unterbricht die Messung nicht – die Gesamtzeit "
                       "läuft weiter. Der Knopf trennt nur den Vorwärts- vom "
                       "Rückwärtsanteil, und genau deren Verhältnis ist die Zielgröße.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", pady=(6, 0))

        # ── Ergebnisse dieser Sitzung ─────────────────────────────────────────
        ttk.Separator(main).pack(fill="x", pady=(12, 8))
        ttk.Label(main, text="Durchgänge dieser Testperson (alle Aufzeichnungen)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.tabelle_var = tk.StringVar(value="noch keine")
        ttk.Label(main, textvariable=self.tabelle_var, font=("Consolas", 10),
                  justify="left").pack(anchor="w", pady=(4, 8))

        fuss = ttk.Frame(self.win, padding=(14, 8))
        fuss.pack(fill="x")
        ttk.Button(fuss, text="Schliessen", command=self.close).pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._anzeige_auffrischen()

    # ── Skizze ────────────────────────────────────────────────────────────────

    def _zeichne_parcours(self, parent):
        """
        Schematische Darstellung statt Foto.

        Eine Zeichnung zeigt genau das, worauf es ankommt – Reihenfolge und
        Umrundungsrichtung – und bleibt richtig, egal wie der Raum aussieht.
        Ein Foto einer fremden Umgebung würde dagegen Details zeigen, die mit
        dem eigenen Aufbau nichts zu tun haben.
        """
        c = tk.Canvas(parent, width=560, height=240, background="white",
                      highlightthickness=1, highlightbackground="#ccc")
        c.pack(anchor="w", pady=(4, 0))

        drei = (72, 125)
        a    = (250, 125)     # Markierung 1 – Start und Wendepunkt
        b    = (452, 125)     # Markierung 2
        r_mark = 18
        r_bahn = 38

        # Markierungen
        for (x, y), name in ((drei, "3"), (a, "1"), (b, "2")):
            c.create_oval(x - r_mark, y - r_mark, x + r_mark, y + r_mark,
                          fill="#ffd9a0", outline="#c07000", width=2)
            c.create_text(x, y, text=name, font=("Segoe UI", 13, "bold"),
                          fill="#7a4400")
        c.create_text(drei[0], drei[1] + r_mark + 14, text="nur rueckwaerts",
                      font=("Segoe UI", 8), fill="#7a4400")

        # Hinweg oben, Rückweg unten – räumlich getrennt, damit die Richtung
        # der Umrundung erkennbar bleibt.
        c.create_line(a[0] + 30, a[1] - 44, b[0] - 30, b[1] - 44,
                      arrow="last", width=3, fill="#0a7", smooth=True)
        c.create_text((a[0] + b[0]) / 2, a[1] - 58, text="① hinfahren",
                      font=("Segoe UI", 9, "bold"), fill="#0a7")

        c.create_arc(b[0] - r_bahn, b[1] - r_bahn, b[0] + r_bahn, b[1] + r_bahn,
                     start=100, extent=-290, style="arc", width=3, outline="#c60")
        c.create_text(b[0], b[1] + r_bahn + 18, text="② umrunden",
                      font=("Segoe UI", 9, "bold"), fill="#c60")

        c.create_line(b[0] - 30, b[1] + 44, a[0] + 30, a[1] + 44,
                      arrow="last", width=3, fill="#06c", smooth=True)
        c.create_text((a[0] + b[0]) / 2, a[1] + 58, text="③ zurueckfahren",
                      font=("Segoe UI", 9, "bold"), fill="#06c")

        c.create_arc(a[0] - r_bahn, a[1] - r_bahn, a[0] + r_bahn, a[1] + r_bahn,
                     start=80, extent=290, style="arc", width=3, outline="#c60")
        c.create_text(a[0], a[1] + r_bahn + 18, text="④ umrunden",
                      font=("Segoe UI", 9, "bold"), fill="#c60")

        # Rückwärtsstrecke: auf der Achse selbst, gestrichelt und in einer
        # eigenen Farbe – sie unterscheidet sich nicht nur im Verlauf, sondern
        # in der Art der Fortbewegung von allem anderen.
        c.create_line(a[0] - r_bahn - 8, a[1], drei[0] + r_mark + 8, drei[1],
                      arrow="last", width=3, fill="#a0a", dash=(6, 4))
        c.create_text((drei[0] + a[0]) / 2, a[1] - 16,
                      text="⑤ rueckwaerts (Doppeltipp)",
                      font=("Segoe UI", 9, "bold"), fill="#a0a")
        c.create_text((drei[0] + a[0]) / 2, a[1] + 16,
                      text="gerade, ohne Lenken",
                      font=("Segoe UI", 8), fill="#a0a")

    # ── Ablauf ────────────────────────────────────────────────────────────────

    def _abstand_m(self) -> str:
        """Eingetragener Abstand der Markierungen 1 und 2 als Zahltext."""
        return self._meter(self.var_abstand)

    def _rw_strecke_m(self) -> str:
        """Eingetragene Länge der Rückwärtsstrecke als Zahltext."""
        return self._meter(self.var_rw_strecke)

    @staticmethod
    def _meter(var) -> str:
        """
        Meterangabe aus einem Eingabefeld, "" wenn unlesbar.

        Deutsche Tastaturen liefern gern ein Komma – das wird akzeptiert.
        Unlesbares wird nicht verworfen, sondern als leeres Feld gespeichert
        und fällt so in der Auswertung sofort auf.
        """
        try:
            return f"{float(var.get().strip().replace(',', '.')):.2f}"
        except (TypeError, ValueError, AttributeError):
            return ""

    @staticmethod
    def _mmss(sekunden: float) -> str:
        s = max(0, int(round(sekunden)))
        return f"{s // 60}:{s % 60:02d}"

    @staticmethod
    def _tempo(strecke_m: str, dauer_s) -> str:
        """
        Tempo in m/s aus Streckentext und Dauer, "" wenn eines davon fehlt.

        Die Tempi sind der Grund, warum unterschiedlich lange Abschnitte
        überhaupt gegenübergestellt werden können: Ein Zeitverhältnis hinge an
        der gewählten Streckenlänge, ein Tempoverhältnis nicht.
        """
        try:
            strecke = float(strecke_m)
            dauer   = float(dauer_s)
        except (TypeError, ValueError):
            return ""
        if strecke <= 0 or dauer <= 0:
            return ""
        return f"{strecke / dauer:.3f}"

    # ── Eingewöhnung ──────────────────────────────────────────────────────────

    def _eing_umschalten(self):
        if self._eing_laeuft:
            # Vorzeitiger Abbruch wird als solcher festgehalten: Eine verkürzte
            # Eingewöhnung ist eine andere Vorbereitung und muss bei der
            # Auswertung erkennbar sein.
            self._eing_beenden(abgebrochen=True)
        else:
            self._eing_starten()

    def _eing_starten(self):
        start_server_once()
        self._eing_laeuft = True
        self._eing_ende_t = clock.t_abs() + EINGEWOEHNUNG_S
        self.eing_button.config(text="■  Eingewöhnung abbrechen")
        hinweis = ("" if recorder.active else
                   "  (keine Aufzeichnung aktiv – wird nicht protokolliert)")
        self.eing_status_var.set("läuft – freies Fahren ohne Aufgabe." + hinweis)
        recorder.log_event("eingewoehnung_start",
                           f"{AUFGABE_NAME}; Vorgabe {EINGEWOEHNUNG_S} s")
        self._eing_tick()

    def _eing_tick(self):
        if not self._eing_laeuft:
            return
        rest = self._eing_ende_t - clock.t_abs()
        if rest <= 0:
            self.eing_var.set("0:00")
            self._eing_beenden(abgebrochen=False)
            return
        self.eing_var.set(self._mmss(rest))
        self._eing_job = self.win.after(200, self._eing_tick)

    def _eing_beenden(self, abgebrochen: bool):
        self._eing_laeuft = False
        if self._eing_job is not None:
            try:
                self.win.after_cancel(self._eing_job)
            except Exception:
                pass
            self._eing_job = None

        gefahren = EINGEWOEHNUNG_S - max(0.0, self._eing_ende_t - clock.t_abs())
        self._eing_absolviert = not abgebrochen
        self.eing_button.config(text="🕒  Eingewöhnung starten")
        self.eing_var.set(self._mmss(EINGEWOEHNUNG_S))
        if abgebrochen:
            self.eing_status_var.set(
                f"ABGEBROCHEN nach {gefahren:.0f} s von {EINGEWOEHNUNG_S} s – "
                f"bitte in den Notizen vermerken.")
        else:
            self.eing_status_var.set(
                f"abgeschlossen ({EINGEWOEHNUNG_S} s). "
                f"Bereit für Durchgang {self._naechste_nr()}.")
        recorder.log_event(
            "eingewoehnung_ende",
            f"{gefahren:.1f} s von {EINGEWOEHNUNG_S} s; "
            f"{'ABGEBROCHEN' if abgebrochen else 'vollstaendig'}")

    # ── Durchgangsnummer und Verlauf ─────────────────────────────────────────

    def _naechste_nr(self) -> int:
        """
        Nummer des naechsten Durchgangs dieser Testperson.

        Massgeblich sind die bereits gespeicherten Durchgaenge, damit die
        Zaehlung ueber Aufzeichnungsgrenzen hinweg durchlaeuft. Der lokale
        Stand geht trotzdem mit ein: Wurde ohne Aufzeichnung gefahren, steht
        der Durchgang in keiner Datei, soll die Anzeige aber trotzdem
        weiterzaehlen statt dieselbe Nummer noch einmal anzubieten.
        """
        pid = recorder.participant_id
        gespeichert = naechste_versuchsnummer(pid) if pid else 1
        return max(gespeichert, self._versuch + 1)

    def _verlauf_laden(self):
        """
        Ergebnisliste aus den gespeicherten Durchgaengen der Person aufbauen.

        Die Tabelle zeigt damit alle Durchgaenge der Testperson, nicht nur die
        der laufenden Aufzeichnung - genau die Zusammenstellung, die spaeter
        auch ausgewertet wird (Durchgang 1 verwerfen, Rest mitteln).
        """
        pid = recorder.participant_id
        # Wird zwischendurch eine andere Testperson aufgezeichnet, gehoeren die
        # ungespeicherten Durchgaenge der vorigen Person nicht mehr hierher.
        if pid != self._ungespeichert_pid:
            self._ungespeichert = []
            self._ungespeichert_pid = pid

        eintraege = []
        for satz in durchgaenge_der_person(pid):
            nummer = (satz.get("versuch") or "").strip()
            if not nummer.isdigit():
                continue
            try:
                dauer = float(satz.get("dauer_s") or "nan")
            except ValueError:
                dauer = float("nan")
            eintraege.append((
                int(nummer), dauer,
                (satz.get("t_standard_s") or "").strip(),
                (satz.get("t_rueckwaerts_s") or "").strip(),
                (satz.get("ratio_rw_vw") or "").strip(),
                (satz.get("gueltig") or "").strip() == "1",
            ))
        self._ergebnisse = eintraege + self._ungespeichert

    # ── Durchgang ─────────────────────────────────────────────────────────────

    def _umschalten(self):
        if self._laeuft:
            self._beenden(gueltig=True)
        else:
            self._starten()

    def _starten(self):
        start_server_once()
        self._versuch  = self._naechste_nr()
        self._laeuft   = True
        self._t_start  = clock.t_abs()
        self._zwischen = []

        # Abschnitt 1 beginnt mit dem Durchgang.
        self._abschnitt_i     = 0
        self._abschnitt_t0    = self._t_start
        self._abschnitt_zeit  = {}
        self._abschnitt_rueck = {}
        self._rueck_bei_abschnitt = 0

        # Nebenkennzahlen: Zustand beim Start als Bezug, dann Wechsel zählen.
        with data_lock:
            self._letzter_zustand = latest_data["state"]
            self._letztes_backward = latest_data["backward_until"]
        self._halte = 0
        self._lenkwechsel = 0
        self._rueckwaerts = 0
        self._hr = []

        # Die Rückwärtsparameter werden beim Start eingefroren, nicht am Ende
        # ausgelesen: Sie sind über das Fahrverhalten-Fenster zur Laufzeit
        # verstellbar, und gespeichert gehört der Wert, unter dem tatsächlich
        # gefahren wurde.
        self._backward_dauer = BACKWARD_DURATION
        self._backward_speed = BACKWARD_SPEED

        self.start_button.config(text="■  Durchgang beenden")
        self.zwischen_button.config(state="normal")
        self.verwerfen_button.config(state="normal")
        self.zwischen_var.set("")
        hinweis = ("" if recorder.active else
                   "   ACHTUNG: keine Aufzeichnung aktiv – dieser Durchgang "
                   "wird NICHT gespeichert.")
        if not self._eing_absolviert:
            hinweis += ("   HINWEIS: Eingewöhnung wurde nicht vollständig "
                        "durchgeführt.")
        self.status_var.set(f"Durchgang {self._versuch} läuft.{hinweis}")
        self._abschnitt_anzeigen()
        recorder.log_event("aufgabe_start",
                           f"{AUFGABE_NAME}; Durchgang {self._versuch}; "
                           f"Abstand {self._abstand_m() or '?'} m; "
                           f"Rueckwaertsstrecke {self._rw_strecke_m() or '?'} m; "
                           f"backward {self._backward_dauer} s / "
                           f"{self._backward_speed}")
        self._tick()

    # ── Abschnitte ────────────────────────────────────────────────────────────

    def _abschnitt_anzeigen(self):
        """Laufenden Abschnitt und die zugehörige Handlung im Fenster zeigen."""
        if not self._laeuft:
            self.abschnitt_var.set("")
            self.abschnitt_hinweis_var.set("")
            self.zwischen_button.config(text="⏭  Abschnitt beendet")
            return
        _, ueberschrift, handlung = AUFGABE_ABSCHNITTE[self._abschnitt_i]
        self.abschnitt_var.set(
            f"Abschnitt {self._abschnitt_i + 1} von {len(AUFGABE_ABSCHNITTE)}: "
            f"{ueberschrift}")
        self.abschnitt_hinweis_var.set(handlung)
        letzter = self._abschnitt_i >= len(AUFGABE_ABSCHNITTE) - 1
        # Im letzten Abschnitt gibt es nichts mehr weiterzuschalten – der
        # Durchgang endet dort. Ein aktiver Knopf würde nur zu einem
        # versehentlich übersprungenen Abschnitt einladen.
        self.zwischen_button.config(
            state="disabled" if letzter else "normal",
            text="⏭  Abschnitt beendet")

    def _abschnitt_weiter(self):
        """Laufenden Abschnitt abschließen und den nächsten beginnen."""
        if not self._laeuft:
            return
        if self._abschnitt_i >= len(AUFGABE_ABSCHNITTE) - 1:
            return
        self._abschnitt_abschliessen()
        self._abschnitt_i += 1
        self._abschnitt_anzeigen()

    def _abschnitt_abschliessen(self):
        """
        Zeit und Doppeltipps des laufenden Abschnitts festhalten.

        Die Doppeltipps werden je Abschnitt getrennt gezählt, weil sie dort
        Verschiedenes bedeuten: In den Vorwärtsabschnitten sind sie
        Fehlerkorrekturen, im Rückwärtsabschnitt das Fortbewegungsmittel. In
        einer gemeinsamen Summe wären beide nicht mehr auseinanderzuhalten.
        """
        schluessel = AUFGABE_ABSCHNITTE[self._abschnitt_i][0]
        jetzt = clock.t_abs()
        self._abschnitt_zeit[schluessel]  = jetzt - self._abschnitt_t0
        self._abschnitt_rueck[schluessel] = (self._rueckwaerts
                                             - self._rueck_bei_abschnitt)
        self._rueck_bei_abschnitt = self._rueckwaerts
        self._abschnitt_t0 = jetzt

        seit_start = jetzt - self._t_start
        self._zwischen.append(round(seit_start, 1))
        self.zwischen_var.set(
            "Abschnitte: " + "  ".join(
                f"{AUFGABE_ABSCHNITTE[i][0]} {self._abschnitt_zeit[k]:.1f}s"
                for i, (k, _, _) in enumerate(AUFGABE_ABSCHNITTE)
                if k in self._abschnitt_zeit))
        recorder.log_event(
            "aufgabe_abschnitt",
            f"Durchgang {self._versuch}; {schluessel}; "
            f"{self._abschnitt_zeit[schluessel]:.2f} s; "
            f"bei {seit_start:.2f} s; "
            f"Doppeltipps {self._abschnitt_rueck[schluessel]}")

    def _tick(self):
        if not self._laeuft:
            return
        verstrichen = clock.t_abs() - self._t_start
        im_abschnitt = clock.t_abs() - self._abschnitt_t0
        self.zeit_var.set(f"{verstrichen:.1f} s   ({im_abschnitt:.1f} s)")

        with data_lock:
            zustand  = latest_data["state"]
            backward = latest_data["backward_until"]
            hr       = latest_data["heart_rate"]

        # Halt = Übergang von einer Fahrbewegung in die Neutrallage.
        if zustand != self._letzter_zustand:
            if zustand == "neutral":
                self._halte += 1
            elif zustand in ("left", "right"):
                self._lenkwechsel += 1
            self._letzter_zustand = zustand

        # Jede Verlängerung des Rückwärtsfensters ist ein neuer Doppeltipp.
        if backward > self._letztes_backward:
            self._rueckwaerts += 1
        self._letztes_backward = backward

        if valid_hr(hr):
            self._hr.append(float(hr))

        self._job = self.win.after(self.TAKT_MS, self._tick)

    def _verwerfen(self):
        if not self._laeuft:
            return
        offen = AUFGABE_ABSCHNITTE[self._abschnitt_i][0]
        notiz = (f"verworfen (waehrend des Durchgangs abgebrochen, "
                 f"Abschnitt {offen})")
        self._beenden(gueltig=False, notiz=notiz)

    def _beenden(self, gueltig: bool, notiz: str = ""):
        self._laeuft_bis_ende = (self._abschnitt_i
                                 >= len(AUFGABE_ABSCHNITTE) - 1)
        # Den laufenden Abschnitt noch abschließen, sonst fehlte seine Zeit.
        # Das gilt auch für verworfene Durchgänge: Wo der Abbruch passiert ist,
        # gehört zur Begründung dazu.
        if self._laeuft:
            self._abschnitt_abschliessen()
        self._laeuft = False
        if self._job is not None:
            try:
                self.win.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

        t_ende  = clock.t_abs()
        dauer   = t_ende - self._t_start
        hr_mit  = round(sum(self._hr) / len(self._hr), 1) if self._hr else ""
        hr_max  = round(max(self._hr), 1) if self._hr else ""

        # Wird der Durchgang beendet, bevor alle Abschnitte durchlaufen sind,
        # bleiben deren Felder leer statt auf 0 zu stehen: Eine 0 wäre in der
        # Auswertung eine Zahl wie jede andere, ein leeres Feld ist erkennbar
        # ein fehlender Wert.
        if not self._laeuft_bis_ende and not notiz:
            offen = AUFGABE_ABSCHNITTE[self._abschnitt_i][0]
            notiz = f"vorzeitig beendet (letzter Abschnitt: {offen})"

        def zeit(schluessel):
            wert = self._abschnitt_zeit.get(schluessel)
            return f"{wert:.2f}" if wert is not None else ""

        def tipps(schluessel):
            wert = self._abschnitt_rueck.get(schluessel)
            return wert if wert is not None else ""

        t_hin   = self._abschnitt_zeit.get("hinweg")
        t_rund  = self._abschnitt_zeit.get("rundkurs")
        t_rueck = self._abschnitt_zeit.get("rueckwaerts")
        # Summe der Vorwärtsabschnitte = die frühere Gesamtzeit des Parcours.
        t_standard = (f"{t_hin + t_rund:.2f}"
                      if t_hin is not None and t_rund is not None else "")

        abstand    = self._abstand_m()
        rw_strecke = self._rw_strecke_m()
        v_hin   = self._tempo(abstand, t_hin)
        v_rueck = self._tempo(rw_strecke, t_rueck)
        ratio   = (f"{float(v_rueck) / float(v_hin):.3f}"
                   if v_hin and v_rueck and float(v_hin) > 0 else "")

        korrektur = sum(self._abschnitt_rueck.get(k, 0)
                        for k in ("hinweg", "rundkurs"))

        zeile = {
            "versuch":               self._versuch,
            "abstand_m":             abstand,
            "rw_strecke_m":          rw_strecke,
            "backward_dauer_s":      f"{self._backward_dauer:.1f}",
            "backward_speed":        self._backward_speed,
            "t_rel_start_s":         f"{recorder.t_rel(self._t_start):.3f}" if recorder.active else "",
            "t_rel_ende_s":          f"{recorder.t_rel(t_ende):.3f}" if recorder.active else "",
            "dauer_s":               f"{dauer:.2f}",
            "t_hinweg_s":            zeit("hinweg"),
            "t_rundkurs_s":          zeit("rundkurs"),
            "t_standard_s":          t_standard,
            "t_rueckwaerts_s":       zeit("rueckwaerts"),
            "v_hinweg_ms":           v_hin,
            "v_rueckwaerts_ms":      v_rueck,
            "ratio_rw_vw":           ratio,
            "zwischenzeiten_s":      " ".join(f"{z:.1f}" for z in self._zwischen),
            "halte":                 self._halte,
            "lenkwechsel":           self._lenkwechsel,
            "rueckwaerts":           self._rueckwaerts,
            "rueckwaerts_korrektur": korrektur,
            "rueckwaerts_strecke":   tipps("rueckwaerts"),
            "hr_mittel":             hr_mit,
            "hr_max":                hr_max,
            "gueltig":               1 if gueltig else 0,
            "notiz":                 notiz,
        }
        recorder.log_aufgabe(zeile)
        recorder.log_event(
            "aufgabe_ende",
            f"Durchgang {self._versuch}; {dauer:.2f} s "
            f"(vorwaerts {t_standard or '?'} s, rueckwaerts "
            f"{zeit('rueckwaerts') or '?'} s, Ratio {ratio or '?'}); "
            f"{'gueltig' if gueltig else 'VERWORFEN'}; "
            f"Halte {self._halte}; Lenkeingaben {self._lenkwechsel}; "
            f"Rueckwaerts {self._rueckwaerts} "
            f"(Korrektur {korrektur}, Strecke {tipps('rueckwaerts') or 0})")

        # Bei laufender Aufzeichnung steht der Durchgang jetzt in der
        # Sammeltabelle der Person und wird von dort gelesen; ihn zusaetzlich
        # lokal zu merken, wuerde ihn in der Tabelle verdoppeln.
        if not recorder.active:
            self._ungespeichert_pid = recorder.participant_id
            self._ungespeichert.append((self._versuch, dauer, t_standard,
                                        zeit("rueckwaerts"), ratio, gueltig))
        self.zeit_var.set(f"{dauer:.1f} s")
        self.start_button.config(text="▶  Aufgabe starten")
        self.zwischen_button.config(state="disabled")
        self.verwerfen_button.config(state="disabled")
        self._abschnitt_anzeigen()

        gespeichert = "gespeichert" if recorder.active else "NICHT gespeichert (keine Aufzeichnung)"
        self.status_var.set(
            f"Durchgang {self._versuch} {'beendet' if gueltig else 'verworfen'}: "
            f"{dauer:.1f} s – {gespeichert}. "
            f"Bereit für Durchgang {self._naechste_nr()}.")
        self._anzeige_auffrischen()

    # ── Anzeige ───────────────────────────────────────────────────────────────

    def _anzeige_auffrischen(self):
        self.person_var.set(self._teilnehmer_text())
        # Die Rückwärtsparameter stehen sichtbar im Fenster, weil sie über das
        # Fahrverhalten-Fenster verstellbar sind und unmittelbar bestimmen, wie
        # weit ein Doppeltipp trägt. Ändern sie sich mitten in der Erhebung,
        # sind die Rückwärtszeiten nicht mehr vergleichbar.
        self.rueck_var.set(
            f"Rückwärtsfahrt zurzeit: {BACKWARD_DURATION:.1f} s je Doppeltipp "
            f"bei Tempo {BACKWARD_SPEED}. Über die gesamte Studie unverändert "
            f"lassen – wird mit jedem Durchgang gespeichert.")

        self._verlauf_laden()
        # Solange in diesem Fenster noch kein Durchgang gefahren wurde, zeigt
        # die Statuszeile die Nummer, die als naechstes vergeben wird. Bei
        # einer Person mit bereits gefahrenen Durchgaengen ist das nicht die 1
        # - und genau das muss die betreuende Person sehen, bevor sie startet.
        if not self._laeuft and self._versuch == 0:
            self.status_var.set(f"Bereit für Durchgang {self._naechste_nr()}.")

        if not self._ergebnisse:
            self.tabelle_var.set("noch keine")
            return
        zeilen = ["  Nr | gesamt  | vorwaerts | rueckw. | Ratio | gueltig",
                  "  ---+---------+-----------+---------+-------+--------"]
        gewertet = []
        for nr, dauer, t_std, t_rw, ratio, gueltig in self._ergebnisse:
            zeilen.append(
                f"  {nr:2d} | {dauer:6.1f}s | {(t_std or '-'):>8s}s | "
                f"{(t_rw or '-'):>6s}s | {(ratio or '-'):>5s} | "
                f"{'ja' if gueltig else 'NEIN'}")
            # Durchgang 1 ist der Übungsdurchgang und geht laut Protokoll nicht
            # in die Wertung ein – er wird gespeichert, aber hier nicht gemittelt.
            if gueltig and nr > 1:
                gewertet.append((dauer, ratio))
        if len(gewertet) >= 2:
            zeiten = [d for d, _ in gewertet]
            zeilen.append("")
            zeilen.append(f"  Mittel der gewerteten Durchgaenge (ab Nr. 2): "
                          f"{sum(zeiten) / len(zeiten):.1f} s  (n={len(zeiten)})")
            ratios = [float(r) for _, r in gewertet if r]
            if ratios:
                zeilen.append(f"  Mittleres Ratio rueckwaerts/vorwaerts: "
                              f"{sum(ratios) / len(ratios):.3f}")
        self.tabelle_var.set("\n".join(zeilen))

    def close(self):
        if self._laeuft:
            # Ein laufender Durchgang darf nicht stillschweigend verschwinden.
            self._beenden(gueltig=False, notiz="verworfen (Fenster geschlossen)")
        if self._eing_laeuft:
            self._eing_beenden(abgebrochen=True)
        self._scroll.unbind_mousewheel()
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Nullrichtung ausrichten
# ─────────────────────────────────────────────────────────────────────────────

class AusrichtFenster:
    """
    Legt fest, welche Richtung für den Sphero "vorwärts" bedeutet.

    Das Problem: Der Sphero kennt keine Weltrichtung, sondern nur seine eigene
    Nullrichtung. Die entsteht beim Verbinden aus der zufälligen Lage des Balls.
    Zeigt sie nicht zur steuernden Person, fährt er bei "vorwärts" schräg an ihr
    vorbei, und links/rechts stimmen nicht mit dem überein, was sie sieht.

    Warum das für die Studie zählt: Ohne festen Ausrichtschritt fährt jede
    Testperson eine geringfügig andere Zuordnung zwischen Handbewegung und
    beobachteter Fahrtrichtung – ein Störfaktor, der sich über die ganze
    Erhebung zieht und die Gruppen unvergleichbar macht.

    Bedienung wie in den offiziellen Sphero-Apps: Die blaue Rückleuchte wird
    eingeschaltet, der Ball per Tastendruck auf der Stelle gedreht, bis die
    Leuchte zur Person zeigt. Danach wird diese Richtung als 0° übernommen.

    Technisch wird nicht gefahren, sondern nur der Kurs gesetzt: Bei Tempo 0
    dreht sich der Ball an Ort und Stelle. Am Ende macht reset_aim() die
    aktuelle Blickrichtung zur neuen Null.
    """

    SCHRITTE   = (-45, -15, -5, 5, 15, 45)
    UEBERGABE_TIMEOUT_S = 3.0

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Sphero ausrichten")
        self.win.geometry("560x430")
        self.win.transient(parent)

        self._winkel     = 0      # bisher aufsummierte Drehung
        self._uebernommen = False

        rahmen = ttk.Frame(self.win, padding=16)
        rahmen.pack(fill="both", expand=True)

        ttk.Label(rahmen, text="Sphero ausrichten",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(rahmen,
                  text="Der Sphero dreht sich auf der Stelle. Drehen Sie ihn so, dass die "
                       "blaue Rückleuchte zur steuernden Person zeigt – diese Richtung "
                       "wird dann zu „vorwärts“.\n\n"
                       "Solange dieses Fenster offen ist, reagiert der Sphero nicht auf "
                       "die Uhr.",
                  foreground="#555", wraplength=500,
                  justify="left").pack(anchor="w", pady=(6, 12))

        self.status_var = tk.StringVar(value="Verbinde mit dem Sphero...")
        ttk.Label(rahmen, textvariable=self.status_var, wraplength=500,
                  justify="left").pack(anchor="w")

        self.winkel_var = tk.StringVar(value="Drehung: 0°")
        ttk.Label(rahmen, textvariable=self.winkel_var,
                  font=("Consolas", 20, "bold"),
                  foreground="#06c").pack(anchor="w", pady=(10, 10))

        # Tasten in der Reihenfolge, wie sich der Ball dreht: links herum links.
        tasten = ttk.Frame(rahmen)
        tasten.pack(fill="x")
        self._tasten = []
        for schritt in self.SCHRITTE:
            pfeil = "↺" if schritt < 0 else "↻"
            b = ttk.Button(tasten, text=f"{pfeil} {abs(schritt)}°",
                           width=8, command=lambda s=schritt: self._drehe(s))
            b.pack(side="left", padx=(0, 6))
            self._tasten.append(b)

        ttk.Label(rahmen,
                  text="↺ dreht gegen den Uhrzeigersinn, ↻ im Uhrzeigersinn "
                       "(von oben betrachtet).",
                  foreground="#777", wraplength=500,
                  justify="left").pack(anchor="w", pady=(8, 0))

        knoepfe = ttk.Frame(rahmen)
        knoepfe.pack(fill="x", side="bottom", pady=(16, 0))
        self.ok_button = ttk.Button(knoepfe, text="✔  Diese Richtung ist vorwärts",
                                    command=self._uebernehmen)
        self.ok_button.pack(side="left")
        ttk.Button(knoepfe, text="Abbrechen",
                   command=self.close).pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.after(50, self._uebernahme_anfordern)

    # ── Übergabe der Verbindung ───────────────────────────────────────────────

    def _uebernahme_anfordern(self):
        """Steuerungsschleife anhalten und warten, bis sie den Sphero freigibt."""
        if sphero_api is None:
            self._sperren("Der Sphero ist nicht verbunden. Bitte zuerst "
                          "„Sphero starten“ drücken.")
            return
        aim_request.set()
        if not aim_active.wait(self.UEBERGABE_TIMEOUT_S):
            aim_request.clear()
            self._sperren("Die Steuerung hat den Sphero nicht freigegeben. "
                          "Bitte erneut versuchen.")
            return
        self.status_var.set("Bereit. Die blaue Rückleuchte leuchtet.")
        recorder.log_event("ausrichten_start", "")

    def _sperren(self, text):
        self.status_var.set(text)
        for b in self._tasten:
            b.config(state="disabled")
        self.ok_button.config(state="disabled")

    # ── Drehen ────────────────────────────────────────────────────────────────

    def _drehe(self, schritt: int):
        if sphero_api is None or not aim_active.is_set():
            return
        self._winkel = (self._winkel + schritt) % 360
        try:
            # Tempo ist 0, deshalb dreht sich der Ball nur, statt zu fahren.
            sphero_api.set_heading(self._winkel)
        except Exception as e:
            self.status_var.set(f"Befehl fehlgeschlagen: {type(e).__name__}")
            return
        self.winkel_var.set(f"Drehung: {self._winkel}°")

    def _uebernehmen(self):
        if sphero_api is None or not aim_active.is_set():
            return
        try:
            # Macht die aktuelle Blickrichtung zur neuen Null. Danach ist der
            # mitgeführte Kurs der Steuerung ebenfalls 0 (aim_heading_reset).
            sphero_api.reset_aim()
            sphero_api.set_heading(0)
        except Exception as e:
            self.status_var.set(f"Übernehmen fehlgeschlagen: {type(e).__name__}")
            return
        self._uebernommen = True
        recorder.log_event("ausrichten_uebernommen", f"Drehung {self._winkel} Grad")
        set_status("Nullrichtung übernommen – „vorwärts“ zeigt jetzt zur Person.")
        self.close()

    def close(self):
        if aim_active.is_set() and not self._uebernommen:
            recorder.log_event("ausrichten_abgebrochen", f"Drehung {self._winkel} Grad")
        if self._uebernommen:
            aim_heading_reset.set()
        aim_request.clear()
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Ruhepuls messen
# ─────────────────────────────────────────────────────────────────────────────

class RuhepulsFenster:
    """
    Geführte Messung des Ruhepulses als Bezugsgröße für die Herzfrequenzzonen.

    Ablauf: Die Person sitzt ruhig, die Uhr sendet weiter ihre Werte. Über die
    Messdauer werden alle plausiblen Herzfrequenzen gesammelt; als Ergebnis
    dient der MEDIAN, nicht der Mittelwert. Ein einzelner Ausreißer – ein
    Bewegungsartefakt der optischen Messung, ein kurzer Schreck – verschiebt den
    Median kaum, den Mittelwert dagegen deutlich. Da der Ruhepuls anschließend
    die gesamte Zonenberechnung trägt, würde sich ein solcher Fehler auf jede
    Warnschwelle dieser Person durchschlagen.

    Die ersten Sekunden werden verworfen (EINSCHWINGEN_S): Direkt nach dem
    Hinsetzen ist der Puls noch erhöht, und die Uhr braucht einen Moment, bis
    ihre Messung stabil ist.
    """

    MESSDAUER_S    = 60      # Gesamtdauer der Messung
    EINSCHWINGEN_S = 15      # davon Vorlauf, dessen Werte verworfen werden
    TAKT_MS        = 500     # Abtastung der zuletzt empfangenen Herzfrequenz
    MIN_WERTE      = 10      # weniger Messpunkte gelten als nicht verwertbar

    def __init__(self, parent, teilnehmer_id: str, alter: int, fertig=None):
        self.win = tk.Toplevel(parent)
        self.win.title(f"Ruhepuls messen – {teilnehmer_id}")
        self.win.geometry("520x430")
        self.win.transient(parent)

        self.teilnehmer_id = teilnehmer_id
        self.alter    = alter
        self._fertig  = fertig or (lambda bpm: None)
        self._werte   = []
        self._läuft   = False
        self._job     = None
        self._start_t = 0.0
        self.ergebnis = None

        rahmen = ttk.Frame(self.win, padding=16)
        rahmen.pack(fill="both", expand=True)

        ttk.Label(rahmen, text="Ruhepuls messen",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(rahmen,
                  text=f"Die Testperson sitzt entspannt und bewegt sich {self.MESSDAUER_S} "
                       f"Sekunden lang nicht. Die Apple Watch muss getragen werden und "
                       f"Daten senden.\n\n"
                       f"Der Ruhepuls legt zusammen mit dem Alter fest, ab welcher "
                       f"Herzfrequenz gewarnt wird. Ohne ihn muss die Anwendung auf "
                       f"gröbere Ersatzwerte ausweichen.",
                  foreground="#555", wraplength=470,
                  justify="left").pack(anchor="w", pady=(6, 12))

        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(rahmen, textvariable=self.status_var, font=("Segoe UI", 11),
                  wraplength=470, justify="left").pack(anchor="w")

        self.live_var = tk.StringVar(value="")
        ttk.Label(rahmen, textvariable=self.live_var, font=("Consolas", 24, "bold"),
                  foreground="#c00").pack(anchor="w", pady=(8, 4))

        self.fortschritt = ttk.Progressbar(rahmen, maximum=self.MESSDAUER_S)
        self.fortschritt.pack(fill="x", pady=(4, 10))

        self.detail_var = tk.StringVar(value="")
        ttk.Label(rahmen, textvariable=self.detail_var, font=("Consolas", 9),
                  foreground="#555", wraplength=470,
                  justify="left").pack(anchor="w")

        knoepfe = ttk.Frame(rahmen)
        knoepfe.pack(fill="x", side="bottom", pady=(14, 0))
        self.start_button = ttk.Button(knoepfe, text="▶  Messung starten",
                                       command=self.starten)
        self.start_button.pack(side="left")
        ttk.Button(knoepfe, text="Schliessen", command=self.close).pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self.close)

    # ── Ablauf ────────────────────────────────────────────────────────────────

    def starten(self):
        if self._läuft:
            return
        # Ohne laufenden Server empfängt die Anwendung gar keine Uhrdaten – dann
        # liefe die Messung eine Minute lang ins Leere.
        start_server_once()

        with data_lock:
            letzte = latest_data["last_update"]
        if letzte == 0.0 or time.time() - letzte > DATA_TIMEOUT:
            messagebox.showwarning(
                "Keine Daten von der Uhr",
                "Es kommen gerade keine Sensordaten an.\n\n"
                "Bitte zuerst die App auf der Apple Watch starten und prüfen, "
                "dass sie an diesen Rechner sendet.",
                parent=self.win)
            return

        self._werte   = []
        self._läuft   = True
        self._start_t = time.time()
        self.start_button.config(state="disabled")
        recorder.log_event("ruhepuls_messung_start", self.teilnehmer_id)
        self._tick()

    def _tick(self):
        if not self._läuft:
            return
        verstrichen = time.time() - self._start_t

        with data_lock:
            hr     = latest_data["heart_rate"]
            letzte = latest_data["last_update"]

        daten_frisch = letzte != 0.0 and time.time() - letzte <= DATA_TIMEOUT
        if valid_hr(hr) and daten_frisch and verstrichen >= self.EINSCHWINGEN_S:
            self._werte.append(float(hr))

        self.live_var.set(f"{hr:.0f} BPM" if valid_hr(hr) else "-- BPM")
        self.fortschritt["value"] = min(verstrichen, self.MESSDAUER_S)

        if verstrichen < self.EINSCHWINGEN_S:
            self.status_var.set(
                f"Einschwingen... noch {self.EINSCHWINGEN_S - verstrichen:.0f} s "
                f"(diese Werte zaehlen nicht)")
        else:
            self.status_var.set(
                f"Messung laeuft... noch {max(0, self.MESSDAUER_S - verstrichen):.0f} s")

        if not daten_frisch:
            self.detail_var.set("Achtung: gerade keine Daten von der Uhr.")
        else:
            self.detail_var.set(f"{len(self._werte)} verwertbare Messpunkte")

        if verstrichen >= self.MESSDAUER_S:
            self._abschliessen()
            return
        self._job = self.win.after(self.TAKT_MS, self._tick)

    def _abschliessen(self):
        self._läuft = False
        self.start_button.config(state="normal")
        self.fortschritt["value"] = self.MESSDAUER_S

        if len(self._werte) < self.MIN_WERTE:
            self.status_var.set("Zu wenige verwertbare Messwerte.")
            self.detail_var.set(
                f"Nur {len(self._werte)} Messpunkte (mindestens {self.MIN_WERTE} noetig). "
                "Sitzt die Uhr richtig auf und sendet sie durchgehend?")
            recorder.log_event("ruhepuls_messung_abbruch",
                               f"nur {len(self._werte)} Messpunkte")
            return

        werte  = sorted(self._werte)
        mitte  = len(werte) // 2
        median = (werte[mitte] if len(werte) % 2
                  else (werte[mitte - 1] + werte[mitte]) / 2.0)
        self.ergebnis = int(round(median))

        zonen = hr_zonen(self.alter, self.ergebnis)
        self.status_var.set(f"Ruhepuls: {self.ergebnis} BPM")
        self.live_var.set(f"{self.ergebnis} BPM")
        self.detail_var.set(
            f"Median aus {len(werte)} Messpunkten (Spanne {werte[0]:.0f}"
            f"...{werte[-1]:.0f} BPM)\n"
            f"Daraus: Warnung ab {zonen['warn']} BPM, Gefahr ab {zonen['gefahr']} BPM\n"
            f"{zonen['verfahren']}")
        recorder.log_event("ruhepuls_gemessen",
                           f"{self.ergebnis} BPM aus {len(werte)} Messpunkten")
        self._fertig(self.ergebnis)

    def close(self):
        self._läuft = False
        if self._job is not None:
            try:
                self.win.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Fahrverhalten justieren
# ─────────────────────────────────────────────────────────────────────────────

class DrivingTuneWindow:
    """
    Schieberegler für ALLE Faktoren, die das Fahrverhalten bestimmen.

    Ursprünglich nur für die Lenkung gedacht ("Lenkung feinjustieren"), deckt
    das Fenster inzwischen Tempo, Lenkung, Anhalten und Rückwärtsfahrt ab – also
    das vollständige Fahrverhalten. Es ist damit kein Werkzeug für die
    Entwicklung mehr, sondern die Stelle, an der die Steuerung an die einzelne
    Testperson angepasst wird.

    Die Regler schreiben direkt in die Modulvariablen, die die Steuerungsschleife
    bei jedem Durchlauf neu liest – Änderungen wirken deshalb sofort, ohne
    Neustart und ohne die laufende Aufzeichnung zu unterbrechen.

    Wie die Größen zusammenwirken:

      1. Grund-/Höchsttempo       – Tempobereich der Geradeausfahrt
      2. Schwelle (je Seite)      – ab wann die Kurve überhaupt beginnt
      3. Max. Winkel (je Seite)   – wie viel Grad je Bezugstakt
      4. Kurven-Tempo             – wie schnell er in der Kurve fährt
      5. Bezugstakt der Lenkung   – auf welche Zeitspanne sich Punkt 3 bezieht
      6. Zykluszeit               – wie oft die Steuerung nachregelt

    Punkte 3 und 5 gehören zusammen: Die Drehgeschwindigkeit ist Winkel geteilt
    durch Bezugstakt. Punkt 6 wirkt NICHT mehr darauf – das war bis zur
    Umstellung auf die zeitbasierte Drehrate anders und der Grund, weshalb
    dieselbe Handhaltung je nach Rechenlast und Funkqualität unterschiedlich
    schnell gedreht hat.

    Nachvollziehbarkeit: Jede Änderung wird in events.csv festgehalten (siehe
    _log_changes). Ohne das wäre in der Auswertung nicht mehr zu erkennen,
    warum sich das Tempo mitten in einer Aufzeichnung ändert – metadata.json
    hält nur den Stand am Ende der Sitzung fest.
    """

    VERGLEICHS_NEIGUNGEN = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

    # Sphero-Obergrenze für roll(). Höhere Werte nimmt das Gerät nicht an.
    SPEED_MAX = 255

    # Wartezeit nach der letzten Reglerbewegung, bevor die Änderung ins
    # Ereignisprotokoll geht. Ohne diese Verzögerung entstünde beim Ziehen
    # eines Reglers pro Pixel ein Protokolleintrag.
    LOG_DEBOUNCE_MS = 700

    def __init__(self, parent: tk.Tk, speichern=None, person_text=None):
        """
        `speichern` wird vom Hauptfenster gestellt und legt die aktuellen Werte
        bei der gewählten Testperson ab; es gibt einen Rückmeldetext zurück.
        Die Probandenverwaltung bleibt damit Sache des Hauptfensters – dieses
        Fenster kennt nur Regler.
        """
        self.win = tk.Toplevel(parent)
        self.win.title("Fahrverhalten justieren")
        self.win.geometry("660x800")
        self.win.minsize(600, 500)

        self._speichern_cb = speichern
        self._person_text  = person_text or (lambda: "keine Testperson ausgewählt")

        self._max_links  = 0.0     # größter erreichter Ausschlag nach links
        self._max_rechts = 0.0     # ... und nach rechts
        self._running    = True
        self._log_job    = None
        self._anwenden_laeuft = False
        self._geloggt    = fahrverhalten_werte()

        scroll = probanden._ScrollFrame(self.win, height=650)
        scroll.pack(fill="both", expand=True, padx=14, pady=(12, 0))
        self._scroll = scroll
        main = scroll.inner

        ttk.Label(main, text="Fahrverhalten einstellen",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="Alle Regler wirken sofort, auch während der Sphero fährt. "
                       "Änderungen werden in der laufenden Aufzeichnung vermerkt.",
                  foreground="#555", wraplength=580,
                  justify="left").pack(anchor="w", pady=(4, 10))

        # ── Regelgrößen ───────────────────────────────────────────────────────
        self.var_tempo_min  = tk.DoubleVar(value=float(MIN_SPEED_DYN))
        self.var_tempo_max  = tk.DoubleVar(value=float(MAX_SPEED_DYN))
        self.var_schwelle_l = tk.DoubleVar(value=abs(GY_LEFT_THRESHOLD))
        self.var_winkel_l   = tk.DoubleVar(value=float(MAX_TURN_ANGLE_LEFT))
        self.var_schwelle_r = tk.DoubleVar(value=abs(GY_RIGHT_THRESHOLD))
        self.var_winkel_r   = tk.DoubleVar(value=float(MAX_TURN_ANGLE_RIGHT))
        self.var_tempo      = tk.DoubleVar(value=float(TURN_SPEED_FACTOR))
        self.var_zyklus     = tk.DoubleVar(value=float(ROLL_COMMAND_DURATION))
        self.var_vollgas    = tk.DoubleVar(value=abs(GX_FULL_SPEED))
        self.var_stoppzeit  = tk.DoubleVar(value=float(STOP_TIME))
        self.var_rueck_v    = tk.DoubleVar(value=float(BACKWARD_SPEED))
        self.var_rueck_t    = tk.DoubleVar(value=float(BACKWARD_DURATION))
        self.var_glaettung  = tk.DoubleVar(value=float(GLAETTUNG_TAU_S))
        self.var_hyst_gy    = tk.DoubleVar(value=float(GY_HYSTERESE))
        self.var_hyst_gx    = tk.DoubleVar(value=float(GX_HYSTERESE))
        self.var_modus      = tk.StringVar(value=STEUERMODUS)
        self.var_pos_l      = tk.DoubleVar(value=float(POS_MAX_OFFSET_LEFT))
        self.var_pos_r      = tk.DoubleVar(value=float(POS_MAX_OFFSET_RIGHT))
        self.var_expo       = tk.DoubleVar(value=float(LENK_EXPO))
        self.var_kopplung   = tk.DoubleVar(value=float(LENK_TEMPO_KOPPLUNG))
        self.var_bezugstakt = tk.DoubleVar(value=float(LENK_BEZUGSTAKT_S))
        self.var_maxdrehung = tk.DoubleVar(value=float(LENK_MAX_DREHUNG))

        # ── Fahrprofile ───────────────────────────────────────────────────────
        self._abschnitt(main, "Fahrprofil")
        ttk.Label(main,
                  text="Setzt Tempo und Wendigkeit auf einen erprobten Ausgangspunkt. "
                       "Die Kippschwellen bleiben dabei unverändert – sie gehören zur "
                       "Person, nicht zum Fahrstil.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))
        profil_reihe = ttk.Frame(main)
        profil_reihe.pack(fill="x", pady=(0, 2))
        for name in FAHRPROFILE:
            ttk.Button(profil_reihe, text=name,
                       command=lambda n=name: self._profil_anwenden(n)
                       ).pack(side="left", padx=(0, 6))
        self.profil_var = tk.StringVar(
            value="aktuell: von Hand eingestellt" if fahrprofil_name == PROFIL_MANUELL
            else f"aktuell: Profil {fahrprofil_name}")
        ttk.Label(main, textvariable=self.profil_var, foreground="#06c",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 0))

        # ── Art der Kursführung ───────────────────────────────────────────────
        self._abschnitt(main, "Art der Lenkung")
        ttk.Radiobutton(
            main, variable=self.var_modus, value=STEUERMODUS_RATE,
            text="Ratensteuerung – gehaltene Drehung dreht ihn immer weiter",
            command=lambda: self._uebernehmen()).pack(anchor="w")
        ttk.Label(main,
                  text="Bisheriges Verhalten. Beliebig grosse Kurven moeglich, "
                       "man muss die Hand aber im richtigen Moment zurueckdrehen.",
                  foreground="#777", wraplength=540,
                  justify="left").pack(anchor="w", padx=(22, 0), pady=(0, 6))
        ttk.Radiobutton(
            main, variable=self.var_modus, value=STEUERMODUS_POSITION,
            text="Positionssteuerung – die Handhaltung bestimmt den Kurs direkt",
            command=lambda: self._uebernehmen()).pack(anchor="w")
        ttk.Label(main,
                  text="Halbe Drehung = dauerhaft halber Kursversatz, Hand zurueck "
                       "= Kurs zurueck. Leichter zu dosieren, weniger Ueberschwingen. "
                       "Bei Geradeausfahrt rastet der erreichte Kurs ein, sodass "
                       "trotzdem jede Richtung erreichbar bleibt.",
                  foreground="#777", wraplength=540,
                  justify="left").pack(anchor="w", padx=(22, 0), pady=(0, 2))

        self._abschnitt(main, "1  Tempo geradeaus")
        self._regler(main, "Grundtempo", self.var_tempo_min, 0, 150, 1,
                     "hoeher = er rollt sofort zuegig los, kein sachtes Anfahren mehr",
                     "tiefer = die Fahrt beginnt ganz langsam und ist leichter zu "
                     "kontrollieren (gut fuer den ersten Versuch)",
                     was="Tempo in dem Moment, in dem er losfaehrt – also bei gerade "
                         "eben gesenkter Hand. Skala 0 bis 255.")
        self._regler(main, "Hoechsttempo", self.var_tempo_max, 20, self.SPEED_MAX, 5,
                     "hoeher = bei voll gesenkter Hand faehrt er deutlich schneller "
                     "(fuer sportliche Testpersonen)",
                     "tiefer = das Tempo bleibt insgesamt gedaempft, mehr Zeit zum "
                     "Reagieren",
                     was="Tempo bei voll gesenkter Hand. Zwischen Grundtempo und "
                         "Hoechsttempo steigt es gleichmaessig mit der Handneigung. "
                         "255 ist die Grenze des Sphero.")
        self._regler(main, "Neigung fuer Vollgas", self.var_vollgas, 0.30, 0.95, 0.05,
                     "hoeher = Vollgas erst bei weit gesenkter Hand; der ganze "
                     "Bewegungsweg steht zum Dosieren zur Verfuegung",
                     "tiefer = Vollgas schon bei wenig Neigung; hilft bei "
                     "eingeschraenkter Beweglichkeit, ist dafuer grober dosierbar",
                     was="Wie weit die Hand gesenkt sein muss, damit das Hoechsttempo "
                         "anliegt. 0.95 ist fast senkrecht nach unten.")

        self._abschnitt(main, "2  Linkskurve")
        self._regler(main, "Schwelle links", self.var_schwelle_l, 0.40, 0.95, 0.01,
                     "hoeher = die Hand muss weiter gedreht werden, bevor er ueberhaupt "
                     "lenkt; versehentliches Lenken wird seltener",
                     "tiefer = er lenkt schon bei leichter Drehung; leichter erreichbar, "
                     "aber auch schneller ungewollt ausgeloest",
                     was="Ab welcher Handdrehung die Linkskurve beginnt. Darunter faehrt "
                         "er geradeaus. 1.00 waere die Hand komplett auf der Seite.")
        self._regler(main, "Max. Winkel links", self.var_winkel_l, 10, 90, 1,
                     "hoeher = er dreht sich bei voller Handdrehung sehr schnell; gut "
                     "zum Wenden, aber schwerer fein zu dosieren",
                     "tiefer = er dreht insgesamt gemaechlicher; Kurven lassen sich "
                     "deutlich genauer treffen",
                     was="Hoechste Drehgeschwindigkeit, erreicht bei voller Handdrehung. "
                         "Angabe in Grad je Bezugstakt (siehe Regler weiter unten): "
                         "60 Grad je 0.22 s sind rund 270 Grad je Sekunde. Die Drehung "
                         "wird mit der tatsaechlich vergangenen Zeit gerechnet und "
                         "haengt deshalb NICHT mehr daran, wie schnell der Laptop und "
                         "die Funkverbindung gerade sind.")

        self._abschnitt(main, "3  Rechtskurve")
        self._regler(main, "Schwelle rechts", self.var_schwelle_r, 0.40, 0.95, 0.01,
                     "hoeher = die Hand muss weiter gedreht werden, bevor er ueberhaupt "
                     "lenkt; versehentliches Lenken wird seltener",
                     "tiefer = er lenkt schon bei leichter Drehung; leichter erreichbar, "
                     "aber auch schneller ungewollt ausgeloest",
                     was="Ab welcher Handdrehung die Rechtskurve beginnt. Darf sich von "
                         "der linken Schwelle unterscheiden – die Drehung faellt in eine "
                         "Richtung anatomisch schwerer.")
        self._regler(main, "Max. Winkel rechts", self.var_winkel_r, 10, 90, 1,
                     "hoeher = er dreht sich bei voller Handdrehung sehr schnell; gut "
                     "zum Wenden, aber schwerer fein zu dosieren",
                     "tiefer = er dreht insgesamt gemaechlicher; Kurven lassen sich "
                     "deutlich genauer treffen",
                     was="Hoechste Drehgeschwindigkeit nach rechts, erreicht bei voller "
                         "Handdrehung. Gleiche Einheit wie links: Grad je Bezugstakt.")

        self._abschnitt(main, "3b  Nur bei Positionssteuerung")
        ttk.Label(main,
                  text="Wie weit der Kurs bei voller Handdrehung vom Bezugskurs "
                       "abweicht. In der Ratensteuerung ohne Wirkung.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))
        self._regler(main, "Kursversatz links", self.var_pos_l, 10, 180, 5,
                     "hoeher = volle Handdrehung schwenkt ihn weiter herum; eine "
                     "Rechtwinkelkurve gelingt in einem Zug",
                     "tiefer = kleinere Schwenks je Zug, dafuer feiner dosierbar",
                     was="Nur bei Positionssteuerung: um wie viel Grad der Kurs bei "
                         "voller Handdrehung nach links abweicht.")
        self._regler(main, "Kursversatz rechts", self.var_pos_r, 10, 180, 5,
                     "hoeher = volle Handdrehung schwenkt ihn weiter herum; eine "
                     "Rechtwinkelkurve gelingt in einem Zug",
                     "tiefer = kleinere Schwenks je Zug, dafuer feiner dosierbar",
                     was="Nur bei Positionssteuerung: dasselbe fuer die rechte Seite.")

        self._abschnitt(main, "3c  Kurven fahren")
        ttk.Label(main,
                  text="Diese beiden Regler entscheiden, wie gut sich eine Kurve um ein "
                       "Objekt dosieren laesst. Auf 1.0 und 0.00 gestellt ergibt sich "
                       "das Verhalten von vorher.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))
        self._regler(main, "Feinfuehligkeit (Expo)", self.var_expo, 1.0, 4.0, 0.1,
                     "hoeher = der untere Teil des Handwegs dreht viel langsamer, "
                     "Kurven lassen sich genau dosieren; der Vollausschlag dreht "
                     "unveraendert schnell",
                     "tiefer = gleichmaessige Verteilung; die ganze Kurvendosierung "
                     "draengt sich in die ersten Hundertstel der Handdrehung (1.0 = aus)",
                     was="Verteilt die Drehgeschwindigkeit ueber den Handweg um. Bei 1.0 "
                         "ergibt halbe Handdrehung halbe Drehgeschwindigkeit, bei 2.0 nur "
                         "noch ein Viertel davon. KEINE Verzoegerung – der Sphero "
                         "reagiert genauso schnell wie vorher, nur mit anderer "
                         "Uebersetzung.")
        self._regler(main, "Kurve an Tempo koppeln", self.var_kopplung, 0.0, 1.0, 0.05,
                     "hoeher = gehaltene Hand ergibt einen Kreis mit gleichbleibendem "
                     "Radius wie bei einem Lenkrad; zum Kurvenfahren muss nicht mehr "
                     "abgebremst werden",
                     "tiefer = Drehgeschwindigkeit unabhaengig vom Tempo: langsame Fahrt "
                     "dreht eng, schnelle Fahrt macht weite Boegen (0 = aus)",
                     was="Wie stark die Drehgeschwindigkeit dem Fahrtempo folgt. Bei 1.00 "
                         "drehen sich Tempo und Kurve immer im gleichen Verhaeltnis, "
                         "der Kurvenradius bleibt also konstant.")
        self._regler(main, "Hoechste Drehung je Handdrehung", self.var_maxdrehung,
                     0, 360, 10,
                     "hoeher = er dreht sich bei gehaltener Hand weiter herum, bevor er "
                     "auf Kurs bleibt; 360 = mehr als eine volle Umdrehung",
                     "tiefer = eine einzelne Handdrehung schwenkt ihn nur ein Stueck "
                     "weit; zum Weiterdrehen muss die Hand kurz zurueck in die Mitte "
                     "(0 = aus, Voreinstellung)",
                     was="Begrenzt, wie weit EINE durchgehende Handdrehung den Kurs "
                         "aendern darf. VORSICHT beim Einschalten: Ist das Budget "
                         "aufgebraucht, faehrt der Ball geradeaus weiter, obwohl die "
                         "Hand noch gedreht ist – eine Dauerkurve um ein Objekt bricht "
                         "dann mitten drin ab und die Steuerung fuehlt sich an wie eine "
                         "Positionssteuerung. Deshalb steht der Regler auf 0. Sinnvoll "
                         "nur, wenn eine Testperson zum Ueberdrehen neigt. Ist das "
                         "Budget aufgebraucht, wird die Kurvenfarbe dunkler. Hand kurz "
                         "in die Mitte = wieder volles Budget. Nur bei Ratensteuerung.")
        self._regler(main, "Bezugstakt der Lenkung (s)", self.var_bezugstakt,
                     0.05, 0.40, 0.01,
                     "hoeher = dieselbe Handhaltung dreht langsamer (der Winkel oben "
                     "verteilt sich auf mehr Zeit)",
                     "tiefer = dieselbe Handhaltung dreht schneller",
                     was="Zeitspanne, auf die sich 'Max. Winkel' bezieht. Die Drehung wird "
                         "mit der wirklich vergangenen Zeit gerechnet, ist also unabhaengig "
                         "davon, wie schnell der Laptop und die Funkverbindung gerade sind. "
                         "0.22 s entspricht dem Verhalten der frueheren Fassung. "
                         "Zusammen mit 'Max. Winkel' ergibt sich die Drehrate: "
                         "60 Grad je 0.22 s sind rund 270 Grad je Sekunde.")

        self._abschnitt(main, "4  Gilt fuer beide Seiten")
        self._regler(main, "Kurven-Tempo", self.var_tempo, 0.20, 1.00, 0.05,
                     "hoeher = er bleibt in der Kurve schnell und faehrt einen weiten "
                     "Bogen",
                     "tiefer = er wird in der Kurve deutlich langsamer und dreht eng, "
                     "fast auf der Stelle",
                     was="Anteil des normalen Tempos, den er waehrend einer Kurve noch "
                         "faehrt. 0.70 heisst: in der Kurve 70 Prozent des Tempos, das "
                         "die Handneigung sonst ergaebe.")
        self._regler(main, "Zykluszeit (s)", self.var_zyklus, 0.04, 0.15, 0.01,
                     "hoeher = seltener nachgesteuert: er reagiert traeger, dafuer wird "
                     "die Funkverbindung geschont",
                     "tiefer = oefter nachgesteuert: er folgt der Hand direkter, "
                     "belastet aber die Funkverbindung (Abrissgefahr)",
                     was="Angestrebter Abstand zwischen zwei Durchlaeufen der Steuerung, "
                         "also die Reaktionszeit auf eine Handbewegung. Die "
                         "Drehgeschwindigkeit haengt NICHT mehr daran – dafuer sind "
                         "'Max. Winkel' und 'Bezugstakt der Lenkung' zustaendig. "
                         "Der tatsaechlich erreichte Takt steht unten in der Anzeige.")

        self._abschnitt(main, "5  Anhalten und Rueckwaerts")
        self._regler(main, "Nachlauf bis Stopp (s)", self.var_stoppzeit, 0.2, 2.0, 0.1,
                     "hoeher = kurzes Zurueckziehen der Hand stoppt ihn nicht sofort; "
                     "verzeiht Zittern und kurze Pausen",
                     "tiefer = er bleibt sofort stehen, sobald die Hand zurueckgeht",
                     was="Wie lange die Hand in Neutralstellung sein muss, bevor der "
                         "Stopp-Befehl geht.")
        self._regler(main, "Rueckwaerts-Tempo", self.var_rueck_v, 20, 200, 5,
                     "hoeher = die Rueckwaertsfahrt nach Doppeltipp ist zuegiger",
                     "tiefer = er setzt langsam und kontrolliert zurueck",
                     was="Festes Tempo waehrend der Rueckwaertsfahrt. Haengt nicht an "
                         "der Handneigung – die Rueckwaertsfahrt laeuft von selbst ab.")
        self._regler(main, "Rueckwaerts-Dauer (s)", self.var_rueck_t, 0.5, 5.0, 0.5,
                     "hoeher = ein Doppeltipp bringt ihn weiter zurueck",
                     "tiefer = kurzes Zuruecksetzen, dann wieder normale Steuerung",
                     was="Wie lange ein Doppeltipp auf die Uhr rueckwaerts fahren laesst. "
                         "Waehrend dieser Zeit ist die Handsteuerung ausgesetzt.")

        self._abschnitt(main, "6  Ruhige Hand (Zittern ausgleichen)")
        ttk.Label(main,
                  text="Die Uhr misst auch unwillkuerliches Zittern mit. Diese drei "
                       "Regler entscheiden, wie stark es die Steuerung erreicht. "
                       "Alle auf 0 gestellt ergibt das Verhalten von vorher.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))
        self._regler(main, "Glaettung (s)", self.var_glaettung, 0.0, 0.60, 0.05,
                     "hoeher = Zittern wird stark gedaempft, die Steuerung folgt der "
                     "Hand dafuer merklich verzoegert",
                     "tiefer = folgt der Hand unmittelbar, uebernimmt aber auch jedes "
                     "Zucken (0 = aus)",
                     was="Als einziger Regler eine ZEIT: Wie traege die Steuerung der "
                         "Hand folgt. Bei 0.15 s ist eine ruckartige Handbewegung nach "
                         "0,15 Sekunden zu knapp zwei Dritteln uebernommen.")
        self._regler(main, "Halteband Drehung", self.var_hyst_gy, 0.0, 0.15, 0.01,
                     "hoeher = eine begonnene Kurve bleibt bestehen, bis die Hand "
                     "deutlich zurueckgedreht wird; kein Flattern an der Schwelle",
                     "tiefer = die Kurve endet genau an der Schwelle, dafuer kann "
                     "Zittern dort staendig ein- und ausschalten (0 = aus)",
                     was="Wie weit die Hand ueber die Schwelle ZURUECK muss, um eine "
                         "begonnene Kurve wieder zu beenden. Angefangen wird weiterhin "
                         "genau an der Schwelle – wie bei einem Thermostat.")
        self._regler(main, "Halteband Fahren", self.var_hyst_gx, 0.0, 0.10, 0.01,
                     "hoeher = die Fahrt reisst nicht ab, wenn die Hand kurz etwas "
                     "hochgeht",
                     "tiefer = er haelt genau an der Schwelle an (0 = aus)",
                     was="Dasselbe fuer Fahren und Anhalten: wie weit die Hand ueber die "
                         "Fahrschwelle zurueck muss, damit die Fahrt endet.")

        self._abschnitt(main, "7  Richtungspfeil am Ball")
        ttk.Label(main,
                  text="Der Ball zeigt auf seiner LED-Matrix einen Pfeil in die Richtung, "
                       "in die er losfahren wuerde; der Rest der Matrix bleibt dunkel. "
                       "Der Pfeil traegt zugleich die Zustandsfarbe: gruen geradeaus, "
                       "gelb Rechtskurve, blau Linkskurve, rot steht, violett rueckwaerts. "
                       "Links und rechts sind blau und gelb statt rot und gruen, weil die "
                       "Rot-Gruen-Schwaeche haeufig ist und bei der Richtung kein Zweifel "
                       "entstehen darf.\n"
                       "Abgeschaltet leuchtet stattdessen wieder die ganze Matrix in der "
                       "Zustandsfarbe, so lassen sich beide Darstellungen im Fahrversuch "
                       "gegeneinander stellen.\n"
                       "Welche Kante der Matrix nach vorne zeigt, haengt an der "
                       "Einbaulage. Ball auf den Boden legen, kurz vorwaerts fahren "
                       "lassen und die Drehung so waehlen, dass der Pfeil in die "
                       "Fahrtrichtung zeigt. Der Wert bleibt danach fuer die ganze "
                       "Studie gleich.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 6))

        self.var_pfeil_an = tk.BooleanVar(value=bool(MATRIX_PFEIL_AN))
        ttk.Checkbutton(main, text="Richtungspfeil anzeigen",
                        variable=self.var_pfeil_an,
                        command=self._pfeil_uebernehmen).pack(anchor="w")

        pfeil_zeile = ttk.Frame(main)
        pfeil_zeile.pack(fill="x", pady=(4, 0))
        ttk.Label(pfeil_zeile, text="Drehung:").pack(side="left")
        self.var_pfeil_drehung = tk.IntVar(value=int(MATRIX_PFEIL_DREHUNG))
        for grad, richtung in ((0, "nach oben"), (90, "nach rechts"),
                               (180, "nach unten"), (270, "nach links")):
            ttk.Radiobutton(pfeil_zeile, text=f"{grad}\u00b0 ({richtung})",
                            value=grad, variable=self.var_pfeil_drehung,
                            command=self._pfeil_uebernehmen).pack(side="left",
                                                                  padx=(8, 0))
        ttk.Label(main,
                  text="Die Richtungsangabe beschreibt, wohin der Pfeil auf der Matrix "
                       "gedreht wird – nicht, wohin der Ball faehrt.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", pady=(2, 0))

        # ── Live-Anzeige ──────────────────────────────────────────────────────
        ttk.Separator(main).pack(fill="x", pady=(12, 8))
        self.live_var = tk.StringVar(value="Warte auf Sensordaten...")
        ttk.Label(main, textvariable=self.live_var, font=("Consolas", 10),
                  justify="left").pack(anchor="w")

        ttk.Separator(main).pack(fill="x", pady=(10, 8))
        ttk.Label(main, text="Vergleich beider Seiten",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tabelle_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.tabelle_var, font=("Consolas", 10),
                  justify="left").pack(anchor="w", pady=(4, 0))

        # ── Für die Testperson sichern ────────────────────────────────────────
        # Ohne diese Möglichkeit stünden bei jedem Programmstart wieder die
        # Vorgabewerte – eine Folgesitzung derselben Person wäre dann nicht mehr
        # mit der vorherigen vergleichbar, weil unbemerkt anders gefahren wurde.
        self._abschnitt(main, "Fuer diese Testperson sichern")
        self.person_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.person_var, foreground="#555",
                  wraplength=560, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(main, text="💾  Werte fuer diese Testperson speichern",
                   command=self._fuer_person_speichern).pack(anchor="w")
        self.speicher_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.speicher_var, foreground="#0a5",
                  wraplength=560, justify="left").pack(anchor="w", pady=(4, 0))

        # ── Übernahme ─────────────────────────────────────────────────────────
        ttk.Separator(main).pack(fill="x", pady=(10, 8))
        ttk.Label(main, text="Als neue Vorgabe fuer ALLE: diese Zeilen oben in "
                             "sphero_reha_main.py eintragen:",
                  font=("Segoe UI", 9)).pack(anchor="w")
        self.code_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.code_var, font=("Consolas", 10),
                  foreground="#0a5", justify="left").pack(anchor="w", pady=(2, 8))

        knoepfe = ttk.Frame(self.win, padding=(14, 8))
        knoepfe.pack(fill="x")
        ttk.Button(knoepfe, text="Ausschlaege zuruecksetzen",
                   command=self._reset_max).pack(side="left")
        ttk.Button(knoepfe, text="Werte in Konsole ausgeben",
                   command=self._print_werte).pack(side="left", padx=(8, 0))
        ttk.Button(knoepfe, text="Schliessen",
                   command=self.close).pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._update()

    # ── Aufbau ────────────────────────────────────────────────────────────────

    def _abschnitt(self, parent, titel):
        ttk.Separator(parent).pack(fill="x", pady=(10, 4))
        ttk.Label(parent, text=titel,
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 2))

    def _regler(self, parent, titel, variable, von, bis, schritt, hoch, tief,
                was=""):
        """
        Ein Schieberegler mit dreiteiliger Erklärung.

        `was`  – was der Wert überhaupt bedeutet, neutral und mit Einheit
        `hoch` – was ein höherer Wert im Fahren bewirkt
        `tief` – was ein niedrigerer Wert bewirkt

        Die neutrale Zeile ist wichtig: Aus "höher/tiefer" allein geht nicht
        hervor, WAS da eigentlich verstellt wird. Wer das Fenster zum ersten Mal
        öffnet, soll jeden Regler verstehen, ohne im Quelltext nachsehen zu
        müssen.
        """
        rahmen = ttk.Frame(parent)
        rahmen.pack(fill="x", pady=(6, 0))
        kopf = ttk.Frame(rahmen)
        kopf.pack(fill="x")
        ttk.Label(kopf, text=titel, font=("Segoe UI", 10, "bold")).pack(side="left")
        wert = ttk.Label(kopf, text="", font=("Consolas", 11), foreground="#06c")
        wert.pack(side="right")

        if was:
            ttk.Label(rahmen, text=was, foreground="#333",
                      wraplength=560, justify="left").pack(anchor="w", pady=(1, 2))

        tk.Scale(rahmen, from_=von, to=bis, resolution=schritt,
                 orient="horizontal", variable=variable, showvalue=False,
                 length=560).pack(fill="x")
        ttk.Label(rahmen, text="▲  " + hoch, foreground="#a60",
                  wraplength=560, justify="left").pack(anchor="w")
        ttk.Label(rahmen, text="▼  " + tief, foreground="#06a",
                  wraplength=560, justify="left").pack(anchor="w")

        # Anzeige UND Übernahme hängen an der Variablen, nicht am `command` des
        # Reglers: Dessen Rückruf löst nur das Ziehen mit der Maus aus. Ein
        # Fahrprofil, das die Variablen direkt setzt, bliebe damit wirkungslos.
        nachkomma = 2 if schritt < 1 else 0

        def geaendert(*_):
            wert.config(text=f"{variable.get():.{nachkomma}f}")
            self._uebernehmen()

        variable.trace_add("write", geaendert)
        wert.config(text=f"{variable.get():.{nachkomma}f}")

    # ── Wirkung ───────────────────────────────────────────────────────────────

    def _pfeil_uebernehmen(self):
        """
        Pfeileinstellung sofort wirksam machen.

        Bewusst getrennt von _uebernehmen: Der Pfeil ist reine Anzeige und
        veraendert das Fahrverhalten nicht. Wuerde er ueber dieselbe Methode
        laufen, wuerde jedes Umschalten der Pfeilrichtung das Fahrprofil auf
        "von Hand eingestellt" setzen und in der Auswertung wie eine Aenderung
        am Fahrverhalten aussehen.
        """
        global MATRIX_PFEIL_AN, MATRIX_PFEIL_DREHUNG
        MATRIX_PFEIL_AN      = bool(self.var_pfeil_an.get())
        MATRIX_PFEIL_DREHUNG = int(self.var_pfeil_drehung.get())
        pfeil_neu_zeichnen.set()
        recorder.log_event(
            "richtungspfeil",
            f"{'an' if MATRIX_PFEIL_AN else 'aus'}; "
            f"Drehung {MATRIX_PFEIL_DREHUNG} Grad")

    def _uebernehmen(self, profil: str = PROFIL_MANUELL):
        """
        Reglerwerte in die Modulvariablen schreiben (wirkt sofort).

        `profil` ist der Name des zuletzt gewählten Fahrprofils; beim Ziehen
        eines einzelnen Reglers ist die Einstellung keinem Profil mehr
        zuzuordnen und wird deshalb als PROFIL_MANUELL geführt.
        """
        # Die Klemmung unten schreibt in eine Reglervariable zurück und löst
        # damit erneut diese Methode aus. Ohne die Sperre entstünde daraus eine
        # Endlosschleife.
        if self._anwenden_laeuft:
            return
        self._anwenden_laeuft = True
        try:
            self._anwenden(profil)
        finally:
            self._anwenden_laeuft = False

    def _anwenden(self, profil: str):
        global MIN_SPEED_DYN, MAX_SPEED_DYN, GX_FULL_SPEED
        global GY_LEFT_THRESHOLD, MAX_TURN_ANGLE_LEFT
        global GY_RIGHT_THRESHOLD, MAX_TURN_ANGLE_RIGHT
        global TURN_SPEED_FACTOR, ROLL_COMMAND_DURATION
        global STOP_TIME, BACKWARD_SPEED, BACKWARD_DURATION
        global GLAETTUNG_TAU_S, GY_HYSTERESE, GX_HYSTERESE
        global STEUERMODUS, POS_MAX_OFFSET_LEFT, POS_MAX_OFFSET_RIGHT
        global LENK_EXPO, LENK_TEMPO_KOPPLUNG
        global LENK_BEZUGSTAKT_S, LENK_MAX_DREHUNG
        global fahrprofil_name

        MIN_SPEED_DYN = int(round(float(self.var_tempo_min.get())))
        MAX_SPEED_DYN = int(round(float(self.var_tempo_max.get())))
        # Ein Grundtempo über dem Höchsttempo würde die Rampe in calc_speed()
        # umkehren: Je stärker die Hand gesenkt wird, desto langsamer führe er.
        # Der Regler wird auf den geklemmten Wert zurückgesetzt, damit Anzeige
        # und tatsächliches Verhalten nicht auseinanderlaufen.
        if MIN_SPEED_DYN > MAX_SPEED_DYN:
            MIN_SPEED_DYN = MAX_SPEED_DYN
            self.var_tempo_min.set(MIN_SPEED_DYN)

        # Vollgas liegt bei gesenkter Hand, also im Negativen.
        GX_FULL_SPEED         = -round(abs(float(self.var_vollgas.get())), 2)
        GY_LEFT_THRESHOLD     = round(abs(float(self.var_schwelle_l.get())), 3)
        MAX_TURN_ANGLE_LEFT   = int(round(float(self.var_winkel_l.get())))
        # rechts wird als negativer Wert geführt (Kippen in die Gegenrichtung)
        GY_RIGHT_THRESHOLD    = -round(abs(float(self.var_schwelle_r.get())), 3)
        MAX_TURN_ANGLE_RIGHT  = int(round(float(self.var_winkel_r.get())))
        TURN_SPEED_FACTOR     = round(float(self.var_tempo.get()), 2)
        ROLL_COMMAND_DURATION = round(float(self.var_zyklus.get()), 3)
        STOP_TIME             = round(float(self.var_stoppzeit.get()), 2)
        BACKWARD_SPEED        = int(round(float(self.var_rueck_v.get())))
        BACKWARD_DURATION     = round(float(self.var_rueck_t.get()), 1)
        GLAETTUNG_TAU_S       = round(float(self.var_glaettung.get()), 2)
        GY_HYSTERESE          = round(float(self.var_hyst_gy.get()), 2)
        GX_HYSTERESE          = round(float(self.var_hyst_gx.get()), 2)
        POS_MAX_OFFSET_LEFT   = int(round(float(self.var_pos_l.get())))
        POS_MAX_OFFSET_RIGHT  = int(round(float(self.var_pos_r.get())))
        LENK_EXPO             = round(float(self.var_expo.get()), 2)
        LENK_TEMPO_KOPPLUNG   = round(float(self.var_kopplung.get()), 2)
        LENK_BEZUGSTAKT_S     = round(float(self.var_bezugstakt.get()), 3)
        LENK_MAX_DREHUNG      = int(round(float(self.var_maxdrehung.get())))
        if self.var_modus.get() in STEUERMODI:
            STEUERMODUS = self.var_modus.get()

        fahrprofil_name = profil
        self.profil_var.set(
            "aktuell: von Hand eingestellt" if profil == PROFIL_MANUELL
            else f"aktuell: Profil {profil}")
        self._log_anstossen()

    def _profil_anwenden(self, name: str):
        """Fahrprofil auf die Regler legen; die Kippschwellen bleiben unberührt."""
        werte = FAHRPROFILE.get(name)
        if not werte:
            return
        # Höchsttempo zuerst: Wird das Grundtempo zuerst angehoben, während das
        # alte, niedrigere Höchsttempo noch steht, klemmt _anwenden() es auf
        # diesen alten Wert – das Profil käme dann zu langsam heraus.
        self.var_tempo_max.set(werte["MAX_SPEED_DYN"])
        self.var_tempo_min.set(werte["MIN_SPEED_DYN"])
        self.var_tempo.set(werte["TURN_SPEED_FACTOR"])
        self.var_winkel_l.set(werte["MAX_TURN_ANGLE_LEFT"])
        self.var_winkel_r.set(werte["MAX_TURN_ANGLE_RIGHT"])
        self.var_zyklus.set(werte["ROLL_COMMAND_DURATION"])
        self.var_stoppzeit.set(werte["STOP_TIME"])
        self.var_rueck_v.set(werte["BACKWARD_SPEED"])
        self.var_glaettung.set(werte["GLAETTUNG_TAU_S"])
        self.var_hyst_gy.set(werte["GY_HYSTERESE"])
        self.var_hyst_gx.set(werte["GX_HYSTERESE"])
        self.var_pos_l.set(werte["POS_MAX_OFFSET_LEFT"])
        self.var_pos_r.set(werte["POS_MAX_OFFSET_RIGHT"])
        self.var_expo.set(werte["LENK_EXPO"])
        self.var_kopplung.set(werte["LENK_TEMPO_KOPPLUNG"])
        # .get() mit Vorgabe: Ein selbst ergänztes Profil ohne die neueren
        # Schlüssel soll die Regler nicht mit einem KeyError abbrechen lassen.
        self.var_bezugstakt.set(werte.get("LENK_BEZUGSTAKT_S", LENK_BEZUGSTAKT_S))
        self.var_maxdrehung.set(werte.get("LENK_MAX_DREHUNG", LENK_MAX_DREHUNG))
        self.var_modus.set(werte["STEUERMODUS"])
        self._uebernehmen(profil=name)
        set_status(f"Fahrprofil „{name}“ übernommen.")

    # ── Nachvollziehbarkeit ───────────────────────────────────────────────────

    def _log_anstossen(self):
        """
        Protokollierung anstoßen, aber erst nachdem der Regler zur Ruhe kam.

        Ein tk.Scale feuert während des Ziehens laufend; ohne diese Verzögerung
        entstünde für eine einzige Reglerbewegung eine lange Kette von
        Zwischenwerten in events.csv.
        """
        if self._log_job is not None:
            try:
                self.win.after_cancel(self._log_job)
            except Exception:
                pass
        self._log_job = self.win.after(self.LOG_DEBOUNCE_MS, self._log_changes)

    def _log_changes(self):
        self._log_job = None
        jetzt = fahrverhalten_werte()
        geaendert = [f"{name}: {self._geloggt[name]} -> {wert}"
                     for name, wert in jetzt.items()
                     if self._geloggt.get(name) != wert]
        if not geaendert:
            return
        self._geloggt = jetzt
        recorder.log_event("fahrverhalten_geaendert",
                           f"profil={fahrprofil_name}; " + "; ".join(geaendert))

    def regler_nachfuehren(self):
        """
        Regler auf den aktuellen Stand der Modulvariablen bringen.

        Nötig, wenn die Werte von außen gesetzt wurden – beim Laden der für eine
        Testperson gespeicherten Einstellung. Ohne das zeigten die Regler noch
        die alten Positionen, und die nächste Reglerbewegung schriebe den
        veralteten Stand aller übrigen Größen zurück.
        """
        if self._anwenden_laeuft:
            return
        self._anwenden_laeuft = True
        try:
            self.var_tempo_min.set(MIN_SPEED_DYN)
            self.var_tempo_max.set(MAX_SPEED_DYN)
            self.var_vollgas.set(abs(GX_FULL_SPEED))
            self.var_schwelle_l.set(abs(GY_LEFT_THRESHOLD))
            self.var_winkel_l.set(MAX_TURN_ANGLE_LEFT)
            self.var_schwelle_r.set(abs(GY_RIGHT_THRESHOLD))
            self.var_winkel_r.set(MAX_TURN_ANGLE_RIGHT)
            self.var_tempo.set(TURN_SPEED_FACTOR)
            self.var_zyklus.set(ROLL_COMMAND_DURATION)
            self.var_stoppzeit.set(STOP_TIME)
            self.var_rueck_v.set(BACKWARD_SPEED)
            self.var_rueck_t.set(BACKWARD_DURATION)
            self.var_glaettung.set(GLAETTUNG_TAU_S)
            self.var_hyst_gy.set(GY_HYSTERESE)
            self.var_hyst_gx.set(GX_HYSTERESE)
            self.var_pos_l.set(POS_MAX_OFFSET_LEFT)
            self.var_pos_r.set(POS_MAX_OFFSET_RIGHT)
            self.var_expo.set(LENK_EXPO)
            self.var_kopplung.set(LENK_TEMPO_KOPPLUNG)
            self.var_bezugstakt.set(LENK_BEZUGSTAKT_S)
            self.var_maxdrehung.set(LENK_MAX_DREHUNG)
            self.var_modus.set(STEUERMODUS)
            self.profil_var.set(
                "aktuell: von Hand eingestellt" if fahrprofil_name == PROFIL_MANUELL
                else f"aktuell: Profil {fahrprofil_name}")
        finally:
            self._anwenden_laeuft = False
        self._geloggt = fahrverhalten_werte()
        self.person_var.set(self._person_text())

    def _fuer_person_speichern(self):
        if self._speichern_cb is None:
            self.speicher_var.set("Speichern ist hier nicht verfügbar.")
            return
        try:
            self.speicher_var.set(self._speichern_cb())
        except Exception as e:
            self.speicher_var.set(f"Konnte nicht gespeichert werden: {e}")

    def _reset_max(self):
        self._max_links = self._max_rechts = 0.0

    def _werte_text(self) -> str:
        return (f"MIN_SPEED_DYN         = {MIN_SPEED_DYN}\n"
                f"MAX_SPEED_DYN         = {MAX_SPEED_DYN}\n"
                f"GX_FULL_SPEED         = {GX_FULL_SPEED:+.2f}\n"
                f"GY_LEFT_THRESHOLD     = {GY_LEFT_THRESHOLD:+.2f}\n"
                f"MAX_TURN_ANGLE_LEFT   = {MAX_TURN_ANGLE_LEFT}\n"
                f"GY_RIGHT_THRESHOLD    = {GY_RIGHT_THRESHOLD:+.2f}\n"
                f"MAX_TURN_ANGLE_RIGHT  = {MAX_TURN_ANGLE_RIGHT}\n"
                f"TURN_SPEED_FACTOR     = {TURN_SPEED_FACTOR}\n"
                f"ROLL_COMMAND_DURATION = {ROLL_COMMAND_DURATION}\n"
                f"STOP_TIME             = {STOP_TIME}\n"
                f"BACKWARD_SPEED        = {BACKWARD_SPEED}\n"
                f"BACKWARD_DURATION     = {BACKWARD_DURATION}\n"
                f"GLAETTUNG_TAU_S       = {GLAETTUNG_TAU_S}\n"
                f"GY_HYSTERESE          = {GY_HYSTERESE}\n"
                f"GX_HYSTERESE          = {GX_HYSTERESE}\n"
                f"STEUERMODUS           = {STEUERMODUS!r}\n"
                f"POS_MAX_OFFSET_LEFT   = {POS_MAX_OFFSET_LEFT}\n"
                f"POS_MAX_OFFSET_RIGHT  = {POS_MAX_OFFSET_RIGHT}\n"
                f"LENK_EXPO             = {LENK_EXPO}\n"
                f"LENK_TEMPO_KOPPLUNG   = {LENK_TEMPO_KOPPLUNG}\n"
                f"LENK_BEZUGSTAKT_S     = {LENK_BEZUGSTAKT_S}\n"
                f"LENK_MAX_DREHUNG      = {LENK_MAX_DREHUNG}")

    def _print_werte(self):
        print("[FAHRVERHALTEN] " + self._werte_text().replace("\n", "   "))

    # ── Laufende Anzeige ──────────────────────────────────────────────────────

    def _update(self):
        if not self._running:
            return
        try:
            self._zeichne()
        except Exception as e:
            self.live_var.set(f"Anzeigefehler: {e}")
        if self._running:
            self.win.after(150, self._update)

    def _zeichne(self):
        with data_lock:
            gy      = latest_data["gy"]
            gx      = latest_data["gx"]
            state   = latest_data["state"]
            loop_hz = latest_data.get("loop_hz", 0.0)

        if gy > 0:
            self._max_links = max(self._max_links, gy)
        else:
            self._max_rechts = min(self._max_rechts, gy)

        # Bezugsgröße der Drehrate ist seit der Umstellung auf zeitbasiertes
        # Lenken der Bezugstakt, NICHT mehr die Zykluszeit der Schleife. Beides
        # zu verwechseln war genau der Fehler, der die Kurven unberechenbar
        # gemacht hat, deshalb steht hier bewusst der Bezugstakt.
        zyklus   = LENK_BEZUGSTAKT_S if LENK_BEZUGSTAKT_S > 1e-6 else 0.22
        richtung = {"left": "LINKS", "right": "RECHTS"}.get(state, "geradeaus")
        tempo    = int(calc_speed(gx) * TURN_SPEED_FACTOR) if state in ("left", "right") \
            else calc_speed(gx)
        dreht    = state in ("left", "right")

        # Die beiden Steuerungsarten haben verschiedene Masseinheiten: Grad JE
        # SCHRITT (die sich aufsummieren) gegenüber Grad KURSVERSATZ (die es
        # nicht tun). Eine gemeinsame Anzeige wäre in einem der beiden Fälle
        # irreführend.
        if STEUERMODUS == STEUERMODUS_POSITION:
            versatz = calc_kursversatz(gy) if dreht else 0.0
            kopf = (f"gY jetzt : {gy:+.2f}  ->  {richtung:9s} "
                    f"Kursversatz {versatz:+6.1f} Grad (bleibt stehen)")
        else:
            # Tempo mitgeben, damit die Anzeige die Tempo-Kopplung mit abbildet
            # und nicht eine Drehrate zeigt, die so gar nicht gefahren wird.
            winkel = calc_turn(gy, tempo=tempo) if dreht else 0.0
            kopf = (f"gY jetzt : {gy:+.2f}  ->  {richtung:9s} {winkel:5.1f} "
                    f"Grad je Bezugstakt = {winkel / zyklus:6.0f} Grad/s")
            if LENK_TEMPO_KOPPLUNG > 0:
                kopf += f"  (Tempo-Kopplung {LENK_TEMPO_KOPPLUNG:.2f})"

        # Der gemessene Takt gehört sichtbar in die Oberfläche: Er ist die
        # Kenngröße dafür, ob die Steuerung gerade unter guten Bedingungen
        # läuft. Fällt er weit unter den eingestellten Wert, wird der Laptop zu
        # warm oder die Funkverbindung ist gestört – die Drehrate selbst bleibt
        # davon jetzt unberührt, das Ansprechverhalten aber nicht.
        ziel_hz = 1.0 / max(ROLL_COMMAND_DURATION + CONTROL_LOOP_SLEEP, 1e-6)
        if loop_hz > 0:
            takt_text = (f"Steuertakt gemessen {loop_hz:4.1f} Hz "
                         f"({1000/loop_hz:3.0f} ms), eingestellt {ziel_hz:4.1f} Hz")
            if loop_hz < 0.7 * ziel_hz:
                takt_text += "   <-- deutlich zu langsam"
        else:
            takt_text = (f"Steuertakt: Sphero-Steuerung laeuft nicht "
                         f"(eingestellt {ziel_hz:4.1f} Hz)")

        if STEUERMODUS == STEUERMODUS_POSITION or LENK_MAX_DREHUNG <= 0:
            budget_text = "Dreh-Budget: aus"
        else:
            budget_text = (f"Dreh-Budget: {LENK_MAX_DREHUNG} Grad je Handdrehung "
                           f"(Hand in die Mitte = wieder voll)")

        self.live_var.set(
            kopf + "\n"
            f"gX jetzt : {gx:+.2f}  ->  Tempo {tempo:3d}   "
            f"(Bereich {MIN_SPEED_DYN}...{MAX_SPEED_DYN}, "
            f"Vollgas ab gX {GX_FULL_SPEED:+.2f})\n"
            f"groesster Ausschlag   links {self._max_links:+.2f}   "
            f"rechts {self._max_rechts:+.2f}\n"
            f"{takt_text}\n"
            f"{budget_text}"
        )

        if STEUERMODUS == STEUERMODUS_POSITION:
            zeilen = ["  Handdrehung |  Kursversatz",
                      "              |  links   rechts",
                      "  ------------+----------------"]
            for n in self.VERGLEICHS_NEIGUNGEN:
                li = -calc_kursversatz(+n) if +n > abs(GY_LEFT_THRESHOLD)  else 0.0
                re = +calc_kursversatz(-n) if +n > abs(GY_RIGHT_THRESHOLD) else 0.0
                zeilen.append(f"      {n:.2f}    | {li:6.1f}   {re:6.1f}")
        else:
            zeilen = ["  Handdrehung |     links     |    rechts",
                      "              | /Takt   Grad/s| /Takt   Grad/s",
                      "  ------------+---------------+---------------"]
            for n in self.VERGLEICHS_NEIGUNGEN:
                li = calc_turn(+n, tempo=tempo) if +n > abs(GY_LEFT_THRESHOLD)  else 0.0
                re = calc_turn(-n, tempo=tempo) if +n > abs(GY_RIGHT_THRESHOLD) else 0.0
                zeilen.append(f"      {n:.2f}    |{li:6.1f} {li/zyklus:7.0f} "
                              f"|{re:6.1f} {re/zyklus:7.0f}")
        self.tabelle_var.set("\n".join(zeilen))
        self.code_var.set(self._werte_text())
        # Laufend nachziehen: Die Testperson kann im Hauptfenster gewechselt
        # werden, während dieses Fenster offen steht.
        self.person_var.set(self._person_text())

    def close(self):
        self._running = False
        self._scroll.unbind_mousewheel()
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-UI
# ─────────────────────────────────────────────────────────────────────────────

def show_controller_ui():
    root = tk.Tk()
    root.title("Sphero Reha-Controller")
    root.geometry("520x700")
    root.minsize(460, 600)
    root.resizable(True, True)

    status_var  = tk.StringVar(value=last_status)
    sensor_var  = tk.StringVar(value="Keine Sensordaten")
    state_var   = tk.StringVar(value="◯  neutral")
    heading_var = tk.StringVar(value="Heading: 0°")

    graph_window_ref = [None]
    camera_thread_ref = [None]
    tune_window_ref = [None]
    aim_window_ref  = [None]
    task_window_ref = [None]

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
    # In einem Canvas + Scrollbar untergebracht, damit auf kleinen Laptop-
    # Displays nichts verloren geht, wenn das Fenster nicht groß genug ist:
    # Inhalt wird dann per Mausrad oder Scrollbar erreichbar statt abgeschnitten.
    scroll_container = ttk.Frame(root)
    scroll_container.pack(fill="both", expand=True)

    main_canvas = tk.Canvas(scroll_container, highlightthickness=0)
    main_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical",
                                    command=main_canvas.yview)
    main_canvas.configure(yscrollcommand=main_scrollbar.set)
    main_canvas.pack(side="left", fill="both", expand=True)
    main_scrollbar.pack(side="right", fill="y")

    main = ttk.Frame(main_canvas, padding=18)
    main_window_id = main_canvas.create_window((0, 0), window=main, anchor="nw")

    def _main_scrollregion_update(_event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    main.bind("<Configure>", _main_scrollregion_update)

    def _canvas_width_update(event):
        # main soll immer genau so breit sein wie der sichtbare Canvas-Bereich,
        # sonst würden waagerechte fill="x"-Layouts bei Breitenänderung brechen.
        main_canvas.itemconfigure(main_window_id, width=event.width)
    main_canvas.bind("<Configure>", _canvas_width_update)

    def _mausrad_scrollen(event):
        main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    # Nur binden, solange die Maus über dem Canvas ist – sonst würde das
    # Mausrad auch anderen Fenstern/Widgets das Scrollen wegnehmen.
    main_canvas.bind("<Enter>", lambda _e: main_canvas.bind_all("<MouseWheel>", _mausrad_scrollen))
    main_canvas.bind("<Leave>", lambda _e: main_canvas.unbind_all("<MouseWheel>"))

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

    # ── Herzfrequenzzonen der gewählten Person ────────────────────────────────
    # Die Zonen stehen sichtbar im Hauptfenster, weil sie die Sicherheitsgrenze
    # der Sitzung sind: Die betreuende Person muss ohne Umweg über ein
    # Untermenü sehen, ab wann die Anzeige orange bzw. rot wird und worauf
    # dieser Wert beruht.
    hr_zone_var = tk.StringVar(value="Herzfrequenzzonen: Standardwerte (100 / 120 BPM)")
    ttk.Label(person_frame, textvariable=hr_zone_var,
              font=("Consolas", 9), foreground="#a05000",
              wraplength=440, justify="left").pack(anchor="w", pady=(8, 0))

    ruhepuls_button = ttk.Button(person_frame, text="❤  Ruhepuls messen")
    ruhepuls_button.pack(anchor="w", pady=(6, 0))

    video_check = ttk.Checkbutton(
        person_frame, variable=video_consent_var,
        text="Videoaufzeichnung dieser Sitzung (freiwillig)")
    video_check.pack(anchor="w", pady=(8, 0))

    def refresh_person_box():
        ids = registry.ids()
        person_box["values"] = [registry.label(pid) for pid in ids]
        if selected_pid[0] in ids:
            participant_var.set(registry.label(selected_pid[0]))

    def hr_zone_anzeige_aktualisieren(record):
        """Zonen neu berechnen und anzeigen; ohne Person auf Vorgabe zurück."""
        setze_hr_zonen(record)
        if not record:
            hr_zone_var.set(
                f"Herzfrequenzzonen: Standardwerte "
                f"({HR_WARN_DEFAULT} / {HR_DANGER_DEFAULT} BPM) – "
                f"keine Testperson gewählt")
            ruhepuls_button.config(state="disabled")
            return
        ruhepuls_button.config(state="normal")
        ruhe = record.get("resting_hr_bpm")
        if valid_hr(ruhe):
            hr_zone_var.set(
                f"Herzfrequenz: Warnung ab {HR_WARN} BPM, Gefahr ab {HR_DANGER} BPM\n"
                f"(Alter {record['age_years']} J., Ruhepuls {int(ruhe)} BPM, "
                f"HFmax {hr_max_tanaka(record['age_years']):.0f}, Karvonen)")
        else:
            hr_zone_var.set(
                f"Herzfrequenz: Warnung ab {HR_WARN} BPM, Gefahr ab {HR_DANGER} BPM\n"
                f"⚠ Ruhepuls fehlt – Ersatzrechnung über HFmax. "
                f"Für genauere Zonen bitte Ruhepuls messen.")

    def fahrwerte_laden(record):
        """
        Die für diese Person gespeicherten Fahrwerte übernehmen.

        Ist noch nichts gespeichert, bleibt die aktuelle Einstellung stehen –
        sie ist als Ausgangspunkt brauchbarer als ein Rücksprung auf die
        Vorgabe, denn beim Anlegen einer neuen Person wurde meist gerade erst
        passend eingestellt.
        """
        if not record:
            return None
        gespeichert = record.get("fahrverhalten")
        if not gespeichert:
            return None
        werte  = gespeichert.get("werte", {})
        profil = gespeichert.get("profil")
        geaendert = fahrverhalten_anwenden(werte, profil=profil)
        tw = tune_window_ref[0]
        if tw is not None and tw.win.winfo_exists():
            tw.regler_nachfuehren()
        recorder.log_event(
            "fahrverhalten_geladen",
            f"Testperson {record.get('participant_id')}; profil={fahrprofil_name}; "
            f"{len(geaendert)} Werte geaendert")
        return gespeichert.get("gespeichert_iso")

    def apply_selection(pid):
        selected_pid[0] = pid
        record = registry.get(pid) if pid else None
        if not record:
            participant_info_var.set("Keine Testperson ausgewählt.")
            video_consent_var.set(False)
            video_check.config(state="disabled")
            hr_zone_anzeige_aktualisieren(None)
            return
        hr_zone_anzeige_aktualisieren(record)
        geladen_am = fahrwerte_laden(record)
        allowed = record.get("consent", {}).get("video", False)
        video_check.config(state="normal" if allowed else "disabled")
        # Standard = das, was die Person bei der Aufnahme zugestimmt hat.
        # Ohne Zustimmung bleibt die Box zwangsweise aus.
        video_consent_var.set(bool(allowed))
        parq_hint = ("PAR-Q: mindestens ein „Ja“ – bitte abklären"
                     if record.get("parq_any_yes") else "PAR-Q: unauffällig")
        video_hint = ("Video: eingewilligt" if allowed
                      else "Video: NICHT eingewilligt – keine Videoaufnahme möglich")
        fahr_hint = (f"Fahrwerte geladen (gespeichert {geladen_am[:16].replace('T', ' ')})"
                     if geladen_am else "Fahrwerte: noch keine gespeichert")
        participant_info_var.set(
            f"{record['age_years']} J. (Gruppe {record['age_group']}) | "
            f"{record['sex']} | {record['handedness']} | "
            f"Technikaffinität {record['tech_affinity']}/5\n"
            f"{parq_hint}  |  {video_hint}\n"
            f"{fahr_hint}"
        )

    def ruhepuls_messen():
        pid = selected_pid[0]
        record = registry.get(pid) if pid else None
        if not record:
            messagebox.showinfo(
                "Keine Testperson gewählt",
                "Bitte zuerst eine Testperson auswählen oder anlegen.")
            return

        def uebernehmen(bpm):
            registry.update(pid,
                            resting_hr_bpm=int(bpm),
                            resting_hr_measured_iso=datetime.now().isoformat(
                                timespec="seconds"))
            # Zonen sofort neu rechnen, damit die Anzeige und die
            # Graphen-Schwellen ab jetzt mit dem gemessenen Wert arbeiten.
            hr_zone_anzeige_aktualisieren(registry.get(pid))
            set_status(f"Ruhepuls {int(bpm)} BPM gespeichert – "
                       f"Zonen jetzt {HR_WARN}/{HR_DANGER} BPM.")

        RuhepulsFenster(root, pid, record["age_years"], fertig=uebernehmen)

    def ruhepuls_nachfragen(pid):
        """
        Nach fehlendem Ruhepuls fragen, statt ihn stillschweigend zu ersetzen.

        Ohne Nachfrage würde die Ersatzrechnung über HFmax unbemerkt greifen –
        die Sitzung liefe mit gröberen Zonen, ohne dass es jemandem auffällt.
        """
        record = registry.get(pid)
        if not record or valid_hr(record.get("resting_hr_bpm")):
            return
        if messagebox.askyesno(
                "Ruhepuls noch nicht gemessen",
                f"Für {pid} liegt noch kein Ruhepuls vor.\n\n"
                "Ohne ihn werden die Herzfrequenzzonen nur grob über die "
                "maximale Herzfrequenz geschätzt. Die Messung dauert eine "
                "Minute im Sitzen.\n\n"
                "Jetzt messen?"):
            ruhepuls_messen()

    def on_person_selected(_event=None):
        label = participant_var.get()
        for pid in registry.ids():
            if registry.label(pid) == label:
                apply_selection(pid)
                ruhepuls_nachfragen(pid)
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
            # Direkt nach dem Anlegen ist der richtige Zeitpunkt: Die Person
            # sitzt noch, bevor die Übung beginnt – später wäre der Puls durch
            # die Bewegung erhöht und als Ruhepuls unbrauchbar.
            ruhepuls_nachfragen(pid)

    person_box.bind("<<ComboboxSelected>>", on_person_selected)
    new_person_button.config(command=new_person_clicked)
    ruhepuls_button.config(command=ruhepuls_messen)
    refresh_person_box()
    apply_selection(None)

    # ── Buttons ────────────────────────────────────────────────────────────────
    buttons = ttk.Frame(main)
    buttons.pack(fill="x")

    start_button  = ttk.Button(buttons, text="▶  Sphero starten")
    stop_button   = ttk.Button(buttons, text="■  Sphero stoppen")
    graph_button  = ttk.Button(buttons, text="📊  Live Graphen")
    camera_button = ttk.Button(buttons, text="📷  Kamera starten")
    # Tempo, Lenkung, Anhalten und Rückwärtsfahrt – die Anpassung der Steuerung
    # an die einzelne Testperson. Bewusst ein eigenes Fenster: Während der Fahrt
    # wird hier nachjustiert, das Hauptfenster soll dabei sichtbar bleiben.
    tune_button   = ttk.Button(buttons, text="🎛  Fahrverhalten justieren")
    # Gehört direkt zur Inbetriebnahme: Ohne ausgerichtete Nullrichtung fährt
    # der Sphero bei "vorwärts" in eine beliebige Richtung. Deshalb gleich
    # neben "Sphero starten" und nicht in einem Untermenü.
    aim_button    = ttk.Button(buttons, text="🧭  Sphero ausrichten")
    # Die Standardaufgabe ist die eigentliche Messung der Studie und steht
    # deshalb gleichrangig neben der Aufzeichnung, nicht in einem Untermenü.
    task_button   = ttk.Button(buttons, text="🏁  Standardaufgabe")
    record_button = ttk.Button(buttons, text="⏺  Aufzeichnung starten")

    start_button.grid( row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
    stop_button.grid(  row=0, column=1, sticky="ew",              pady=3)
    aim_button.grid(   row=1, column=0, columnspan=2, sticky="ew", pady=3)
    graph_button.grid( row=2, column=0, sticky="ew", padx=(0, 6), pady=3)
    camera_button.grid(row=2, column=1, sticky="ew",              pady=3)
    tune_button.grid(  row=3, column=0, columnspan=2, sticky="ew", pady=3)
    record_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=3)
    task_button.grid(  row=5, column=0, columnspan=2, sticky="ew", pady=(3, 0))
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    # ── Tragearm der Uhr ──────────────────────────────────────────────────────
    # Bewusst hier bei der Steuerung und NICHT bei den Probandendaten: Es ist
    # eine Einstellung des Geräts, keine Eigenschaft der Person, und sie muss
    # auch ohne ausgewählte Testperson (freies Testen) verstellbar sein.
    # Jederzeit umschaltbar, damit der Arm im laufenden Betrieb gewechselt
    # werden kann, ohne die Steuerung neu zu starten.
    arm_frame = ttk.LabelFrame(main, text=" Apple Watch getragen am ", padding=8)
    arm_frame.pack(fill="x", pady=(10, 0))

    arm_var = tk.StringVar(value=watch_arm)

    def arm_geaendert():
        global watch_arm
        neu = arm_var.get()
        if neu == watch_arm:
            return
        watch_arm = neu
        recorder.log_event("watch_arm_changed", neu)
        set_status(f"Uhr wird am {neu}en Arm getragen – Achsen entsprechend umgerechnet.")

    arm_row = ttk.Frame(arm_frame)
    arm_row.pack(fill="x")
    ttk.Radiobutton(arm_row, text="linker Arm", value=WATCH_ARM_LEFT,
                    variable=arm_var, command=arm_geaendert).pack(side="left")
    ttk.Radiobutton(arm_row, text="rechter Arm", value=WATCH_ARM_RIGHT,
                    variable=arm_var, command=arm_geaendert).pack(side="left", padx=(16, 0))
    ttk.Label(arm_frame,
              text="Am rechten Arm liegt die Uhr um 180° gedreht am Handgelenk. "
                   "Ohne diese Angabe wären Fahren/Stopp und Links/Rechts vertauscht.",
              foreground="#777", wraplength=430,
              justify="left").pack(anchor="w", pady=(4, 0))

    # ── Vibrationsgürtel ──────────────────────────────────────────────────────
    # Optional. Ohne Häkchen läuft kein Thread und es wird keine Verbindung
    # aufgebaut – der Gürtel kostet dann nichts und kann nichts stören.
    guertel_frame = ttk.LabelFrame(main, text=" Vibrationsgürtel ", padding=8)
    guertel_frame.pack(fill="x", pady=(10, 0))

    guertel_var    = tk.BooleanVar(value=False)
    guertel_status = tk.StringVar(value="aus")

    def guertel_umschalten():
        if guertel_var.get():
            if vibrationsguertel.start():
                set_status("Gürtel wird verbunden...")
        else:
            # Nicht auf das Trennen warten: Die Oberfläche darf dabei nicht
            # einfrieren, falls der Gürtel gerade nicht antwortet.
            vibrationsguertel.stop(warten=False)

    ttk.Checkbutton(guertel_frame, variable=guertel_var,
                    text="Gürtel verwenden (vibriert mit der Lenkrichtung)",
                    command=guertel_umschalten).pack(anchor="w")
    ttk.Label(guertel_frame, textvariable=guertel_status,
              font=("Consolas", 9), foreground="#555",
              wraplength=430, justify="left").pack(anchor="w", pady=(4, 0))
    ttk.Label(guertel_frame,
              text="rechts = Vibration rechts   ·   links = Vibration links   ·   "
                   "rückwärts = Vibration hinten",
              foreground="#777", wraplength=430,
              justify="left").pack(anchor="w")

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
            loop_hz     = latest_data.get("loop_hz", 0.0)
            status_text = last_status

        # Steuertakt gehört ins Hauptfenster, nicht nur ins Einstellfenster:
        # Er ist die Kenngröße dafür, ob die Steuerung gerade unter guten
        # Bedingungen läuft, und muss während einer Testfahrt ohne Scrollen und
        # ohne zusätzliches Fenster ablesbar sein. Fällt er deutlich unter den
        # eingestellten Wert, wird der Laptop zu warm oder die Funkverbindung
        # ist gestört – die Drehrate bleibt davon zwar unberührt (siehe
        # LENK_BEZUGSTAKT_S), das Ansprechen auf die Hand aber nicht.
        ziel_hz = 1.0 / max(ROLL_COMMAND_DURATION + CONTROL_LOOP_SLEEP, 1e-6)
        if loop_hz <= 0:
            takt_text = "Takt –"
        elif loop_hz < 0.7 * ziel_hz:
            takt_text = f"Takt {loop_hz:.1f} Hz von {ziel_hz:.0f}  ZU LANGSAM"
        else:
            takt_text = f"Takt {loop_hz:.1f} Hz"

        # Angezeigt werden die bereits auf den Tragearm normierten Werte –
        # dieselben, mit denen die Steuerung rechnet.
        sensor_var.set(f"gX={gx:+.2f}  gY={gy:+.2f}  gZ={gz:+.2f}   "
                       f"(Uhr {watch_arm})   |   {takt_text}")
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

        # Gürtelstatus anzeigen. Ist der Gürtel-Thread beendet (Verbindung
        # fehlgeschlagen oder verloren), das Häkchen wieder abwählen – sonst
        # sähe es aus, als wäre der Gürtel noch aktiv.
        guertel_status.set(vibrationsguertel.status())
        if guertel_var.get() and not vibrationsguertel.ist_aktiv():
            guertel_var.set(False)

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

    def fahrwerte_speichern() -> str:
        """
        Aktuelle Fahrwerte bei der gewählten Testperson ablegen.

        Rückgabe ist der Text, den das Einstellfenster anzeigt.
        """
        pid = selected_pid[0]
        if not pid:
            return "Keine Testperson gewählt – bitte im Hauptfenster auswählen."
        gespeichert_iso = datetime.now().isoformat(timespec="seconds")
        registry.update(pid, fahrverhalten={
            "profil":           fahrprofil_name,
            "werte":            fahrverhalten_werte(),
            "gespeichert_iso":  gespeichert_iso,
        })
        recorder.log_event("fahrverhalten_gespeichert",
                           f"Testperson {pid}; profil={fahrprofil_name}")
        set_status(f"Fahrwerte für {pid} gespeichert.")
        apply_selection(pid)      # Hinweiszeile im Hauptfenster nachziehen
        return (f"Gespeichert für {pid} um "
                f"{gespeichert_iso[11:16]} Uhr. Wird beim nächsten Auswählen "
                f"dieser Testperson automatisch geladen.")

    def person_beschreibung() -> str:
        pid = selected_pid[0]
        if not pid:
            return ("Keine Testperson gewählt. Die Werte gelten nur für diesen "
                    "Programmlauf und gehen beim Beenden verloren.")
        record = registry.get(pid) or {}
        gespeichert = (record.get("fahrverhalten") or {}).get("gespeichert_iso")
        if gespeichert:
            return (f"Gewählt: {pid}. Zuletzt gespeichert am "
                    f"{gespeichert[:16].replace('T', ' ')} Uhr.")
        return f"Gewählt: {pid}. Für diese Testperson ist noch nichts gespeichert."

    def toggle_tuning():
        tw = tune_window_ref[0]
        if tw is not None and tw.win.winfo_exists():
            tw.close()
        else:
            start_server_once()
            tune_window_ref[0] = DrivingTuneWindow(
                root, speichern=fahrwerte_speichern, person_text=person_beschreibung)

    def aufgabe_clicked():
        aw = task_window_ref[0]
        if aw is not None and aw.win.winfo_exists():
            aw.win.lift()
            return
        if not recorder.active:
            # Kein harter Abbruch: Für eigenes Ausprobieren ist der Timer auch
            # ohne Aufzeichnung nützlich. Aber es muss klar sein, dass dann
            # nichts in den Daten landet.
            if not messagebox.askyesno(
                    "Keine Aufzeichnung aktiv",
                    "Es läuft gerade keine Aufzeichnung. Die Zeit wird dann zwar "
                    "angezeigt, aber NICHT gespeichert.\n\n"
                    "Für eine Studiensitzung bitte zuerst die Aufzeichnung starten.\n\n"
                    "Trotzdem fortfahren (zum Ausprobieren)?"):
                return
        task_window_ref[0] = AufgabeFenster(root, person_beschreibung)

    def ausrichten_clicked():
        aw = aim_window_ref[0]
        if aw is not None and aw.win.winfo_exists():
            aw.win.lift()
            return
        if sphero_api is None:
            messagebox.showinfo(
                "Sphero nicht verbunden",
                "Bitte zuerst „Sphero starten“ drücken. Das Ausrichten sendet "
                "Befehle an den Ball und braucht dafür eine Verbindung.")
            return
        aim_window_ref[0] = AusrichtFenster(root)

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
                pid = recorder.participant_id
                messagebox.showinfo(
                    "Aufzeichnung gespeichert",
                    f"Sitzung wurde gespeichert unter:\n{session_dir}\n\n"
                    f"Alle Durchgänge dieser Testperson:\n"
                    f"{sammel_csv_pfad(pid) if pid else '-'}\n\n"
                    f"Alle Durchgänge aller Testpersonen:\n"
                    f"{os.path.join(SESSIONS_DIR, STUDIE_SAMMEL_DATEI)}"
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
        vibrationsguertel.stop(warten=False)
        gw = graph_window_ref[0]
        if gw is not None and gw.win.winfo_exists():
            gw.close()
        tw = tune_window_ref[0]
        if tw is not None and tw.win.winfo_exists():
            tw.close()
        if recorder.active:
            recorder.stop()
        root.destroy()


    # ── Callbacks zuweisen ────────────────────────────────────────────────────
    start_button.config( command=start_clicked)
    stop_button.config(  command=stop_clicked,  state="disabled")
    graph_button.config( command=toggle_graphs)
    camera_button.config(command=toggle_camera)
    tune_button.config(  command=toggle_tuning)
    aim_button.config(   command=ausrichten_clicked)
    task_button.config(  command=aufgabe_clicked)
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
