"""
Probandenverwaltung – pseudonymisierte Stammdaten für die Studienauswertung

Datenschutz-Konzept (DSGVO Art. 4(5) Pseudonymisierung, Art. 5(1)(c) Datenminimierung):
  * In dieser Anwendung wird NIEMALS ein Klarname, Geburtsdatum oder eine
    Kontaktinformation gespeichert. Gespeichert wird ausschließlich eine
    vergebene Teilnehmer-ID (P01, P02, ...) plus die für die Auswertung
    zwingend nötigen Merkmale.
  * Die Zuordnung "Teilnehmer-ID <-> reale Person" existiert nur auf der
    unterschriebenen Einverständniserklärung in Papierform und wird getrennt
    von den Messdaten aufbewahrt.
  * Herzfrequenz-, Skelett- und Videodaten sind besondere Kategorien
    personenbezogener Daten (DSGVO Art. 9). Die Videoaufzeichnung ist deshalb
    grundsätzlich freiwillig und standardmäßig deaktiviert.

Erhobene Merkmale und ihre Begründung:
  Alter            – Kernvariable für den Altersgruppenvergleich der Studie
  Geschlecht       – Standard-Kovariate in Bewegungsstudien ("keine Angabe" möglich)
  Händigkeit       – ohne sie sind die getrennt erfassten Arm-Winkel links/rechts
                     nicht interpretierbar
  Technikaffinität – erklärt Varianz in den Usability-Werten zwischen Altersgruppen
  Vorerfahrung     – dito
  PAR-Q            – Physical Activity Readiness Questionnaire (revidiert 2002),
                     international etabliertes 7-Item-Screening vor
                     Bewegungsinterventionen. Erhoben werden nur Ja/Nein-Antworten,
                     KEINE Diagnosen im Klartext.
  Ruhepuls         – Bezugsgröße für die individuellen Herzfrequenzzonen
                     (Karvonen-Verfahren, siehe hr_zonen() in sphero_reha_main).
                     Ohne ihn müssten die Warnschwellen auf absolute BPM-Werte
                     zurückfallen, die im Altersgruppenvergleich nicht
                     vergleichbar sind.

Zusätzlich am Datensatz gespeichert, aber KEINE personenbezogenen Merkmale:
  fahrverhalten    – die für diese Person eingestellten Steuerungsparameter
                     (Tempo, Kippschwellen, Wendigkeit). Gerätekonfiguration,
                     kein Personenmerkmal – ohne sie wäre eine Folgesitzung
                     nicht mit der vorherigen vergleichbar.
"""

import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

PARTICIPANTS_FILE = "participants.json"

# ── PAR-Q (Physical Activity Readiness Questionnaire, revidiert 2002) ─────────
PARQ_ITEMS = [
    ("parq_1", "Hat Ihnen ein Arzt jemals gesagt, dass Sie ein Herzleiden haben und nur "
               "die von einem Arzt empfohlene körperliche Aktivität ausüben sollten?"),
    ("parq_2", "Haben Sie Schmerzen in der Brust, wenn Sie körperlich aktiv sind?"),
    ("parq_3", "Hatten Sie im letzten Monat Schmerzen in der Brust, ohne körperlich "
               "aktiv gewesen zu sein?"),
    ("parq_4", "Verlieren Sie durch Schwindel das Gleichgewicht oder haben Sie jemals "
               "das Bewusstsein verloren?"),
    ("parq_5", "Haben Sie Knochen- oder Gelenkprobleme, die sich durch körperliche "
               "Aktivität verschlimmern könnten?"),
    ("parq_6", "Verschreibt Ihnen ein Arzt derzeit Medikamente gegen Bluthochdruck "
               "oder ein Herzleiden?"),
    ("parq_7", "Ist Ihnen ein anderer Grund bekannt, warum Sie keine körperliche "
               "Aktivität ausüben sollten?"),
]

SEX_OPTIONS = [("w", "weiblich"), ("m", "männlich"),
               ("d", "divers"), ("k.A.", "keine Angabe")]

HANDEDNESS_OPTIONS = [("rechts", "Rechtshänder:in"),
                      ("links", "Linkshänder:in"),
                      ("beid", "beidhändig")]

TECH_AFFINITY_LABELS = {
    1: "1 – gar nicht vertraut",
    2: "2 – wenig vertraut",
    3: "3 – teils/teils",
    4: "4 – vertraut",
    5: "5 – sehr vertraut",
}

MIN_AGE = 18
MAX_AGE = 99


def age_group(age_years: int) -> str:
    """Bildet das Alter auf die Vergleichsgruppen der Studie ab (Dekaden)."""
    if age_years < 30:
        return "<30"
    if age_years >= 70:
        return "70+"
    decade = (age_years // 10) * 10
    return f"{decade}-{decade + 9}"


def analysis_view(record: dict) -> dict:
    """
    Reduziert einen Probanden-Datensatz auf die Felder, die in die
    Sitzungs-Metadaten geschrieben werden. Enthält bewusst weder Notizen
    noch Einwilligungs-Details, sondern nur Auswertungsmerkmale.
    """
    return {
        "participant_id":   record.get("participant_id"),
        "age_years":        record.get("age_years"),
        "age_group":        record.get("age_group"),
        "sex":              record.get("sex"),
        "handedness":       record.get("handedness"),
        "tech_affinity":    record.get("tech_affinity"),
        "prior_experience": record.get("prior_experience"),
        "parq_any_yes":     record.get("parq_any_yes"),
        # Bezugsgröße der Herzfrequenzzonen. Ohne sie ist in der Auswertung
        # nicht nachvollziehbar, wie die Warnschwellen dieser Sitzung zustande
        # kamen (Karvonen mit Ruhepuls oder ersatzweise Prozent von HFmax).
        "resting_hr_bpm":   record.get("resting_hr_bpm"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry – participants.json
# ─────────────────────────────────────────────────────────────────────────────

class ParticipantRegistry:
    """Lädt und speichert die pseudonymisierten Stammdaten aller Testpersonen."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.path     = os.path.join(base_dir, PARTICIPANTS_FILE)
        self._data    = {}
        self.load()

    # ── Persistenz ────────────────────────────────────────────────────────────

    def load(self):
        if not os.path.exists(self.path):
            self._data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._data = loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            # Nicht stillschweigend eine leere Liste weiterverwenden – sonst
            # würde ein späteres save() die vorhandenen Probandendaten
            # überschreiben.
            raise RuntimeError(
                f"participants.json konnte nicht gelesen werden ({e}). "
                "Datei bitte prüfen/sichern, bevor die Aufzeichnung fortgesetzt wird."
            )

    def save(self):
        os.makedirs(self.base_dir, exist_ok=True)
        # Atomar schreiben: ein Absturz mitten im Schreiben darf die bereits
        # erfassten Probanden nicht zerstören.
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── Zugriff ───────────────────────────────────────────────────────────────

    def ids(self) -> list:
        return sorted(self._data.keys())

    def get(self, participant_id: str) -> dict:
        return self._data.get(participant_id)

    def next_id(self) -> str:
        numbers = []
        for pid in self._data:
            if pid.startswith("P") and pid[1:].isdigit():
                numbers.append(int(pid[1:]))
        return f"P{(max(numbers) + 1) if numbers else 1:02d}"

    def add(self, record: dict):
        self._data[record["participant_id"]] = record
        self.save()

    def update(self, participant_id: str, **felder):
        """
        Ergänzt einen bestehenden Datensatz um einzelne Felder und speichert.

        Für Angaben, die erst nach dem Anlegen entstehen – gemessener Ruhepuls,
        die für diese Person eingestellten Fahrwerte. Die Stammdaten und die
        Einwilligung bleiben dabei unberührt: Es wird ausschließlich ergänzt,
        nie ein vorhandenes Feld gelöscht.
        """
        record = self._data.get(participant_id)
        if record is None:
            raise KeyError(f"Unbekannte Teilnehmer-ID: {participant_id}")
        record.update(felder)
        self.save()
        return record

    def label(self, participant_id: str) -> str:
        """Kurzbeschreibung für die Auswahlliste – ohne identifizierende Angaben."""
        r = self._data.get(participant_id)
        if not r:
            return participant_id
        return (f"{participant_id}  –  {r['age_years']} J. ({r['age_group']}), "
                f"{r['sex']}, {r['handedness']}")


# ─────────────────────────────────────────────────────────────────────────────
# Eingabemaske
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollFrame(ttk.Frame):
    """Scrollbarer Container, damit die Maske auch auf kleinen Displays passt."""

    def __init__(self, master, height=460):
        super().__init__(master)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                 height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event):
        if self._canvas.winfo_exists():
            self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def unbind_mousewheel(self):
        # bind_all gilt applikationsweit – beim Schließen wieder lösen, sonst
        # scrollt das Mausrad im Hauptfenster ins Leere eines toten Canvas.
        self._canvas.unbind_all("<MouseWheel>")


class ParticipantDialog(tk.Toplevel):
    """
    Modale Eingabemaske für eine neue Testperson.

    Ergebnis liegt nach dem Schließen in `self.result` (dict oder None).
    """

    def __init__(self, parent, participant_id: str):
        super().__init__(parent)
        self.title(f"Neue Testperson anlegen – {participant_id}")
        self.geometry("640x680")
        self.minsize(560, 480)
        self.transient(parent)
        self.resizable(True, True)

        self.participant_id = participant_id
        self.result = None

        # ── Eingabevariablen ─────────────────────────────────────────────────
        self.var_age        = tk.StringVar()
        self.var_sex        = tk.StringVar(value="")
        self.var_hand       = tk.StringVar(value="")
        self.var_tech       = tk.StringVar(value="")
        self.var_prior      = tk.StringVar(value="")
        self.var_parq       = {key: tk.StringVar(value="") for key, _ in PARQ_ITEMS}
        self.var_consent    = tk.BooleanVar(value=False)
        self.var_video      = tk.BooleanVar(value=False)

        self._build()

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.wait_window(self)

    # ── Aufbau ────────────────────────────────────────────────────────────────

    def _build(self):
        head = ttk.Frame(self, padding=(16, 14, 16, 6))
        head.pack(fill="x")
        ttk.Label(head, text=f"Testperson {self.participant_id}",
                  font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(head,
                  text="Es werden keine Namen, Geburtsdaten oder Kontaktdaten gespeichert.\n"
                       "Die Zuordnung zur Person erfolgt ausschließlich über diese ID auf "
                       "der unterschriebenen Einverständniserklärung.",
                  foreground="#555", wraplength=580,
                  justify="left").pack(anchor="w", pady=(4, 0))

        scroll = _ScrollFrame(self)
        scroll.pack(fill="both", expand=True, padx=16, pady=(10, 0))
        self._scroll = scroll
        body = scroll.inner

        # ── Abschnitt 1: Stammdaten ──────────────────────────────────────────
        self._section(body, "1  Stammdaten")

        row = ttk.Frame(body); row.pack(fill="x", pady=(2, 8))
        ttk.Label(row, text="Alter (Jahre):", width=22).pack(side="left")
        age_entry = ttk.Entry(row, textvariable=self.var_age, width=8)
        age_entry.pack(side="left")
        ttk.Label(row, text=f"({MIN_AGE}–{MAX_AGE})",
                  foreground="#777").pack(side="left", padx=(8, 0))
        age_entry.focus_set()

        self._radio_row(body, "Geschlecht:", self.var_sex, SEX_OPTIONS)
        self._radio_row(body, "Händigkeit:", self.var_hand, HANDEDNESS_OPTIONS)

        row = ttk.Frame(body); row.pack(fill="x", pady=(2, 8))
        ttk.Label(row, text="Technikaffinität:", width=22).pack(side="left", anchor="n")
        tech_box = ttk.Combobox(row, textvariable=self.var_tech, state="readonly",
                                width=24,
                                values=[TECH_AFFINITY_LABELS[i] for i in range(1, 6)])
        tech_box.pack(side="left")
        ttk.Label(body,
                  text="„Wie vertraut sind Sie im Umgang mit Apps oder Spielen auf "
                       "Smartphone/Tablet?“",
                  foreground="#777", wraplength=540,
                  justify="left").pack(anchor="w", padx=(22, 0), pady=(0, 8))

        self._radio_row(body, "Vorerfahrung mit Reha-/\nBewegungstechnologie:",
                        self.var_prior, [("ja", "ja"), ("nein", "nein")])

        # ── Abschnitt 2: PAR-Q ───────────────────────────────────────────────
        self._section(body, "2  PAR-Q – Screening vor körperlicher Aktivität")
        ttk.Label(body,
                  text="Bitte alle sieben Fragen mit Ja oder Nein beantworten. "
                       "Ein „Ja“ führt nicht automatisch zum Ausschluss, wird aber "
                       "dokumentiert und vor der Übung mit der Studienleitung besprochen.",
                  foreground="#555", wraplength=560,
                  justify="left").pack(anchor="w", pady=(0, 8))

        for idx, (key, text) in enumerate(PARQ_ITEMS, start=1):
            item = ttk.Frame(body)
            item.pack(fill="x", pady=(0, 6))
            ttk.Label(item, text=f"{idx}.  {text}", wraplength=430,
                      justify="left").pack(side="left", fill="x", expand=True)
            choice = ttk.Frame(item)
            choice.pack(side="right", anchor="n", padx=(10, 0))
            ttk.Radiobutton(choice, text="Ja",   variable=self.var_parq[key],
                            value="ja").pack(side="left")
            ttk.Radiobutton(choice, text="Nein", variable=self.var_parq[key],
                            value="nein").pack(side="left", padx=(6, 0))

        # ── Abschnitt 3: Einwilligung ────────────────────────────────────────
        self._section(body, "3  Einwilligung")
        ttk.Checkbutton(
            body, variable=self.var_consent,
            text="Die Probandeninformation wurde gelesen und die Einverständnis-\n"
                 "erklärung liegt unterschrieben vor (Pflicht für die Aufzeichnung).",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(
            body, variable=self.var_video,
            text="Freiwillig: Ich bin mit einer Videoaufzeichnung der Übung "
                 "einverstanden.",
        ).pack(anchor="w")
        ttk.Label(body,
                  text="Ohne diese Zustimmung werden ausschließlich die nicht "
                       "identifizierenden Messwerte (Winkel, Abstand, Sensorik, "
                       "Herzfrequenz) aufgezeichnet – kein Video. Die Zustimmung kann "
                       "vor jeder einzelnen Sitzung erneut widerrufen werden.",
                  foreground="#777", wraplength=560,
                  justify="left").pack(anchor="w", padx=(22, 0), pady=(2, 12))

        # ── Buttons ──────────────────────────────────────────────────────────
        footer = ttk.Frame(self, padding=(16, 10))
        footer.pack(fill="x")
        ttk.Button(footer, text="Abbrechen", command=self._cancel).pack(side="right")
        ttk.Button(footer, text="Speichern und übernehmen",
                   command=self._save).pack(side="right", padx=(0, 8))

    def _section(self, parent, title):
        ttk.Separator(parent).pack(fill="x", pady=(12, 6))
        ttk.Label(parent, text=title,
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

    def _radio_row(self, parent, label, variable, options):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 8))
        ttk.Label(row, text=label, width=22, justify="left").pack(side="left", anchor="n")
        for value, text in options:
            ttk.Radiobutton(row, text=text, variable=variable,
                            value=value).pack(side="left", padx=(0, 10))

    # ── Aktionen ──────────────────────────────────────────────────────────────

    def _validate(self):
        raw_age = self.var_age.get().strip()
        if not raw_age.isdigit() or not (MIN_AGE <= int(raw_age) <= MAX_AGE):
            return f"Bitte ein gültiges Alter zwischen {MIN_AGE} und {MAX_AGE} eintragen."
        if not self.var_sex.get():
            return "Bitte das Geschlecht auswählen („keine Angabe“ ist möglich)."
        if not self.var_hand.get():
            return "Bitte die Händigkeit auswählen."
        if not self.var_tech.get():
            return "Bitte die Technikaffinität auswählen."
        if not self.var_prior.get():
            return "Bitte angeben, ob Vorerfahrung mit Reha-/Bewegungstechnologie besteht."
        for idx, (key, _) in enumerate(PARQ_ITEMS, start=1):
            if not self.var_parq[key].get():
                return f"PAR-Q Frage {idx} ist noch nicht beantwortet."
        if not self.var_consent.get():
            return ("Ohne vorliegende unterschriebene Einverständniserklärung darf "
                    "keine Aufzeichnung erfolgen.")
        return None

    def _save(self):
        problem = self._validate()
        if problem:
            messagebox.showwarning("Eingabe unvollständig", problem, parent=self)
            return

        age  = int(self.var_age.get().strip())
        parq = {key: (self.var_parq[key].get() == "ja") for key, _ in PARQ_ITEMS}

        self.result = {
            "participant_id":   self.participant_id,
            "created_iso":      datetime.now().isoformat(timespec="seconds"),
            "age_years":        age,
            "age_group":        age_group(age),
            "sex":              self.var_sex.get(),
            "handedness":       self.var_hand.get(),
            "tech_affinity":    int(self.var_tech.get()[0]),
            "prior_experience": self.var_prior.get() == "ja",
            "parq":             parq,
            "parq_any_yes":     any(parq.values()),
            "consent": {
                "declaration_signed": True,
                "video":              bool(self.var_video.get()),
                "recorded_iso":       datetime.now().isoformat(timespec="seconds"),
            },
        }

        if self.result["parq_any_yes"]:
            messagebox.showinfo(
                "PAR-Q Hinweis",
                "Mindestens eine PAR-Q-Frage wurde mit „Ja“ beantwortet.\n\n"
                "Das ist dokumentiert. Bitte vor Beginn der Übung mit der "
                "Studienleitung abklären, ob die Teilnahme erfolgen kann.",
                parent=self,
            )
        self._close()

    def _cancel(self):
        self.result = None
        self._close()

    def _close(self):
        self._scroll.unbind_mousewheel()
        self.grab_release()
        self.destroy()


def ask_new_participant(parent, registry: ParticipantRegistry) -> str:
    """
    Öffnet die Eingabemaske, speichert bei Erfolg und gibt die neue ID zurück
    (oder None bei Abbruch).
    """
    dialog = ParticipantDialog(parent, registry.next_id())
    if dialog.result is None:
        return None
    registry.add(dialog.result)
    return dialog.result["participant_id"]
