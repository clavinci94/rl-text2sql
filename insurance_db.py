"""Case study: a realistic (synthetic) Swiss insurance database for German business questions.

Spider databases are small, clean and English. Real company databases are not: they have
cryptic legacy column names, a generic code table, business terms that appear nowhere in the
schema, NULLs, history tables and data that only starts with a system migration. This module
builds such a database for the fictional "Muster Versicherung AG" so the model can be tested
on a realistic, held-out use case. ALL DATA IS SYNTHETIC (fixed seed, no real persons).

    python insurance_db.py build     # -> data/versicherung.sqlite
    python insurance_db.py check     # run all gold queries of the test set, print a summary
    python insurance_db.py schema    # print the prompt schema with data catalogue

The test questions live in insurance_questions.json (German, French, Italian; some on purpose
unanswerable with this database). The model never trains on this database.
"""
import json
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
DB_ID = "versicherung"
DB_FILE = ROOT / "data" / "versicherung.sqlite"
QUESTIONS_FILE = ROOT / "insurance_questions.json"
STICHTAG = date(2026, 6, 30)        # data as of this day ("Datenstand")
CLAIMS_FROM = date(2021, 1, 1)      # claims system migrated 2021: older claims are not in this DB

# ---------------------------------------------------------------- schema (legacy style)

DDL = """
CREATE TABLE T_CODE (CD_TYP TEXT, CD TEXT, BEZ_DE TEXT, BEZ_FR TEXT, PRIMARY KEY (CD_TYP, CD));
CREATE TABLE T_KANTON (KT_CD TEXT PRIMARY KEY, KT_NAME TEXT, REGION TEXT);
CREATE TABLE T_AGENTUR (AGT_ID INTEGER PRIMARY KEY, AGT_NAME TEXT, KT_CD TEXT, LEITER_MA_ID INTEGER,
  FOREIGN KEY (KT_CD) REFERENCES T_KANTON(KT_CD));
CREATE TABLE T_MITARBEITER (MA_ID INTEGER PRIMARY KEY, NAME TEXT, VORNAME TEXT, FUNKTION_CD TEXT,
  AGT_ID INTEGER, EINTRITT_DAT TEXT, FOREIGN KEY (AGT_ID) REFERENCES T_AGENTUR(AGT_ID));
CREATE TABLE T_PARTNER (PTR_ID INTEGER PRIMARY KEY, PTR_TYP_CD TEXT, NAME TEXT, VORNAME TEXT,
  GEB_DAT TEXT, GESCHL_CD TEXT, SPR_CD TEXT, PLZ TEXT, ORT TEXT, KT_CD TEXT, EROEFF_DAT TEXT, STAT_CD TEXT,
  FOREIGN KEY (KT_CD) REFERENCES T_KANTON(KT_CD));
CREATE TABLE T_SPARTE (SPARTE_CD TEXT PRIMARY KEY, SPARTE_BEZ TEXT);
CREATE TABLE T_PRODUKT (PROD_CD TEXT PRIMARY KEY, PROD_BEZ TEXT, SPARTE_CD TEXT,
  FOREIGN KEY (SPARTE_CD) REFERENCES T_SPARTE(SPARTE_CD));
CREATE TABLE T_VERTRAG (VTR_ID INTEGER PRIMARY KEY, VTR_NR TEXT, PTR_ID INTEGER, PROD_CD TEXT,
  MA_ID_BER INTEGER, BEGINN_DAT TEXT, ABL_DAT TEXT, STORNO_DAT TEXT, STAT_CD TEXT, ZAHLW_CD TEXT,
  JPRAEMIE REAL, SELBSTBEH REAL,
  FOREIGN KEY (PTR_ID) REFERENCES T_PARTNER(PTR_ID), FOREIGN KEY (PROD_CD) REFERENCES T_PRODUKT(PROD_CD),
  FOREIGN KEY (MA_ID_BER) REFERENCES T_MITARBEITER(MA_ID));
CREATE TABLE T_DCK_ART (DCK_ART_CD TEXT PRIMARY KEY, BEZ TEXT, SPARTE_CD TEXT);
CREATE TABLE T_DECKUNG (DCK_ID INTEGER PRIMARY KEY, VTR_ID INTEGER, DCK_ART_CD TEXT, VERS_SUMME REAL,
  FOREIGN KEY (VTR_ID) REFERENCES T_VERTRAG(VTR_ID), FOREIGN KEY (DCK_ART_CD) REFERENCES T_DCK_ART(DCK_ART_CD));
CREATE TABLE T_OBJ_FZG (FZG_ID INTEGER PRIMARY KEY, VTR_ID INTEGER, KENNZ TEXT, MARKE TEXT, MODELL TEXT,
  JG INTEGER, KATPREIS REAL, FOREIGN KEY (VTR_ID) REFERENCES T_VERTRAG(VTR_ID));
CREATE TABLE T_SCHADEN (SCH_ID INTEGER PRIMARY KEY, SCH_NR TEXT, VTR_ID INTEGER, EREIG_DAT TEXT,
  MELD_DAT TEXT, URS_CD TEXT, STAT_CD TEXT, SB_MA_ID INTEGER, RESERVE REAL, KT_CD_EREIG TEXT,
  BETRUG_VERD_FLG INTEGER,
  FOREIGN KEY (VTR_ID) REFERENCES T_VERTRAG(VTR_ID), FOREIGN KEY (SB_MA_ID) REFERENCES T_MITARBEITER(MA_ID));
CREATE TABLE T_SCH_ZLG (ZLG_ID INTEGER PRIMARY KEY, SCH_ID INTEGER, ZLG_DAT TEXT, BETRAG REAL,
  EMPF_TYP_CD TEXT, FOREIGN KEY (SCH_ID) REFERENCES T_SCHADEN(SCH_ID));
CREATE TABLE T_SCH_STAT_HIST (SCH_ID INTEGER, STAT_CD TEXT, AB_DAT TEXT,
  FOREIGN KEY (SCH_ID) REFERENCES T_SCHADEN(SCH_ID));
CREATE TABLE T_RECHNUNG (RG_ID INTEGER PRIMARY KEY, VTR_ID INTEGER, RG_DAT TEXT, FAELLIG_DAT TEXT,
  BETRAG REAL, BEZ_DAT TEXT, MAHNSTUFE INTEGER, FOREIGN KEY (VTR_ID) REFERENCES T_VERTRAG(VTR_ID));
"""

# Data catalogue: what a company's data steward would write down. Used only in the
# "--catalog" prompt variant, to measure how much documented metadata helps.
CATALOG = {
    "T_CODE": ("Generische Codetabelle des Altsystems", {
        "CD_TYP": "Codetyp, z.B. VTR_STAT, SCH_STAT, SCH_URS, ZAHLW, FUNKTION, EMPF_TYP",
        "CD": "Code innerhalb des Typs", "BEZ_DE": "Bezeichnung deutsch", "BEZ_FR": "Bezeichnung französisch"}),
    "T_KANTON": ("Kantone der Schweiz", {
        "KT_CD": "Kantonskürzel, z.B. ZH, AG, VD", "KT_NAME": "Name des Kantons",
        "REGION": "Sprachregion: Deutschschweiz, Romandie oder Tessin"}),
    "T_AGENTUR": ("Generalagenturen", {"LEITER_MA_ID": "Mitarbeiter, der die Agentur leitet"}),
    "T_MITARBEITER": ("Mitarbeitende", {
        "FUNKTION_CD": "Funktion: BER = Kundenberater, SB = Schadenbearbeiter, LTG = Agenturleiter",
        "EINTRITT_DAT": "Eintrittsdatum"}),
    "T_PARTNER": ("Kunden (Versicherungsnehmer), Personen und Unternehmen", {
        "PTR_TYP_CD": "P = natürliche Person, U = Unternehmen", "NAME": "Nachname oder Firmenname",
        "GEB_DAT": "Geburtsdatum, leer bei Unternehmen", "GESCHL_CD": "M oder W, leer bei Unternehmen",
        "SPR_CD": "Korrespondenzsprache DE, FR, IT", "KT_CD": "Wohnkanton",
        "EROEFF_DAT": "Kunde seit (Beginn des ersten Vertrags)", "STAT_CD": "A = aktiver Kunde, I = inaktiv"}),
    "T_SPARTE": ("Versicherungssparten", {}),
    "T_PRODUKT": ("Versicherungsprodukte", {}),
    "T_VERTRAG": ("Versicherungsverträge (Policen)", {
        "VTR_NR": "Policennummer", "PTR_ID": "Versicherungsnehmer", "MA_ID_BER": "zuständiger Kundenberater",
        "BEGINN_DAT": "Vertragsbeginn", "ABL_DAT": "Ablaufdatum", "STORNO_DAT": "Kündigungsdatum, falls storniert",
        "STAT_CD": "AKT = aktiv, STO = storniert, SIS = sistiert, ABL = abgelaufen (Codetyp VTR_STAT)",
        "ZAHLW_CD": "Zahlweise J/H/Q/M (Codetyp ZAHLW)", "JPRAEMIE": "Jahresprämie in CHF",
        "SELBSTBEH": "Selbstbehalt in CHF"}),
    "T_DCK_ART": ("Deckungsarten", {}),
    "T_DECKUNG": ("Versicherte Deckungen pro Vertrag", {"VERS_SUMME": "Versicherungssumme in CHF, leer = unbegrenzt"}),
    "T_OBJ_FZG": ("Versicherte Fahrzeuge (Sparte Motorfahrzeug)", {
        "KENNZ": "Kontrollschild", "JG": "Jahrgang", "KATPREIS": "Katalogpreis in CHF"}),
    "T_SCHADEN": ("Schadenfälle, erfasst erst seit der Systemmigration 2021", {
        "EREIG_DAT": "Datum des Schadenereignisses", "MELD_DAT": "Datum der Schadenmeldung",
        "URS_CD": "Schadenursache (Codetyp SCH_URS)",
        "STAT_CD": "OFF = offen, PRF = in Prüfung, ABG = abgeschlossen, ABL = abgelehnt (Codetyp SCH_STAT)",
        "SB_MA_ID": "zuständiger Schadenbearbeiter", "RESERVE": "Rückstellung in CHF für offene Schäden",
        "KT_CD_EREIG": "Kanton, in dem der Schaden passiert ist", "BETRUG_VERD_FLG": "1 = Betrugsverdacht"}),
    "T_SCH_ZLG": ("Schadenzahlungen", {"BETRAG": "ausbezahlter Betrag in CHF",
                                        "EMPF_TYP_CD": "Empfänger: KUNDE, WERKST (Werkstatt), DRITT (Geschädigter)"}),
    "T_SCH_STAT_HIST": ("Statusverlauf der Schäden", {"AB_DAT": "Status gültig ab"}),
    "T_RECHNUNG": ("Prämienrechnungen der letzten 12 Monate", {
        "FAELLIG_DAT": "Fälligkeitsdatum", "BEZ_DAT": "Zahlungseingang, leer = noch nicht bezahlt",
        "MAHNSTUFE": "0 = keine Mahnung, 1 bis 3 = Mahnstufe"}),
}

GLOSSARY = {
    "Stichtag": f"Datenstand {STICHTAG.isoformat()}. 'Heute' bedeutet dieses Datum.",
    "Kunde": "Eintrag in T_PARTNER (Personen und Unternehmen).",
    "aktiver Kunde": "T_PARTNER.STAT_CD = 'A'.",
    "aktiver Vertrag, Bestand": "T_VERTRAG.STAT_CD = 'AKT'.",
    "Prämienvolumen": "Summe JPRAEMIE der aktiven Verträge.",
    "Schadenaufwand": "Summe T_SCH_ZLG.BETRAG.",
    "Schadenquote": "Schadenaufwand eines Kalenderjahres (nach Zahlungsdatum) geteilt durch das Prämienvolumen "
                    "(vereinfachte Projektdefinition).",
    "offener Schaden": "T_SCHADEN.STAT_CD IN ('OFF', 'PRF').",
    "Grossschaden": "Schaden mit Schadenaufwand über 50'000 CHF.",
    "überfällige Rechnung": f"BEZ_DAT IS NULL AND FAELLIG_DAT < '{STICHTAG.isoformat()}'.",
    "Kasko": "Deckungsarten TK (Teilkasko) und VK (Vollkasko).",
    "Sprachregion": "T_KANTON.REGION (Deutschschweiz, Romandie, Tessin).",
}

# ---------------------------------------------------------------- reference data

KANTONE = [  # code, name, region, weight (approx. population share), towns
    ("ZH", "Zürich", "Deutschschweiz", 18, ["Zürich", "Winterthur", "Uster", "Dübendorf"]),
    ("BE", "Bern", "Deutschschweiz", 12, ["Bern", "Biel/Bienne", "Thun", "Köniz"]),
    ("LU", "Luzern", "Deutschschweiz", 5, ["Luzern", "Emmen", "Kriens"]),
    ("UR", "Uri", "Deutschschweiz", 0.5, ["Altdorf"]),
    ("SZ", "Schwyz", "Deutschschweiz", 2, ["Schwyz", "Einsiedeln"]),
    ("OW", "Obwalden", "Deutschschweiz", 0.5, ["Sarnen"]),
    ("NW", "Nidwalden", "Deutschschweiz", 0.5, ["Stans"]),
    ("GL", "Glarus", "Deutschschweiz", 0.5, ["Glarus"]),
    ("ZG", "Zug", "Deutschschweiz", 1.5, ["Zug", "Baar"]),
    ("FR", "Freiburg", "Romandie", 3.5, ["Fribourg", "Bulle"]),
    ("SO", "Solothurn", "Deutschschweiz", 3, ["Solothurn", "Olten", "Grenchen"]),
    ("BS", "Basel-Stadt", "Deutschschweiz", 2.2, ["Basel", "Riehen"]),
    ("BL", "Basel-Landschaft", "Deutschschweiz", 3.3, ["Liestal", "Allschwil", "Reinach"]),
    ("SH", "Schaffhausen", "Deutschschweiz", 1, ["Schaffhausen"]),
    ("AR", "Appenzell Ausserrhoden", "Deutschschweiz", 0.6, ["Herisau"]),
    ("AI", "Appenzell Innerrhoden", "Deutschschweiz", 0.2, ["Appenzell"]),
    ("SG", "St. Gallen", "Deutschschweiz", 6, ["St. Gallen", "Rapperswil-Jona", "Wil"]),
    ("GR", "Graubünden", "Deutschschweiz", 2.3, ["Chur", "Davos"]),
    ("AG", "Aargau", "Deutschschweiz", 8, ["Aarau", "Baden", "Wettingen", "Lenzburg", "Wohlen"]),
    ("TG", "Thurgau", "Deutschschweiz", 3.2, ["Frauenfeld", "Kreuzlingen"]),
    ("TI", "Tessin", "Tessin", 4, ["Lugano", "Bellinzona", "Locarno", "Mendrisio"]),
    ("VD", "Waadt", "Romandie", 9, ["Lausanne", "Yverdon-les-Bains", "Montreux", "Nyon"]),
    ("VS", "Wallis", "Romandie", 4, ["Sion", "Sierre", "Martigny", "Brig-Glis"]),
    ("NE", "Neuenburg", "Romandie", 2, ["Neuchâtel", "La Chaux-de-Fonds"]),
    ("GE", "Genf", "Romandie", 5.8, ["Genève", "Vernier", "Lancy"]),
    ("JU", "Jura", "Romandie", 0.9, ["Delémont", "Porrentruy"]),
]
PLZ_BASE = {"ZH": 8000, "BE": 3000, "LU": 6000, "UR": 6460, "SZ": 6430, "OW": 6060, "NW": 6370, "GL": 8750,
            "ZG": 6300, "FR": 1700, "SO": 4500, "BS": 4000, "BL": 4100, "SH": 8200, "AR": 9100, "AI": 9050,
            "SG": 9000, "GR": 7000, "AG": 5000, "TG": 8500, "TI": 6500, "VD": 1000, "VS": 1950, "NE": 2000,
            "GE": 1200, "JU": 2800}

VORNAMEN = {
    "DE": {"M": ["Lukas", "Noah", "David", "Simon", "Marco", "Daniel", "Thomas", "Stefan", "Martin", "Peter",
                 "Hans", "Beat", "Reto", "Jonas", "Elias", "Urs", "Patrick", "Fabian", "Michael", "Christian"],
           "W": ["Anna", "Laura", "Lea", "Sarah", "Nina", "Sandra", "Andrea", "Claudia", "Monika", "Ursula",
                 "Corinne", "Fabienne", "Mia", "Lina", "Barbara", "Daniela", "Nicole", "Petra", "Julia", "Esther"]},
    "FR": {"M": ["Julien", "Nicolas", "Mathieu", "Antoine", "Pierre", "Olivier", "Luc", "Yves"],
           "W": ["Camille", "Sophie", "Élodie", "Léa", "Chloé", "Isabelle", "Nathalie", "Céline"]},
    "IT": {"M": ["Matteo", "Alessandro", "Lorenzo", "Davide", "Marco", "Luca"],
           "W": ["Giulia", "Chiara", "Francesca", "Sara", "Elena", "Martina"]},
}
NACHNAMEN = {"DE": ["Müller", "Meier", "Schmid", "Keller", "Weber", "Huber", "Schneider", "Meyer", "Steiner",
                    "Fischer", "Gerber", "Brunner", "Baumann", "Frei", "Zimmermann", "Moser", "Widmer", "Wyss",
                    "Graf", "Roth", "Suter", "Bachmann", "Kunz", "Lüthi", "Bühler"],
             "FR": ["Favre", "Rochat", "Bonvin", "Perrin", "Morel", "Girard", "Dubois", "Blanc", "Rey", "Chevalley"],
             "IT": ["Bernasconi", "Rossi", "Ferrari", "Pedrazzini", "Bianchi", "Fontana", "Galli", "Crivelli"]}
FIRMEN = ["Bau", "Transport", "Treuhand", "Gartenbau", "Elektro", "Informatik", "Holzbau", "Logistik",
          "Immobilien", "Sanitär", "Malerei", "Consulting"]

SPARTEN = [("MF", "Motorfahrzeug"), ("HR", "Hausrat"), ("PH", "Privathaftpflicht"), ("GB", "Gebäude"),
           ("RS", "Rechtsschutz"), ("RE", "Reise")]
PRODUKTE = [  # code, name, sparte, weight, premium range
    ("MF01", "Auto Basis", "MF", 10, (520, 1100)), ("MF02", "Auto Komfort", "MF", 14, (900, 2300)),
    ("HR01", "Hausrat Standard", "HR", 14, (140, 420)), ("HR02", "Hausrat Plus", "HR", 8, (260, 690)),
    ("PH01", "Privathaftpflicht", "PH", 16, (85, 190)), ("GB01", "Gebäude Kompakt", "GB", 6, (420, 2600)),
    ("RS01", "Rechtsschutz Privat", "RS", 7, (210, 390)), ("RE01", "Reise Jahresschutz", "RE", 6, (95, 260)),
]
DCK_ARTEN = [("HAFT", "Motorfahrzeug-Haftpflicht", "MF"), ("TK", "Teilkasko", "MF"), ("VK", "Vollkasko", "MF"),
             ("PARK", "Parkschaden", "MF"), ("HR_GRUND", "Hausrat Grunddeckung", "HR"),
             ("HR_DIEB", "Diebstahl auswärts", "HR"), ("HR_GLAS", "Glasbruch Mobiliar", "HR"),
             ("PH_GRUND", "Privathaftpflicht", "PH"), ("GB_FEUER", "Gebäude Feuer", "GB"),
             ("GB_WASSER", "Gebäude Wasser", "GB"), ("GB_ELEM", "Gebäude Elementar", "GB"),
             ("RS_GRUND", "Rechtsschutz Privat", "RS"), ("RE_ANN", "Annullierungskosten", "RE"),
             ("RE_ASS", "Reise-Assistance", "RE")]
CODES = [("VTR_STAT", "AKT", "aktiv", "actif"), ("VTR_STAT", "STO", "storniert", "résilié"),
         ("VTR_STAT", "SIS", "sistiert", "suspendu"), ("VTR_STAT", "ABL", "abgelaufen", "échu"),
         ("SCH_STAT", "OFF", "offen", "ouvert"), ("SCH_STAT", "PRF", "in Prüfung", "en examen"),
         ("SCH_STAT", "ABG", "abgeschlossen", "clôturé"), ("SCH_STAT", "ABL", "abgelehnt", "refusé"),
         ("ZAHLW", "J", "jährlich", "annuel"), ("ZAHLW", "H", "halbjährlich", "semestriel"),
         ("ZAHLW", "Q", "vierteljährlich", "trimestriel"), ("ZAHLW", "M", "monatlich", "mensuel"),
         ("FUNKTION", "BER", "Kundenberater", "conseiller"), ("FUNKTION", "SB", "Schadenbearbeiter", "gestionnaire sinistres"),
         ("FUNKTION", "LTG", "Agenturleiter", "chef d'agence"),
         ("EMPF_TYP", "KUNDE", "Kunde", "client"), ("EMPF_TYP", "WERKST", "Werkstatt", "garage"),
         ("EMPF_TYP", "DRITT", "Geschädigter Dritter", "tiers lésé"),
         ("SCH_URS", "KOLL", "Kollision", "collision"), ("SCH_URS", "PARK", "Parkschaden", "dommage de parking"),
         ("SCH_URS", "GLAS", "Glasbruch", "bris de glace"), ("SCH_URS", "HAGEL", "Hagel", "grêle"),
         ("SCH_URS", "MARDER", "Marderschaden", "dégâts de fouine"), ("SCH_URS", "DIEB", "Diebstahl", "vol"),
         ("SCH_URS", "WASS", "Wasserschaden", "dégât d'eau"), ("SCH_URS", "FEUER", "Feuer", "incendie"),
         ("SCH_URS", "STURM", "Sturm", "tempête"), ("SCH_URS", "HAFT", "Haftpflichtfall", "cas RC"),
         ("SCH_URS", "RECHT", "Rechtsfall", "litige"), ("SCH_URS", "ANNUL", "Reiseannullierung", "annulation"),
         ("SCH_URS", "KRANK", "Krankheit im Ausland", "maladie à l'étranger")]
URSACHEN = {  # per sparte: (cause, weight, lognormal median CHF, sigma)
    "MF": [("KOLL", 30, 6500, .8), ("PARK", 22, 1800, .5), ("GLAS", 20, 900, .35), ("HAGEL", 12, 5200, .5),
           ("MARDER", 8, 700, .4), ("DIEB", 3, 21000, .6), ("HAFT", 5, 9000, .9)],
    "HR": [("DIEB", 30, 2600, .8), ("WASS", 25, 4200, .8), ("GLAS", 20, 650, .4), ("FEUER", 8, 14000, 1.0),
           ("HAGEL", 7, 1900, .5), ("STURM", 10, 1500, .6)],
    "PH": [("HAFT", 100, 1700, 1.0)],
    "GB": [("WASS", 40, 9000, .8), ("HAGEL", 25, 12000, .6), ("STURM", 15, 7000, .7), ("FEUER", 20, 45000, 1.1)],
    "RS": [("RECHT", 100, 3800, .8)],
    "RE": [("ANNUL", 70, 1600, .6), ("KRANK", 30, 4200, 1.0)],
}
CLAIM_RATE = {"MF": .16, "HR": .07, "PH": .05, "GB": .06, "RS": .05, "RE": .07}  # per contract-year
SEVERITY = {"MF": .85, "HR": .45, "PH": .6, "GB": .65, "RS": .9, "RE": .5}  # calibrates loss ratios to ~0.5-0.8
MARKEN = [("VW", ["Golf", "Tiguan", "Polo", "ID.4"]), ("Skoda", ["Octavia", "Kodiaq", "Enyaq"]),
          ("Toyota", ["Yaris", "Corolla", "RAV4"]), ("BMW", ["320d", "X3", "i4"]),
          ("Mercedes-Benz", ["C 200", "GLC", "A 180"]), ("Audi", ["A4", "Q5", "Q4 e-tron"]),
          ("Tesla", ["Model 3", "Model Y"]), ("Renault", ["Clio", "Captur"]), ("Peugeot", ["208", "3008"]),
          ("Ford", ["Focus", "Kuga"]), ("Volvo", ["XC40", "XC60"]), ("Fiat", ["500", "Panda"])]


def _d(x):
    return x.isoformat() if x else None


def _rand_date(rng, a, b):
    return a + timedelta(days=rng.randrange((b - a).days + 1))


def build(path=DB_FILE, seed=2026):
    rng = random.Random(seed)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.executemany("INSERT INTO T_CODE VALUES (?,?,?,?)", CODES)
    con.executemany("INSERT INTO T_KANTON VALUES (?,?,?)", [k[:3] for k in KANTONE])
    con.executemany("INSERT INTO T_SPARTE VALUES (?,?)", SPARTEN)
    con.executemany("INSERT INTO T_PRODUKT VALUES (?,?,?)", [p[:3] for p in PRODUKTE])
    con.executemany("INSERT INTO T_DCK_ART VALUES (?,?,?)", DCK_ARTEN)
    kt = {k[0]: k for k in KANTONE}
    lang_of = lambda k: {"Romandie": "FR", "Tessin": "IT"}.get(kt[k][2], "DE")

    # agencies and staff
    agt_kt = ["ZH", "ZH", "BE", "BS", "LU", "SG", "AG", "VD", "GE", "TI", "VS"]
    agencies, staff, ma = [], [], 0

    def person_name(lang, sex=None):
        sex = sex or rng.choice("MW")
        return rng.choice(NACHNAMEN[lang]), rng.choice(VORNAMEN[lang][sex]), sex

    for a_id, k in enumerate(agt_kt, 1):
        town = kt[k][4][0] + (" Nord" if k == "ZH" and a_id == 2 else "")
        lang = lang_of(k)
        members = [("LTG", 1)] + [("BER", rng.randint(3, 5)), ("SB", rng.randint(1, 2))]
        leiter = None
        for fn, n in members:
            for _ in range(n):
                ma += 1
                nm, vn, _s = person_name(lang)
                staff.append((ma, nm, vn, fn, a_id, _d(_rand_date(rng, date(1998, 1, 1), date(2025, 12, 1)))))
                leiter = leiter or ma
        agencies.append((a_id, f"Generalagentur {town}", k, leiter))
    con.executemany("INSERT INTO T_AGENTUR VALUES (?,?,?,?)", agencies)
    con.executemany("INSERT INTO T_MITARBEITER VALUES (?,?,?,?,?,?)", staff)
    berater_by_agt = {}
    for s in staff:
        if s[3] == "BER":
            berater_by_agt.setdefault(s[4], []).append(s[0])
    sb_all = [s[0] for s in staff if s[3] == "SB"]
    agt_by_region = {}
    for a in agencies:
        agt_by_region.setdefault(kt[a[2]][2], []).append(a[0])

    # partners
    partners, n_partner = [], 2400
    kt_codes, kt_w = [k[0] for k in KANTONE], [k[3] for k in KANTONE]
    for pid in range(1, n_partner + 1):
        k = rng.choices(kt_codes, kt_w)[0]
        lang = lang_of(k)
        if k in ("BE", "FR", "VS") and rng.random() < .3:
            lang = "FR" if k == "BE" else "DE"  # bilingual cantons
        town = rng.choice(kt[k][4])
        plz = str(PLZ_BASE[k] + rng.randrange(0, 90) * (1 if PLZ_BASE[k] % 100 else 1))
        if rng.random() < .07:  # company
            nm = f"{rng.choice(NACHNAMEN[lang])} {rng.choice(FIRMEN)} AG"
            partners.append([pid, "U", nm, None, None, None, lang, plz, town, k, None, None])
        else:
            nm, vn, sex = person_name(lang)
            geb = _rand_date(rng, date(1938, 1, 1), date(2006, 6, 1))
            partners.append([pid, "P", nm, vn, _d(geb), sex, lang, plz, town, k, None, None])

    # contracts
    vtr, dck, fzg, vtr_id, dck_id, fzg_id = [], [], [], 0, 0, 0
    prod_codes, prod_w = [p[0] for p in PRODUKTE], [p[3] for p in PRODUKTE]
    prod = {p[0]: p for p in PRODUKTE}
    for p in partners:
        n = rng.choices([1, 2, 3, 4, 5], [38, 30, 18, 9, 5])[0] if p[1] == "P" else rng.randint(1, 3)
        region = kt[p[9]][2]
        agt = rng.choice(agt_by_region.get(region) or [1])
        first = None
        chosen = set()
        for _ in range(n):
            pc = rng.choices(prod_codes, prod_w)[0]
            if p[1] == "U":
                pc = rng.choice(["MF02", "GB01", "RS01"])
            if pc in chosen and pc[:2] not in ("MF",):
                continue
            chosen.add(pc)
            vtr_id += 1
            beg = _rand_date(rng, date(2012, 1, 1), date(2026, 6, 15))
            years = rng.choice([1, 3, 5])
            abl = date(beg.year + years, beg.month, min(beg.day, 28))
            r = rng.random()
            storno = None
            if r < .11 and beg < date(2026, 3, 1):
                stat = "STO"
                storno = _rand_date(rng, beg + timedelta(days=60), min(STICHTAG, abl))
            elif r < .13 and pc[:2] == "MF":
                stat = "SIS"
            elif abl < STICHTAG and r < .28:
                stat = "ABL"
            else:
                stat = "AKT"
                while abl < STICHTAG:  # tacit renewal
                    abl = date(abl.year + years, abl.month, abl.day)
            lo, hi = prod[pc][4]
            praemie = round(rng.uniform(lo, hi) * (1.6 if p[1] == "U" else 1.0) / 5) * 5
            zw = rng.choices("JHQM", [55, 20, 15, 10])[0]
            sb = {"MF": rng.choice([0, 500, 1000]), "HR": 200, "PH": rng.choice([0, 200]), "GB": 500,
                  "RS": 300, "RE": 0}[pc[:2]]
            ber = rng.choice(berater_by_agt[agt])
            vtr.append((vtr_id, f"{pc[:2]}-{100000 + vtr_id * 7 % 900000:06d}", p[0], pc, ber, _d(beg), _d(abl),
                        _d(storno), stat, zw, float(praemie), float(sb)))
            first = beg if first is None or beg < first else first
            for code, _bez, sp in DCK_ARTEN:
                if sp != pc[:2]:
                    continue
                if code in ("TK", "PARK") and pc == "MF01" and rng.random() < .6:
                    continue
                if code == "VK" and (pc == "MF01" or rng.random() < .35):
                    continue
                if code == "HR_DIEB" and pc == "HR01":
                    continue
                dck_id += 1
                summe = {"HAFT": None, "PH_GRUND": 5_000_000.0, "RS_GRUND": 300_000.0}.get(
                    code, float(rng.choice([20000, 50000, 80000, 120000, 250000])) if sp in ("HR", "MF")
                    else float(rng.choice([450000, 650000, 900000, 1200000])) if sp == "GB" else 10000.0)
                dck.append((dck_id, vtr_id, code, summe))
            if pc[:2] == "MF":
                fzg_id += 1
                marke, modelle = rng.choice(MARKEN)
                jg = rng.randint(max(2006, beg.year - 8), min(2026, beg.year + 1))
                katp = float(round(rng.uniform(19000, 95000) * (1.3 if marke in ("BMW", "Mercedes-Benz", "Audi", "Tesla") else 1), -2))
                fzg.append((fzg_id, vtr_id, f"{p[9]} {rng.randint(1000, 999999)}", marke, rng.choice(modelle), jg, katp))
        p[10] = _d(first)
    stat_by_ptr = {}
    for v in vtr:
        stat_by_ptr[v[2]] = stat_by_ptr.get(v[2], False) or v[8] == "AKT"
    for p in partners:
        p[11] = "A" if stat_by_ptr.get(p[0]) else "I"
    con.executemany("INSERT INTO T_PARTNER VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", partners)
    con.executemany("INSERT INTO T_VERTRAG VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", vtr)
    con.executemany("INSERT INTO T_DECKUNG VALUES (?,?,?,?)", dck)
    con.executemany("INSERT INTO T_OBJ_FZG VALUES (?,?,?,?,?,?,?)", fzg)

    # claims (only since migration), with hail clusters in summer 2021 and 2024
    sch, zlg, hist, sch_id, zlg_id = [], [], [], 0, 0
    ptr_kt = {p[0]: p[9] for p in partners}
    for v in vtr:
        sp = v[3][:2]
        beg, end = date.fromisoformat(v[5]), STICHTAG
        if v[7]:
            end = date.fromisoformat(v[7])
        elif v[8] == "ABL":
            end = date.fromisoformat(v[6])
        a, b = max(beg, CLAIMS_FROM), min(end, STICHTAG - timedelta(days=3))
        if a >= b:
            continue
        years = (b - a).days / 365.25
        lam = CLAIM_RATE[sp] * years * (1.4 if ptr_kt[v[2]] in ("GE", "BS", "ZH") and sp == "MF" else 1)
        k = 0
        while rng.random() < lam / (k + 1) and k < 6:  # rough Poisson-like count
            k += 1
        for _ in range(k):
            causes = URSACHEN[sp]
            urs, _w, med, sig = rng.choices(causes, [c[1] for c in causes])[0]
            ereig = _rand_date(rng, a, b)
            if urs == "HAGEL":  # hail comes in storms
                y = rng.choice([y for y in (2021, 2023, 2024) if date(y, 6, 20) >= a and date(y, 8, 5) <= b] or [ereig.year])
                ereig = _rand_date(rng, date(y, 6, 20), date(y, 8, 5)) if date(y, 6, 20) >= a else ereig
            meld = min(ereig + timedelta(days=int(rng.expovariate(1 / 6))), STICHTAG)
            age = (STICHTAG - meld).days
            if age > 240:
                stat = rng.choices(["ABG", "ABL", "PRF"], [88, 9, 3])[0]
            elif age > 60:
                stat = rng.choices(["ABG", "ABL", "OFF", "PRF"], [55, 7, 25, 13])[0]
            else:
                stat = rng.choices(["OFF", "PRF", "ABG"], [60, 25, 15])[0]
            amount = round(rng.lognormvariate(0, sig) * med * SEVERITY[sp] / 5) * 5
            if urs == "FEUER" and rng.random() < .1:
                amount *= rng.randint(3, 8)  # a few large losses
            fraud = 1 if rng.random() < (.08 if urs in ("DIEB", "KOLL") else .015) else 0
            sch_id += 1
            k_ereig = ptr_kt[v[2]] if rng.random() < .85 else rng.choices(kt_codes, kt_w)[0]
            reserve = float(amount) if stat in ("OFF", "PRF") else None
            sch.append((sch_id, f"S{ereig.year}-{sch_id:05d}", v[0], _d(ereig), _d(meld), urs, stat,
                        rng.choice(sb_all), reserve, k_ereig, fraud))
            hist.append((sch_id, "OFF", _d(meld)))
            d = meld
            if stat in ("PRF", "ABG", "ABL"):
                d = min(d + timedelta(days=rng.randint(3, 40)), STICHTAG)
                hist.append((sch_id, "PRF", _d(d)))
            if stat in ("ABG", "ABL"):
                d = min(d + timedelta(days=rng.randint(10, 150)), STICHTAG)
                hist.append((sch_id, stat, _d(d)))
            pays = []
            if stat == "ABG":
                parts = rng.choices([1, 2, 3], [60, 30, 10])[0]
                pays = [amount / parts] * parts
            elif stat in ("OFF", "PRF") and rng.random() < .3:
                pays = [amount * .4]
                reserve = amount * .6
                sch[-1] = sch[-1][:8] + (float(round(reserve, 2)),) + sch[-1][9:]
            empf = {"MF": ["WERKST", "KUNDE"], "PH": ["DRITT"], "RS": ["KUNDE", "DRITT"]}.get(sp, ["KUNDE"])
            for i, betrag in enumerate(pays):
                zlg_id += 1
                zd = min(meld + timedelta(days=rng.randint(10, 120) * (i + 1)), STICHTAG)
                zlg.append((zlg_id, sch_id, _d(zd), round(betrag, 2), rng.choice(empf)))
    con.executemany("INSERT INTO T_SCHADEN VALUES (?,?,?,?,?,?,?,?,?,?,?)", sch)
    con.executemany("INSERT INTO T_SCH_ZLG VALUES (?,?,?,?,?)", zlg)
    con.executemany("INSERT INTO T_SCH_STAT_HIST VALUES (?,?,?)", hist)

    # premium invoices of the last 12 months, active contracts only
    rg, rg_id = [], 0
    per_year = {"J": 1, "H": 2, "Q": 4, "M": 12}
    for v in vtr:
        if v[8] != "AKT":
            continue
        n = per_year[v[9]]
        beg = date.fromisoformat(v[5])
        for i in range(n):
            months = i * 12 // n
            m = (beg.month - 1 + months) % 12 + 1
            y = 2025 if m >= 7 else 2026
            rdat = date(y, m, min(beg.day, 28))
            if rdat < beg or rdat > STICHTAG:
                continue
            rg_id += 1
            faellig = rdat + timedelta(days=30)
            paid = faellig > STICHTAG and rng.random() < .55 or faellig <= STICHTAG and rng.random() < .955
            bez = _d(min(rdat + timedelta(days=rng.randint(5, 45)), STICHTAG)) if paid else None
            mahn = 0
            if not paid and faellig < STICHTAG:
                mahn = min(3, 1 + (STICHTAG - faellig).days // 45)
            rg.append((rg_id, v[0], _d(rdat), _d(faellig), round(v[10] / n, 2), bez, mahn))
    con.executemany("INSERT INTO T_RECHNUNG VALUES (?,?,?,?,?,?,?)", rg)
    con.commit()
    con.close()
    return path


def prompt_schema(catalog=False):
    """CREATE TABLE statements as the model sees them. With catalog=True, each table and column
    gets the data steward's description as an SQL comment, and the glossary is appended."""
    con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    out = []
    for name, cols in [(t, con.execute(f'PRAGMA table_info("{t}")').fetchall()) for (t,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid")]:
        fks = con.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
        desc, coldesc = CATALOG.get(name, ("", {}))
        lines = []
        for c in cols:
            line = f"{c[1]} {c[2]}" + (" PRIMARY KEY" if c[5] == 1 and sum(x[5] > 0 for x in cols) == 1 else "")
            if catalog and c[1] in coldesc:
                line += f",  -- {coldesc[c[1]]}"
            lines.append(line)
        pk = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5] > 0]
        if len(pk) > 1:
            lines.append(f"PRIMARY KEY ({', '.join(pk)})")
        for fk in fks:
            lines.append(f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
        # commas: a comment must come after the comma, so rebuild separators
        body = []
        for i, ln in enumerate(lines):
            last = i == len(lines) - 1
            if ",  -- " in ln:
                col, com = ln.split(",  -- ", 1)
                body.append(f"{col}{'' if last else ','}  -- {com}")
            else:
                body.append(ln + ("" if last else ","))
        head = f"-- {desc}\n" if catalog and desc else ""
        out.append(f"{head}CREATE TABLE {name} (\n" + "\n".join(body) + "\n);")
    con.close()
    schema = "\n\n".join(out)
    if catalog:
        schema += "\n\n-- Glossar (Geschäftsbegriffe):\n" + "\n".join(f"-- {k}: {v}" for k, v in GLOSSARY.items())
    return schema


def load_questions():
    """Test examples in the same shape as Spider examples (db_id, question, query) plus metadata.
    query is None for questions this database cannot answer."""
    qs = json.loads(QUESTIONS_FILE.read_text())
    return [{"db_id": DB_ID, **q} for q in qs]


def check():
    from sql_env import execute_sql
    qs = load_questions()
    bad, empty, from_collections = [], [], {}
    for q in qs:
        if q["query"] is None:
            continue
        rows, err = execute_sql(DB_ID, q["query"], timeout=30)
        if err:
            bad.append((q["id"], err))
        elif not rows or all(v is None for r in rows for v in r):
            empty.append(q["id"])
    from collections import Counter
    print(f"{len(qs)} questions | languages {dict(Counter(q['lang'] for q in qs))} | "
          f"levels {dict(Counter(q['level'] for q in qs))} | unanswerable {sum(q['query'] is None for q in qs)}")
    print("gold errors:", bad or "none")
    print("empty gold results:", empty or "none")
    return not bad and not empty


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        p = build()
        con = sqlite3.connect(p)
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            print(f"{t:18s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>6} rows")
        print("written:", p)
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "schema":
        print(prompt_schema(catalog="--plain" not in sys.argv))
