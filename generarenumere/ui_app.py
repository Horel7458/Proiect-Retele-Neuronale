import os
import re
import json
import time
import threading
import csv
import tempfile
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, cast

import sys
import pandas as pd

# UI
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Camera + OCR
import cv2
import easyocr

# NN (optional)
# Note: we keep runtime behavior optional, but type it as Any to avoid noisy Pylance warnings.
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
else:
    torch = cast(Any, torch)
    nn = cast(Any, nn)

# ============================================================
# QUICK NOTES (for evaluation, short comments)
# ============================================================
# This file implements the desktop UI (Tkinter) for the project.
# Goal: user selects a traffic context + plate, then gets a risk score.
#
# Main flows you can demo end-to-end:
# - Login/Register (full name + plate stored in CSV)
# - Select intersection + interval
# - Enter plate manually OR read plate via OCR (camera/image)
# - Run inference (PyTorch model if available) and show LOW/MEDIUM/HIGH
# - Optional: update counters (simulate new accidents) and re-run
#
# Data files used (relative to project root):
# - data/raw/plates_export.csv          (known plates + vehicle accidents)
# - data/raw/intersections.csv          (intersection scenarios + accidents)
# - data/processed/stats_by_judet.csv   (county score)
# - data/processed/model.pth            (trained PyTorch weights)
# - data/processed/nn_scaler.json       (min-max scaler for 3 features)
# - data/processed/drivers.csv          (login/register storage)
#
# UI is designed to be robust on Windows:
# - no absolute paths
# - can run from any working directory
# - PyInstaller-friendly (resource_path + writable_path)
#
# Important design choices (kept simple on purpose):
# - Model is regression: outputs score in [0,1]
# - We map score to 3 categories for readability
# - OCR is best-effort: we allow fuzzy match against known plates
#
# Risk category thresholds used across UI:
# - LOW    : score < 0.40
# - MEDIUM : 0.40 <= score < 0.70
# - HIGH   : score >= 0.70
#
# OCR notes:
# - EasyOCR returns multiple text candidates
# - We clean text to [A-Z0-9] and try direct match first
# - If not found, we try substitution candidates (I<->1, O<->0, etc)
# - As last step we use a small Levenshtein-based fuzzy match
#
# Threading notes:
# - Camera capture / OCR runs in a background thread
# - UI updates must be scheduled back into Tk main thread
# - This avoids UI freeze when OCR takes time
#
# Error handling notes:
# - If camera/OCR/model files are missing, we show a friendly message
# - If model is not available, app can use a fallback formula
#
# Debug tips:
# - If OCR is poor, try better lighting / higher resolution
# - If a plate is not found, check data/raw/plates_export.csv
# - If inference fails, check data/processed/model.pth and nn_scaler.json
#
# Code map (high level):
# - Path helpers: find_project_root, resource_path, writable_path
# - Text helpers: clean_plate_text, county_from_plate, levenshtein
# - CSV loaders: load_plates/intersections/county_stats, load_drivers
# - Model helpers: load_scaler, load_model, predict_score
# - UI class: widgets, handlers, and display logic
#
# Comment style rules for this repo (per requirement):
# - short lines
# - no diacritics
# - focus on what/why, not repeating obvious python syntax

# ------------------------------------------------------------
# DETAILED WALKTHROUGH (short lines, but many)
# ------------------------------------------------------------
# Below is a compact walkthrough of what the UI does.
# It is intentionally written as many short lines to be easy to scan.
#
# Start-up:
# - detect project root
# - build paths to CSV/model/scaler
# - load CSV tables into pandas
# - pre-load OCR reader (may be slow the first time)
# - load model + scaler if torch is available
#
# Login/Register:
# - user enters full name
# - user enters plate (manual or via OCR)
# - we normalize name (trim spaces)
# - we normalize plate (A-Z0-9 only)
# - we store identity in drivers.csv
# - last_login is updated on success
#
# Context selection:
# - user chooses intersection from intersections.csv
# - user chooses interval_label (ex: Dimineata/Pranz/Seara)
# - time_range is shown for clarity
#
# Plate input:
# - manual entry is allowed
# - OCR can fill the entry automatically
# - we still validate before running inference
#
# OCR (camera/image):
# - open camera capture
# - get one frame (or a few attempts)
# - run easyocr.readtext
# - collect candidates + confidence
# - clean candidate text
# - try direct match against known plates
# - if not found, try substitution candidates
# - if still not found, use Levenshtein match
# - show best match + confidence to user
#
# Feature building (3 inputs for MLP):
# - accidente_intersectie (from intersections.csv)
# - accidente_vehicul     (from plates_export.csv for the plate)
# - scor_judet            (from stats_by_judet.csv using county code)
#
# County code extraction:
# - first 1-2 letters from plate
# - ex: "B" or "AG"
# - used as key for county stats
#
# Normalization:
# - we apply min-max using nn_scaler.json
# - scaler is fit on train only (done in training script)
# - UI must use the same scaling to match training
#
# Inference:
# - if torch/model is available, run model(x)
# - output is a float in [0,1] (sigmoid)
# - if model is missing, fallback is a simple heuristic
#
# Decision:
# - map score to category
# - LOW / MEDIUM / HIGH
# - show both numeric and category
#
# Update flow (simulated live data):
# - user can add +1 accident to a vehicle or intersection
# - we write updated values back to CSV
# - app reloads data (or refreshes cached tables)
# - user can re-run inference and see score change
#
# UI stability rules:
# - do not block Tk main loop
# - run OCR in thread
# - handle missing files with messagebox
# - keep paths relative
#
# Common issues and quick fixes:
# - missing model.pth: run src/neural_network/train_nn.py
# - missing nn_scaler.json: run src/neural_network/train_nn.py
# - missing nn_dataset.csv: run src/preprocessing/dataset_builder.py
# - camera not found: check permissions / camera index
# - OCR slow: first call downloads/loads models
# - plate not in CSV: check data/raw/plates_export.csv
#
# Small glossary:
# - plate: license plate text (A-Z0-9)
# - intersection: location name
# - interval_label: time bucket
# - score: regression output in [0,1]
# - category: LOW/MEDIUM/HIGH
#
# Notes about code structure:
# - helpers are grouped: text, csv, model, ui
# - dataclasses keep small data bundles readable
# - we avoid over-engineering to keep demo simple
#
# End-to-end demo recording checklist:
# - open app
# - login/register
# - select context
# - OCR read OR manual plate
# - show score + category
# - do one update (+1 accident)
# - show score changes
#
# Extra notes (short, but explicit):
# - this is a school demo, not a production system
# - csv auth is not secure, it is for assignment scope
# - label is heuristic, not ground-truth
# - metrics are high because target is deterministic


# ============================================================
# PATHS (works when running from anywhere in the project)
# ============================================================

def find_project_root(start_dir: str) -> str:
    """
    Find project root by searching upward for a folder named 'data'.
    This makes relative paths work no matter where you run ui_app.py from.
    """
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # reached filesystem root
            return os.path.abspath(start_dir)
        cur = parent


def resource_path(relative_path: str) -> str:
    """
    Works for normal python run and for PyInstaller.
    - PyInstaller: base is sys._MEIPASS
    - Normal: base is detected project root (folder that contains /data)
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(str(base), relative_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)
    return os.path.join(project_root, relative_path)


PLATES_CSV = resource_path(r"data/raw/plates_export.csv")
INTERSECTIONS_CSV = resource_path(r"data/raw/intersections.csv")
COUNTY_STATS_CSV = resource_path(r"data/processed/stats_by_judet.csv")

MODEL_PATH = resource_path(r"data/processed/model.pth")
SCALER_PATH = resource_path(r"data/processed/nn_scaler.json")


def _is_frozen() -> bool:
    return bool(getattr(sys, "_MEIPASS", None))


def writable_path(relative_path: str) -> str:
    """Return a path we can write to.

    - Dev/run-from-source: write inside the project (same root as data/)
    - PyInstaller: write into LOCALAPPDATA\\RiskApp\\...
    """
    if not _is_frozen():
        return resource_path(relative_path)

    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "RiskApp", relative_path.replace("/", os.sep).replace("\\", os.sep))


DRIVERS_CSV = writable_path(r"data/processed/drivers.csv")


# ============================================================
# Helpers: normalization + fuzzy match
# ============================================================

PLATE_RE = re.compile(r"[A-Z0-9]+")

SUBS = [
    ("I", "1"), ("1", "I"),
    ("O", "0"), ("0", "O"),
    ("B", "8"), ("8", "B"),
    ("S", "5"), ("5", "S"),
    ("Z", "2"), ("2", "Z"),
    ("G", "6"), ("6", "G"),
    ("A", "4"), ("4", "A"),
]


SUB_MAP: Dict[str, List[str]] = {}
for a, b in SUBS:
    SUB_MAP.setdefault(a, []).append(b)


def clean_plate_text(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    parts = PLATE_RE.findall(s)
    return "".join(parts)


def normalize_full_name(name: str) -> str:
    # Keep user's capitalization, but normalize whitespace.
    return " ".join((name or "").strip().split())


def full_name_key(name: str) -> str:
    # Case-insensitive key for matching.
    return normalize_full_name(name).casefold()


DRIVER_FIELDS = [
    "full_name",
    "full_name_key",
    "plate",
    "created_at",
    "last_login",
]


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_drivers_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if not r:
                continue
            # tolerate missing columns in older files
            rows.append({k: str(r.get(k, "") or "") for k in DRIVER_FIELDS})
        return rows


def save_drivers_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = os.path.dirname(path) or os.getcwd()

    fd, tmp_path = tempfile.mkstemp(prefix="drivers_", suffix=".csv", dir=tmp_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DRIVER_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: str(r.get(k, "") or "") for k in DRIVER_FIELDS})
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def upsert_driver(path: str, name: str, plate: str, *, set_created: bool, set_last_login: bool) -> None:
    name_norm = normalize_full_name(name)
    key = full_name_key(name_norm)
    plate_clean = clean_plate_text(plate)

    if not name_norm or not plate_clean:
        raise ValueError("Invalid name or plate")

    rows = load_drivers_csv(path)
    now = _utc_now_iso()

    idx = None
    for i, r in enumerate(rows):
        if r.get("full_name_key", "") == key and r.get("plate", "") == plate_clean:
            idx = i
            break

    if idx is None:
        row = {
            "full_name": name_norm,
            "full_name_key": key,
            "plate": plate_clean,
            "created_at": now if set_created else "",
            "last_login": now if set_last_login else "",
        }
        rows.append(row)
    else:
        rows[idx]["full_name"] = name_norm  # keep latest formatting
        rows[idx]["full_name_key"] = key
        rows[idx]["plate"] = plate_clean
        if set_created and not rows[idx].get("created_at"):
            rows[idx]["created_at"] = now
        if set_last_login:
            rows[idx]["last_login"] = now

    save_drivers_csv(path, rows)


def driver_exists(path: str, name: str, plate: str) -> bool:
    key = full_name_key(name)
    plate_clean = clean_plate_text(plate)
    if not key or not plate_clean:
        return False
    for r in load_drivers_csv(path):
        if r.get("full_name_key", "") == key and r.get("plate", "") == plate_clean:
            return True
    return False


def county_from_plate(plate: str) -> str:
    p = clean_plate_text(plate)
    if len(p) >= 2 and p[0].isalpha() and p[1].isalpha():
        return p[:2]
    if len(p) >= 1 and p[0].isalpha():
        return p[:1]
    return ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _one_step_substitutions(p: str) -> List[str]:
    out = []
    for i, ch in enumerate(p):
        reps = SUB_MAP.get(ch)
        if not reps:
            continue
        for rep in reps:
            out.append(p[:i] + rep + p[i + 1:])
    return out


def generate_substitution_candidates(plate: str, max_candidates: int = 2000, max_steps: int = 2) -> List[str]:
    """Generate candidates by applying common OCR substitutions up to max_steps times."""
    p = clean_plate_text(plate)
    if not p:
        return []

    seen = {p}
    frontier = {p}
    for _ in range(max_steps):
        nxt = set()
        for s in frontier:
            for ns in _one_step_substitutions(s):
                if ns not in seen:
                    seen.add(ns)
                    nxt.add(ns)
                    if len(seen) >= max_candidates:
                        frontier = set()
                        break
            if not frontier and len(seen) >= max_candidates:
                break
        frontier = nxt
        if not frontier:
            break

    out = sorted(seen)
    return out[:max_candidates]


def match_plate_to_csv(ocr_plate: str, known_plates: List[str]) -> Tuple[Optional[str], float]:
    p = clean_plate_text(ocr_plate)
    if not p:
        return None, 0.0

    known_set = set(known_plates)

    if p in known_set:
        return p, 1.0

    # Try common OCR substitutions (allow up to 2 substitutions)
    for c in generate_substitution_candidates(p, max_candidates=2000, max_steps=2):
        if c in known_set:
            # heuristic confidence for substitution-based match
            return c, 0.92

    same_len = [k for k in known_plates if abs(len(k) - len(p)) <= 1]
    if same_len and p and p[0].isalpha():
        filtered = [k for k in same_len if k and k[0] == p[0]]
        if filtered:
            same_len = filtered
    best = None
    best_d = 10**9
    for k in same_len:
        d = levenshtein(p, k)
        if d < best_d:
            best_d = d
            best = k

    if best is not None and best_d <= 3:
        score = max(0.0, 0.88 - 0.18 * best_d)
        return best, score

    return None, 0.0


# ============================================================
# NN model + scaler loader
# ============================================================

if nn is not None:
    nn_ = cast(Any, nn)

    class RiskMLP(nn_.Module):
        def __init__(self, in_dim: int = 3):
            super().__init__()
            self.net = nn_.Sequential(
                nn_.Linear(in_dim, 16),
                nn_.ReLU(),
                nn_.Linear(16, 8),
                nn_.ReLU(),
                nn_.Linear(8, 1),
                nn_.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)
else:
    RiskMLP = None  # type: ignore[assignment]


def load_model_if_available(log_fn):
    if torch is None or nn is None or RiskMLP is None:
        log_fn("[WARN] Torch not available -> using fallback risk formula.")
        return None, None

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        log_fn("[INFO] No model/scaler found -> using fallback risk formula.")
        return None, None

    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    feature_cols = scaler.get("feature_cols", ["acc_intersection", "acc_vehicle", "county_score"])
    in_dim = len(feature_cols)
    model = RiskMLP(in_dim=in_dim)

    state = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=True)
        log_fn("[OK] Loaded model checkpoint with model_state.")
        ck_cols = state.get("feature_cols")
        if ck_cols and isinstance(ck_cols, list):
            scaler["feature_cols"] = ck_cols
            log_fn(f"[OK] Using feature_cols from checkpoint: {ck_cols}")
    else:
        model.load_state_dict(state, strict=True)
        log_fn("[OK] Loaded plain state_dict model.")

    model.eval()
    return model, scaler


def normalize_features(x: List[float], scaler: dict) -> List[float]:
    mins = scaler.get("min", {})
    maxs = scaler.get("max", {})
    cols = scaler.get("feature_cols", [])

    out = []
    for i, val in enumerate(x):
        col = cols[i] if i < len(cols) else f"f{i}"
        mn = float(mins.get(col, 0.0))
        mx = float(maxs.get(col, 1.0))
        if mx - mn == 0:
            out.append(0.0)
        else:
            out.append((float(val) - mn) / (mx - mn))
    return out


def predict_risk(model, scaler, features: List[float]) -> float:
    # Fallback if model missing
    if model is None or scaler is None or torch is None:
        acc_i, acc_v, county_s = features
        # simple weighted + sigmoid squash
        raw = 0.55 * acc_i + 0.30 * acc_v + 0.15 * county_s
        return float(1.0 / (1.0 + pow(2.718281828, -raw / 10.0)))

    x_norm = normalize_features(features, scaler)
    xt = torch.tensor([x_norm], dtype=torch.float32)
    with torch.no_grad():
        y = model(xt).item()
    return float(y)


def risk_category(r: float) -> str:
    if r < 0.40:
        return "LOW"
    if r < 0.70:
        return "MEDIUM"
    return "HIGH"


# ============================================================
# Data structures
# ============================================================

@dataclass
class IntersectionKey:
    intersection: str
    interval_label: str


# ============================================================
# Main App
# ============================================================

class RiskApp(tk.Tk):
    def __init__(self, defer_init: bool = False):
        super().__init__()
        self.title("Risk App (Intersection + Plate + County)")
        self.geometry("1000x640")

        # Logged-in driver (set by LoginDialog before showing the app)
        self.logged_in = False
        self.driver_full_name = ""
        self.driver_plate = ""

        # UI theme tokens (kept simple; ttk colors depend on theme support)
        self.ui_colors = {
            "bg": "#f5f7fb",
            "panel": "#ffffff",
            "accent": "#2563eb",
            "accent_dark": "#1d4ed8",
            "muted": "#64748b",
            "ok": "#16a34a",
            "warn": "#d97706",
            "err": "#dc2626",
        }

        self.plates_df = None
        self.intersections_df = None
        self.county_df = None

        self.plate_to_acc: Dict[str, int] = {}
        self.known_plates: List[str] = []
        self.known_plate_set = set()

        # fallback stats used when a plate isn't present in plates_export.csv
        self.county_vehicle_acc_mean: Dict[str, float] = {}

        self.int_acc_map: Dict[Tuple[str, str], int] = {}
        self.county_score_map: Dict[str, float] = {}

        self.model = None
        self.scaler = None

        self.reader = None  # EasyOCR

        # live scan state
        self.live_running = False
        self.last_live_plate = ""
        self.last_live_time = 0.0

        self._initialized_main_ui = False
        # Ensure ttk styles exist for the login dialog too.
        try:
            self._apply_theme()
        except Exception:
            pass

        if not defer_init:
            self.initialize_main_ui()

        if defer_init:
            self.show_login_ui()

    def show_login_ui(self):
        # Clear window
        for child in list(self.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        self.title("Risk App - Login")
        self.configure(bg=self.ui_colors["bg"])

        frame = LoginFrame(self, DRIVERS_CSV)
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass

    def initialize_main_ui(self):
        if self._initialized_main_ui:
            return
        self._initialized_main_ui = True
        self._build_ui()
        self._load_all()
        # If user is already logged in, prefill plate once widgets exist.
        if self.driver_plate:
            try:
                self._set_plate_in_ui(self.driver_plate, source="login")
            except Exception:
                pass

    def set_logged_driver(self, full_name: str, plate: str):
        self.logged_in = True
        self.driver_full_name = normalize_full_name(full_name)
        self.driver_plate = clean_plate_text(plate)
        if self.driver_full_name and self.driver_plate:
            self.title(f"Risk App - {self.driver_full_name} ({self.driver_plate})")
        elif self.driver_full_name:
            self.title(f"Risk App - {self.driver_full_name}")
        elif self.driver_plate:
            self.title(f"Risk App - {self.driver_plate}")

        # Prefill only if the main UI widgets already exist.
        if hasattr(self, "ent_plate"):
            try:
                self._set_plate_in_ui(self.driver_plate, source="login")
            except Exception:
                pass

    # ---------- Theme ----------
    def _apply_theme(self):
        style = ttk.Style(self)

        # Prefer a theme that allows color customization.
        available = set(style.theme_names())
        for candidate in ("clam", "vista", "xpnative"):
            if candidate in available:
                try:
                    style.theme_use(candidate)
                except Exception:
                    pass
                break

        self.configure(bg=self.ui_colors["bg"])

        base_font = ("Segoe UI", 10)
        style.configure("TFrame", background=self.ui_colors["bg"])
        style.configure("Panel.TFrame", background=self.ui_colors["panel"])
        style.configure("TLabel", background=self.ui_colors["panel"], font=base_font)
        style.configure("Muted.TLabel", foreground=self.ui_colors["muted"], background=self.ui_colors["panel"], font=base_font)
        style.configure("Header.TLabel", background=self.ui_colors["bg"], foreground="#0f172a", font=("Segoe UI", 16, "bold"))
        style.configure("Subheader.TLabel", background=self.ui_colors["bg"], foreground=self.ui_colors["muted"], font=("Segoe UI", 10))

        style.configure("TLabelframe", background=self.ui_colors["panel"])
        style.configure("TLabelframe.Label", background=self.ui_colors["panel"], foreground="#0f172a", font=("Segoe UI", 10, "bold"))

        style.configure(
            "Accent.TButton",
            background=self.ui_colors["accent"],
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ui_colors["accent_dark"]), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#e2e8f0")],
        )

        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("TEntry", padding=(6, 4), font=("Segoe UI", 10))
        style.configure("TCombobox", padding=(6, 4), font=("Segoe UI", 10))

    # ---------- UI ----------
    def _build_ui(self):
        self._apply_theme()

        header = ttk.Frame(self, style="TFrame")
        header.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 6))
        ttk.Label(header, text="Risk Estimator", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Intersecție + interval + plăcuță (OCR) + județ → scor de risc (0..1)",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))

        # Panels
        left_panel = ttk.Frame(left, style="Panel.TFrame")
        left_panel.pack(fill=tk.BOTH, expand=True)
        right_panel = ttk.Frame(right, style="Panel.TFrame")
        right_panel.pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_panel, text="Intersection").grid(row=0, column=0, sticky="w", pady=6, padx=(10, 6))
        self.cmb_intersection = ttk.Combobox(left_panel, values=[], state="readonly", width=35)
        self.cmb_intersection.grid(row=0, column=1, sticky="w", pady=6, padx=(0, 10))

        ttk.Label(left_panel, text="Interval").grid(row=1, column=0, sticky="w", pady=6, padx=(10, 6))
        self.cmb_interval = ttk.Combobox(left_panel, values=[], state="readonly", width=35)
        self.cmb_interval.grid(row=1, column=1, sticky="w", pady=6, padx=(0, 10))

        ttk.Label(left_panel, text="Plate (manual / from scan)").grid(row=2, column=0, sticky="w", pady=6, padx=(10, 6))
        self.ent_plate = ttk.Entry(left_panel, width=38)
        self.ent_plate.grid(row=2, column=1, sticky="w", pady=6, padx=(0, 10))

        btns = ttk.Frame(left_panel, style="Panel.TFrame")
        btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=10)

        self.btn_scan_s = ttk.Button(btns, text="Scan camera (press 's')", command=self.on_scan_camera_press_s)
        self.btn_scan_s.pack(side=tk.LEFT, padx=4)

        self.btn_scan_live = ttk.Button(btns, text="Start live scan (auto)", command=self.on_toggle_live_scan)
        self.btn_scan_live.pack(side=tk.LEFT, padx=4)

        self.btn_upload = ttk.Button(btns, text="Upload image (OCR)", command=self.on_upload_image)
        self.btn_upload.pack(side=tk.LEFT, padx=4)

        self.btn_calc = ttk.Button(btns, text="Calculate risk", style="Accent.TButton", command=self.on_calculate_risk)
        self.btn_calc.pack(side=tk.LEFT, padx=4)

        self.btn_add = ttk.Button(btns, text="Add accident (intersection+interval)", command=self.on_add_accident)
        self.btn_add.pack(side=tk.LEFT, padx=4)

        box = ttk.LabelFrame(left_panel, text="Current data")
        box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 10), padx=10)
        left_panel.grid_rowconfigure(4, weight=1)
        left_panel.grid_columnconfigure(1, weight=1)

        self.lbl_int_acc = ttk.Label(box, text="Intersection accidents: -")
        self.lbl_int_acc.pack(anchor="w", pady=2)

        self.lbl_veh_acc = ttk.Label(box, text="Vehicle accidents: -")
        self.lbl_veh_acc.pack(anchor="w", pady=2)

        self.lbl_county = ttk.Label(box, text="County code: -")
        self.lbl_county.pack(anchor="w", pady=2)

        self.lbl_county_score = ttk.Label(box, text="County score: -")
        self.lbl_county_score.pack(anchor="w", pady=2)

        res = ttk.LabelFrame(right_panel, text="Risk result")
        res.pack(fill=tk.X, pady=(0, 10), padx=10)

        ttk.Label(res, text="Risk (0..1):").pack(anchor="w", pady=2)
        self.lbl_risk_val = ttk.Label(res, text="-", font=("Segoe UI", 22, "bold"))
        self.lbl_risk_val.pack(anchor="w", pady=2)

        ttk.Label(res, text="Category:").pack(anchor="w", pady=2)
        self.lbl_risk_cat = ttk.Label(res, text="-", font=("Segoe UI", 16, "bold"))
        self.lbl_risk_cat.pack(anchor="w", pady=2)

        logbox = ttk.LabelFrame(right_panel, text="Log")
        logbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)

        self.txt_log = tk.Text(logbox, height=18, width=55)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # make log easier to read
        self.txt_log.configure(
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            selectbackground="#334155",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            font=("Consolas", 10),
        )
        self.txt_log.tag_configure("ok", foreground=self.ui_colors["ok"])
        self.txt_log.tag_configure("warn", foreground=self.ui_colors["warn"])
        self.txt_log.tag_configure("err", foreground=self.ui_colors["err"])
        self.txt_log.tag_configure("risk", foreground="#60a5fa")
        self.txt_log.tag_configure("info", foreground="#e5e7eb")

        self.cmb_intersection.bind("<<ComboboxSelected>>", lambda e: self.refresh_current_data())
        self.cmb_interval.bind("<<ComboboxSelected>>", lambda e: self.refresh_current_data())
        self.ent_plate.bind("<KeyRelease>", lambda e: self.refresh_current_data())

    def log(self, msg: str):
        tag = "info"
        m = msg.strip()
        if m.startswith("[OK]"):
            tag = "ok"
        elif m.startswith("[WARN]"):
            tag = "warn"
        elif m.startswith("[ERROR]"):
            tag = "err"
        elif m.startswith("[RISK]"):
            tag = "risk"

        self.txt_log.insert(tk.END, msg + "\n", tag)
        self.txt_log.see(tk.END)

    def _set_risk_colors(self, cat: str):
        cat = (cat or "").upper().strip()
        if cat == "LOW":
            fg = self.ui_colors["ok"]
        elif cat == "MEDIUM":
            fg = self.ui_colors["warn"]
        elif cat == "HIGH":
            fg = self.ui_colors["err"]
        else:
            fg = "#0f172a"

        try:
            self.lbl_risk_cat.configure(foreground=fg)
            self.lbl_risk_val.configure(foreground=fg)
        except Exception:
            # Some ttk themes may ignore foreground changes; ignore gracefully.
            pass

    def log_ui(self, msg: str):
        self.after(0, lambda: self.log(msg))

    # ---------- Load data ----------
    def _load_all(self):
        try:
            self.plates_df = pd.read_csv(PLATES_CSV)
            self.plates_df.columns = [c.strip().lower() for c in self.plates_df.columns]
            self.plates_df["plate_clean"] = self.plates_df["plate"].astype(str).map(clean_plate_text)
            self.plate_to_acc = dict(zip(self.plates_df["plate_clean"], self.plates_df["accidents"].astype(int)))
            self.known_plates = list(self.plate_to_acc.keys())
            self.known_plate_set = set(self.known_plates)

            # Precompute county mean accidents per vehicle (used as a fallback for unknown plates)
            try:
                tmp = self.plates_df[["plate_clean", "accidents"]].copy()
                tmp["county_code"] = tmp["plate_clean"].astype(str).map(county_from_plate)
                tmp = tmp[tmp["county_code"].astype(str).str.len() > 0]
                tmp["county_code"] = tmp["county_code"].astype(str).str.strip().str.upper()
                tmp["accidents"] = pd.to_numeric(tmp["accidents"], errors="coerce")
                tmp = tmp.dropna(subset=["accidents"]).reset_index(drop=True)
                if not tmp.empty:
                    self.county_vehicle_acc_mean = (
                        tmp.groupby("county_code")["accidents"].mean().astype(float).to_dict()
                    )
            except Exception:
                # keep fallback empty; UI will default to 0
                self.county_vehicle_acc_mean = {}

            self.log(f"[OK] Loaded plates: {len(self.plates_df)} rows from {PLATES_CSV}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load plates CSV:\n{PLATES_CSV}\n\n{e}")
            return

        try:
            self.intersections_df = pd.read_csv(INTERSECTIONS_CSV)
            self.intersections_df.columns = [c.strip().lower() for c in self.intersections_df.columns]
            for _, r in self.intersections_df.iterrows():
                key = (str(r["intersection"]), str(r["interval_label"]))
                self.int_acc_map[key] = int(r["accidents"])
            inters = sorted(self.intersections_df["intersection"].astype(str).unique().tolist())
            intervals = sorted(self.intersections_df["interval_label"].astype(str).unique().tolist())
            self.cmb_intersection["values"] = inters
            self.cmb_interval["values"] = intervals
            if inters:
                self.cmb_intersection.set(inters[0])
            if intervals:
                self.cmb_interval.set(intervals[0])
            self.log(f"[OK] Loaded intersections from {INTERSECTIONS_CSV}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load intersections CSV:\n{INTERSECTIONS_CSV}\n\n{e}")
            return

        try:
            self.county_df = pd.read_csv(COUNTY_STATS_CSV)
            self.county_df.columns = [c.strip().lower() for c in self.county_df.columns]
            self.county_df["county_code"] = self.county_df["county_code"].astype(str).str.strip().str.upper()
            score_col = "scor_mediu_accidente"
            if score_col not in self.county_df.columns:
                for c in self.county_df.columns:
                    if "scor" in c and "acc" in c:
                        score_col = c
                        break
            self.county_score_map = dict(zip(self.county_df["county_code"], self.county_df[score_col].astype(float)))
            self.log(f"[OK] Loaded county stats from {COUNTY_STATS_CSV}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load county stats CSV:\n{COUNTY_STATS_CSV}\n\n{e}")
            return

        self.model, self.scaler = load_model_if_available(self.log)
        self.refresh_current_data()

    def resolve_plate(self, plate_input: str) -> Tuple[str, Optional[str], float]:
        """Return (clean_input, resolved_plate_or_none, match_score)."""
        clean_in = clean_plate_text(plate_input)
        if not clean_in:
            return "", None, 0.0
        if clean_in in self.known_plate_set:
            return clean_in, clean_in, 1.0
        best, score = match_plate_to_csv(clean_in, self.known_plates)
        if best is None:
            return clean_in, None, 0.0
        return clean_in, best, float(score)

    # ---------- Getters ----------
    def get_selected_intersection_interval(self) -> Tuple[str, str]:
        return self.cmb_intersection.get(), self.cmb_interval.get()

    def get_plate_clean(self) -> str:
        return clean_plate_text(self.ent_plate.get())

    def get_vehicle_accidents(self, plate_clean: str) -> int:
        return int(self.plate_to_acc.get(plate_clean, 0))

    def get_vehicle_accidents_fallback(self, plate_clean: str) -> Tuple[int, str]:
        """Return (accidents, source) where source describes how value was obtained."""
        plate_clean = clean_plate_text(plate_clean)
        if not plate_clean:
            return 0, "empty"

        if plate_clean in self.plate_to_acc:
            return int(self.plate_to_acc.get(plate_clean, 0)), "csv"

        county = county_from_plate(plate_clean)
        if county:
            m = self.county_vehicle_acc_mean.get(county)
            if m is not None:
                return int(round(float(m))), f"county_mean:{county}"

        return 0, "unknown"

    def get_intersection_accidents(self, intersection: str, interval_label: str) -> int:
        return int(self.int_acc_map.get((intersection, interval_label), 0))

    def get_county_score(self, county: str) -> float:
        return float(self.county_score_map.get(county, 0.0))

    # ---------- UI updates ----------
    def refresh_current_data(self):
        inter, interval = self.get_selected_intersection_interval()
        plate_clean = self.get_plate_clean()

        _, resolved_plate, plate_score = self.resolve_plate(plate_clean)
        plate_for_lookup = resolved_plate or plate_clean

        acc_i = self.get_intersection_accidents(inter, interval)
        acc_v, acc_v_src = self.get_vehicle_accidents_fallback(plate_for_lookup)
        county = county_from_plate(plate_for_lookup)
        county_score = self.get_county_score(county)

        self.lbl_int_acc.config(text=f"Intersection accidents: {acc_i}")
        if resolved_plate:
            # Always show match score, even for exact matches (score=1.00),
            # so plates with 0 accidents don't look like "not found".
            if resolved_plate != plate_clean:
                self.lbl_veh_acc.config(
                    text=f"Vehicle accidents: {acc_v} (matched {resolved_plate}, {plate_score:.2f})"
                )
            else:
                self.lbl_veh_acc.config(text=f"Vehicle accidents: {acc_v} (match {plate_score:.2f})")
        else:
            if acc_v_src.startswith("county_mean:"):
                self.lbl_veh_acc.config(text=f"Vehicle accidents: {acc_v} (estimated {acc_v_src})")
            else:
                self.lbl_veh_acc.config(text=f"Vehicle accidents: {acc_v}")
        self.lbl_county.config(text=f"County code: {county if county else '-'}")
        self.lbl_county_score.config(text=f"County score: {county_score:.3f}")

    # ---------- Actions ----------
    def on_calculate_risk(self):
        inter, interval = self.get_selected_intersection_interval()
        plate_clean = self.get_plate_clean()

        if not inter or not interval:
            messagebox.showwarning("Warning", "Select intersection and interval first.")
            return
        if not plate_clean:
            messagebox.showwarning("Warning", "Type or scan a plate first.")
            return

        _, resolved_plate, plate_score = self.resolve_plate(plate_clean)
        if resolved_plate is not None:
            if resolved_plate != plate_clean:
                self.log(f"[MATCH] input={plate_clean} -> resolved={resolved_plate} (score={plate_score:.2f})")
                # make the resolved plate visible to the user
                self._set_plate_in_ui(resolved_plate, "auto_match")
            plate_clean = resolved_plate
        else:
            # Don't block prediction. Use a sensible fallback for acc_vehicle.
            self.log(f"[INFO] Plate not present in CSV: {plate_clean} -> using fallback features.")

        acc_i = self.get_intersection_accidents(inter, interval)
        acc_v, acc_v_src = self.get_vehicle_accidents_fallback(plate_clean)
        county = county_from_plate(plate_clean)
        county_score = self.get_county_score(county)

        self.log(
            f"[INFO] Using plate={plate_clean} acc_vehicle={acc_v} ({acc_v_src}) county={county} county_score={county_score:.3f}"
        )
        self.log(f"[INFO] Using inter={inter} interval={interval} acc_intersection={acc_i}")

        r = predict_risk(self.model, self.scaler, [acc_i, acc_v, county_score])
        cat = risk_category(r)

        self.lbl_risk_val.config(text=f"{r:.3f}")
        self.lbl_risk_cat.config(text=cat)
        self._set_risk_colors(cat)
        self.refresh_current_data()

        self.log(f"[RISK] risk={r:.3f} category={cat}")

    def on_add_accident(self):
        inter, interval = self.get_selected_intersection_interval()
        if not inter or not interval:
            messagebox.showwarning("Warning", "Select intersection and interval first.")
            return
        if self.intersections_df is None:
            messagebox.showerror("Error", "Intersections data not loaded.")
            return

        key = (inter, interval)
        cur = int(self.int_acc_map.get(key, 0))
        new_val = cur + 1
        self.int_acc_map[key] = new_val

        try:
            mask = (self.intersections_df["intersection"].astype(str) == inter) & \
                   (self.intersections_df["interval_label"].astype(str) == interval)
            if mask.any():
                self.intersections_df.loc[mask, "accidents"] = new_val
            else:
                self.intersections_df = pd.concat([self.intersections_df, pd.DataFrame([{
                    "intersection": inter,
                    "interval_label": interval,
                    "time_range": "",
                    "accidents": new_val
                }])], ignore_index=True)

            self.intersections_df.to_csv(INTERSECTIONS_CSV, index=False)
            self.log(f"[OK] Added 1 accident to {inter} / {interval}. Now accidents={new_val}. Saved to CSV.")
            self.refresh_current_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update intersections CSV:\n{e}")

    # ---------- OCR init ----------
    def _ensure_ocr_reader(self):
        if self.reader is None:
            self.log_ui("[INFO] Initializing EasyOCR (first run may take time)...")
            self.reader = easyocr.Reader(["en"], gpu=False)
            self.log_ui("[OK] EasyOCR ready.")

    # ---------- Set plate in UI ----------
    def _set_plate_in_ui(self, plate_clean: str, source: str = ""):
        plate_clean = clean_plate_text(plate_clean)
        self.ent_plate.delete(0, tk.END)
        self.ent_plate.insert(0, plate_clean)
        self.refresh_current_data()

        acc_v = self.get_vehicle_accidents(plate_clean)
        self.log(f"[UI] Plate set ({source}): {plate_clean} | vehicle_accidents={acc_v}")

    # ==========================
    # 1) camera scan with 's'
    # ==========================
    def on_scan_camera_press_s(self):
        t = threading.Thread(target=self._scan_camera_press_s_worker, daemon=True)
        t.start()

    def _scan_camera_press_s_worker(self):
        try:
            self._ensure_ocr_reader()
        except Exception:
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_ui("[ERROR] Cannot open camera.")
            self.after(0, lambda: messagebox.showerror("Error", "Cannot open camera."))
            return

        self.log_ui("[INFO] Camera opened. Press 's' to OCR, 'q' to quit.")
        win = "Camera - press 's' to OCR, 'q' to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        last_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                self.log_ui("[ERROR] Cannot read frame.")
                break
            last_frame = frame

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):
                plate, plate_conf, _ = self._ocr_plate_from_frame(last_frame)
                if not plate:
                    self.log_ui("[WARN] No plate found on 's'.")
                    continue

                best, match_score = match_plate_to_csv(plate, self.known_plates)
                if best is None:
                    self.log_ui(f"[WARN] OCR={plate} conf={plate_conf:.2f} not matched in CSV.")
                    self.after(0, lambda p=plate: self._set_plate_in_ui(p, "camera_s_unmatched"))
                else:
                    self.log_ui(f"[OK] OCR={plate} conf={plate_conf:.2f} -> matched={best} (match={match_score:.2f})")
                    self.after(0, lambda p=best: self._set_plate_in_ui(p, "camera_s"))

        cap.release()
        cv2.destroyWindow(win)
        self.log_ui("[INFO] Camera closed (press 's' mode).")

    # ==========================
    # 2) LIVE scan auto
    # ==========================
    def on_toggle_live_scan(self):
        if self.live_running:
            self.live_running = False
            self.btn_scan_live.config(text="Start live scan (auto)")
            self.log("[INFO] Live scan stop requested.")
            return

        self.live_running = True
        self.btn_scan_live.config(text="Stop live scan (auto)")
        self.log("[INFO] Live scan started.")
        t = threading.Thread(target=self._live_scan_worker, daemon=True)
        t.start()

    def _live_scan_worker(self):
        try:
            self._ensure_ocr_reader()
        except Exception:
            self.after(0, lambda: messagebox.showerror("Error", "EasyOCR init failed."))
            self.live_running = False
            self.after(0, lambda: self.btn_scan_live.config(text="Start live scan (auto)"))
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_ui("[ERROR] Cannot open camera.")
            self.after(0, lambda: messagebox.showerror("Error", "Cannot open camera."))
            self.live_running = False
            self.after(0, lambda: self.btn_scan_live.config(text="Start live scan (auto)"))
            return

        win = "Live Scan - auto OCR (press 'q' to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        OCR_EVERY_SEC = 0.7
        MIN_OCR_CONF = 0.45
        MIN_MATCH_SCORE = 0.60
        SAME_PLATE_COOLDOWN = 1.6

        last_ocr = 0.0

        try:
            while self.live_running:
                ret, frame = cap.read()
                if not ret:
                    self.log_ui("[ERROR] Cannot read frame.")
                    break

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.live_running = False
                    break

                now = time.time()
                if now - last_ocr < OCR_EVERY_SEC:
                    continue
                last_ocr = now

                plate, plate_conf, _ = self._ocr_plate_from_frame(frame)
                if not plate or plate_conf < MIN_OCR_CONF:
                    continue

                best, match_score = match_plate_to_csv(plate, self.known_plates)
                if best is None or match_score < MIN_MATCH_SCORE:
                    continue

                if best == self.last_live_plate and (now - self.last_live_time) < SAME_PLATE_COOLDOWN:
                    continue

                self.last_live_plate = best
                self.last_live_time = now

                self.after(0, lambda p=best, oc=plate_conf, ms=match_score:
                           (self.log(f"[LIVE] Detected plate={p} (ocr={oc:.2f}, match={ms:.2f})"),
                            self._set_plate_in_ui(p, "live_auto")))

        finally:
            cap.release()
            cv2.destroyWindow(win)
            self.live_running = False
            self.after(0, lambda: self.btn_scan_live.config(text="Start live scan (auto)"))
            self.log_ui("[INFO] Live scan stopped.")

    # ==========================
    # 3) UPLOAD IMAGE OCR
    # ==========================
    def on_upload_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        t = threading.Thread(target=self._upload_image_worker, args=(path,), daemon=True)
        t.start()

    def _upload_image_worker(self, path: str):
        try:
            self._ensure_ocr_reader()
        except Exception:
            return

        img = cv2.imread(path)
        if img is None:
            self.log_ui(f"[ERROR] Could not read image: {path}")
            self.after(0, lambda: messagebox.showerror("Error", f"Could not read image:\n{path}"))
            return

        plate, plate_conf, _ = self._ocr_plate_from_frame(img)
        if not plate:
            self.log_ui("[WARN] No plate found in uploaded image.")
            self.after(0, lambda: messagebox.showwarning("OCR", "No plate found in uploaded image."))
            return

        best, match_score = match_plate_to_csv(plate, self.known_plates)
        if best is None:
            self.log_ui(f"[UPLOAD] OCR={plate} conf={plate_conf:.2f} not matched in CSV.")
            self.after(0, lambda p=plate: self._set_plate_in_ui(p, "upload_unmatched"))
            return

        self.log_ui(f"[UPLOAD] OCR={plate} conf={plate_conf:.2f} -> matched={best} (match={match_score:.2f})")
        self.after(0, lambda p=best: self._set_plate_in_ui(p, "upload"))

    # ---------- OCR core ----------
    def _ocr_plate_from_frame(self, frame) -> Tuple[Optional[str], float, str]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            if self.reader is None:
                raise RuntimeError("OCR reader not initialized")
            results = self.reader.readtext(gray)

            if not results:
                return None, 0.0, ""

            cands = []
            raw_debug = []
            for _, text, conf in results:
                t = clean_plate_text(text)
                raw_debug.append(f"{text}->{t}({conf:.2f})")
                if len(t) >= 5:
                    cands.append((t, float(conf)))

            if not cands:
                return None, 0.0, " | ".join(raw_debug)

            cands.sort(key=lambda x: x[1], reverse=True)
            return cands[0][0], cands[0][1], " | ".join(raw_debug)

        except Exception as e:
            return None, 0.0, f"exception: {e}"


class LoginDialog(tk.Toplevel):
    def __init__(self, parent: RiskApp, drivers_csv_path: str):
        super().__init__(parent)
        self.parent = parent
        self.drivers_csv_path = drivers_csv_path

        self.title("Login")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Autentificare șofer", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            container,
            text="Login/Register folosind numele complet + numărul de înmatriculare.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        nb = ttk.Notebook(container)
        nb.pack(fill=tk.BOTH, expand=True)

        # --- Login tab ---
        tab_login = ttk.Frame(nb, padding=12)
        nb.add(tab_login, text="Login")

        self.login_name = tk.StringVar()
        self.login_plate = tk.StringVar()

        ttk.Label(tab_login, text="Nume complet").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab_login, textvariable=self.login_name, width=34).grid(row=1, column=0, sticky="we", pady=(0, 10))
        ttk.Label(tab_login, text="Număr înmatriculare").grid(row=2, column=0, sticky="w")
        ttk.Entry(tab_login, textvariable=self.login_plate, width=34).grid(row=3, column=0, sticky="we", pady=(0, 10))

        btns_login = ttk.Frame(tab_login)
        btns_login.grid(row=4, column=0, sticky="e")
        ttk.Button(btns_login, text="Anulează", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns_login, text="Login", style="Accent.TButton", command=self._do_login).pack(side=tk.RIGHT)

        # --- Register tab ---
        tab_reg = ttk.Frame(nb, padding=12)
        nb.add(tab_reg, text="Register")

        self.reg_name = tk.StringVar()
        self.reg_plate = tk.StringVar()

        ttk.Label(tab_reg, text="Nume complet").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab_reg, textvariable=self.reg_name, width=34).grid(row=1, column=0, sticky="we", pady=(0, 10))
        ttk.Label(tab_reg, text="Număr înmatriculare").grid(row=2, column=0, sticky="w")
        ttk.Entry(tab_reg, textvariable=self.reg_plate, width=34).grid(row=3, column=0, sticky="we", pady=(0, 10))

        btns_reg = ttk.Frame(tab_reg)
        btns_reg.grid(row=4, column=0, sticky="e")
        ttk.Button(btns_reg, text="Anulează", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns_reg, text="Register", style="Accent.TButton", command=self._do_register).pack(side=tk.RIGHT)

        for t in (tab_login, tab_reg):
            t.grid_columnconfigure(0, weight=1)

        # Center on screen and make modal
        self.transient(parent)
        self.grab_set()
        self.after(50, self._center)
        self.after(100, lambda: self.focus_force())

    def _center(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _on_cancel(self):
        # Close everything if user cancels login.
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.parent.destroy()
        except Exception:
            pass

    def _do_login(self):
        name = normalize_full_name(self.login_name.get())
        plate = clean_plate_text(self.login_plate.get())
        if not name or not plate:
            messagebox.showwarning("Login", "Completează numele complet și numărul de înmatriculare.")
            return

        if not driver_exists(self.drivers_csv_path, name, plate):
            messagebox.showerror("Login", "Cont inexistent. Folosește Register prima dată.")
            return

        try:
            upsert_driver(self.drivers_csv_path, name, plate, set_created=False, set_last_login=True)
        except Exception as e:
            messagebox.showerror("Login", f"Nu pot actualiza fișierul de utilizatori:\n{e}")
            return

        self.parent.set_logged_driver(name, plate)
        try:
            self.parent.initialize_main_ui()
        except Exception as e:
            messagebox.showerror("Login", f"Eroare la inițializarea aplicației:\n{e}")
            return
        try:
            self.grab_release()
        except Exception:
            pass
        self.parent.deiconify()
        self.destroy()

    def _do_register(self):
        name = normalize_full_name(self.reg_name.get())
        plate = clean_plate_text(self.reg_plate.get())
        if not name or not plate:
            messagebox.showwarning("Register", "Completează numele complet și numărul de înmatriculare.")
            return

        # If already exists, treat it as a friendly message and allow login.
        if driver_exists(self.drivers_csv_path, name, plate):
            messagebox.showinfo("Register", "Contul există deja. Te loghez acum.")
            try:
                upsert_driver(self.drivers_csv_path, name, plate, set_created=False, set_last_login=True)
            except Exception as e:
                messagebox.showerror("Register", f"Nu pot actualiza fișierul de utilizatori:\n{e}")
                return
            self.parent.set_logged_driver(name, plate)
            try:
                self.parent.initialize_main_ui()
            except Exception as e:
                messagebox.showerror("Register", f"Eroare la inițializarea aplicației:\n{e}")
                return
            try:
                self.grab_release()
            except Exception:
                pass
            self.parent.deiconify()
            self.destroy()
            return

        try:
            upsert_driver(self.drivers_csv_path, name, plate, set_created=True, set_last_login=True)
        except Exception as e:
            messagebox.showerror("Register", f"Nu pot scrie fișierul de utilizatori:\n{e}")
            return

        messagebox.showinfo("Register", "Cont creat. Te loghez acum.")
        self.parent.set_logged_driver(name, plate)
        try:
            self.parent.initialize_main_ui()
        except Exception as e:
            messagebox.showerror("Register", f"Eroare la inițializarea aplicației:\n{e}")
            return
        try:
            self.grab_release()
        except Exception:
            pass
        self.parent.deiconify()
        self.destroy()


class LoginFrame(ttk.Frame):
    def __init__(self, parent: RiskApp, drivers_csv_path: str):
        super().__init__(parent, style="Panel.TFrame")
        self.parent = parent
        self.drivers_csv_path = drivers_csv_path

        ttk.Label(self, text="Autentificare șofer", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="Login/Register folosind numele complet + numărul de înmatriculare.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        card = ttk.Frame(self, style="Panel.TFrame", padding=12)
        card.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(card)
        nb.pack(fill=tk.BOTH, expand=True)

        # Login tab
        tab_login = ttk.Frame(nb, padding=12)
        nb.add(tab_login, text="Login")
        self.login_name = tk.StringVar()
        self.login_plate = tk.StringVar()

        ttk.Label(tab_login, text="Nume complet").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab_login, textvariable=self.login_name, width=40).grid(row=1, column=0, sticky="we", pady=(0, 10))
        ttk.Label(tab_login, text="Număr înmatriculare").grid(row=2, column=0, sticky="w")
        ttk.Entry(tab_login, textvariable=self.login_plate, width=40).grid(row=3, column=0, sticky="we", pady=(0, 10))

        btns_login = ttk.Frame(tab_login)
        btns_login.grid(row=4, column=0, sticky="e")
        ttk.Button(btns_login, text="Ieșire", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns_login, text="Login", style="Accent.TButton", command=self._do_login).pack(side=tk.RIGHT)

        # Register tab
        tab_reg = ttk.Frame(nb, padding=12)
        nb.add(tab_reg, text="Register")
        self.reg_name = tk.StringVar()
        self.reg_plate = tk.StringVar()

        ttk.Label(tab_reg, text="Nume complet").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab_reg, textvariable=self.reg_name, width=40).grid(row=1, column=0, sticky="we", pady=(0, 10))
        ttk.Label(tab_reg, text="Număr înmatriculare").grid(row=2, column=0, sticky="w")
        ttk.Entry(tab_reg, textvariable=self.reg_plate, width=40).grid(row=3, column=0, sticky="we", pady=(0, 10))

        btns_reg = ttk.Frame(tab_reg)
        btns_reg.grid(row=4, column=0, sticky="e")
        ttk.Button(btns_reg, text="Ieșire", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns_reg, text="Register", style="Accent.TButton", command=self._do_register).pack(side=tk.RIGHT)

        for t in (tab_login, tab_reg):
            t.grid_columnconfigure(0, weight=1)

    def _on_cancel(self):
        try:
            self.parent.destroy()
        except Exception:
            pass

    def _do_login(self):
        name = normalize_full_name(self.login_name.get())
        plate = clean_plate_text(self.login_plate.get())
        if not name or not plate:
            messagebox.showwarning("Login", "Completează numele complet și numărul de înmatriculare.")
            return
        if not driver_exists(self.drivers_csv_path, name, plate):
            messagebox.showerror("Login", "Cont inexistent. Folosește Register prima dată.")
            return

        try:
            upsert_driver(self.drivers_csv_path, name, plate, set_created=False, set_last_login=True)
        except Exception as e:
            messagebox.showerror("Login", f"Nu pot actualiza fișierul de utilizatori:\n{e}")
            return

        self.parent.set_logged_driver(name, plate)
        self.destroy()
        self.parent.initialize_main_ui()

    def _do_register(self):
        name = normalize_full_name(self.reg_name.get())
        plate = clean_plate_text(self.reg_plate.get())
        if not name or not plate:
            messagebox.showwarning("Register", "Completează numele complet și numărul de înmatriculare.")
            return

        if driver_exists(self.drivers_csv_path, name, plate):
            messagebox.showinfo("Register", "Contul există deja. Te loghez acum.")
            try:
                upsert_driver(self.drivers_csv_path, name, plate, set_created=False, set_last_login=True)
            except Exception as e:
                messagebox.showerror("Register", f"Nu pot actualiza fișierul de utilizatori:\n{e}")
                return
            self.parent.set_logged_driver(name, plate)
            self.destroy()
            self.parent.initialize_main_ui()
            return

        try:
            upsert_driver(self.drivers_csv_path, name, plate, set_created=True, set_last_login=True)
        except Exception as e:
            messagebox.showerror("Register", f"Nu pot scrie fișierul de utilizatori:\n{e}")
            return

        messagebox.showinfo("Register", "Cont creat. Te loghez acum.")
        self.parent.set_logged_driver(name, plate)
        self.destroy()
        self.parent.initialize_main_ui()


def main():
    app = RiskApp(defer_init=True)
    app.mainloop()


if __name__ == "__main__":
    main()
