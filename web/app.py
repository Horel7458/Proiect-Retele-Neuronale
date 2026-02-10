import csv
import json
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# QUICK NOTES (for evaluation, short comments)
# ------------------------------
# This file implements the web UI using Streamlit.
# The app is a simple end-to-end demo:
# - login/register (full name + plate) stored in CSV
# - select intersection + interval
# - manual plate input OR OCR from uploaded image
# - run model inference (risk score in [0,1]) and show category
# - optional update of counters to simulate new accidents
#
# Why Streamlit:
# - easy to demo on any PC with a browser
# - minimal server code, fast iteration
#
# Data files used (relative to repo root):
# - data/raw/plates_export.csv
# - data/raw/intersections.csv
# - data/processed/stats_by_judet.csv
# - data/processed/model.pth
# - data/processed/nn_scaler.json
# - data/processed/drivers.csv
#
# Heavy dependencies are optional:
# - cv2/easyocr are used only for OCR flows
# - torch is used only for model inference
# - if a dep is missing, we disable that feature gracefully
#
# Security note (scope of this school project):
# - CSV auth is a demo only, not production security
# - we avoid storing passwords; identity is (name + plate)
#
# UX note:
# - we map numeric score to LOW/MEDIUM/HIGH for readability
# - thresholds are consistent with desktop UI

# Extra walkthrough (short lines, but explicit):
# - Start app with: python -m streamlit run web/app.py
# - Login/Register uses drivers.csv (created on first write)
# - Sidebar stores session state (who is logged in)
# - Main page shows context selectors and actions
# - OCR flow uses an uploaded image (no camera required)
# - OCR is optional: app still works with manual plate input
# - Inference uses model.pth + nn_scaler.json if torch is installed
# - If torch is missing, we can show a fallback score (demo mode)
#
# App logic blocks (high level):
# - UI theme helpers (CSS)
# - path resolution (find_project_root)
# - csv helpers (drivers storage)
# - data loading (plates/intersections/county)
# - ocr helpers (clean_plate_text + optional easyocr)
# - model helpers (RiskMLP + load_model_and_scaler)
# - inference (build features -> scale -> predict)
# - update actions (write updated accidents counters)
#
# Debug tips:
# - If app cannot find CSVs, check ROOT detection and folder layout
# - If OCR fails, try a clearer image (contrast, crop)
# - If model fails to load, re-run training script
# - If you edit CSVs, refresh the page to reload

# Demo script (copy as a checklist when recording):
# 01) start streamlit
# 02) open browser page
# 03) register a new driver
# 04) logout
# 05) login with same identity
# 06) pick an intersection
# 07) pick an interval
# 08) type a known plate
# 09) click predict
# 10) show score + category
# 11) upload an image for OCR (optional)
# 12) run OCR and show extracted plate
# 13) predict again using OCR result
# 14) click update (+1 accident) (optional)
# 15) predict again and show score change
#
# File map (where to look when something breaks):
# - find_project_root: path resolution
# - load_* functions: CSV readers
# - clean_plate_text: plate normalization
# - match_plate_*: best-effort matching
# - load_model_and_scaler: torch + json loading
# - build_features: numeric vector creation
# - predict_*: model inference
# - update_*: write back to CSV
#
# Common corner cases:
# - empty plate input -> do not run inference
# - missing county stats -> use global mean
# - unknown county code -> fallback score
# - invalid numbers in CSV -> coerce to NaN and handle
# - torch missing -> disable model inference
# - easyocr missing -> disable OCR upload
# - cv2 missing -> do not use camera
#
# Performance notes:
# - dataset is small, so pandas ops are fast
# - model is tiny (MLP), inference is fast on CPU
# - OCR is the slowest step (depends on image size)


# ------------------------------
# Theme / UI helpers
# ------------------------------

THEME = {
        "bg": "#0b1220",
        "panel": "#0f1b33",
        "panel2": "#0b152b",
        "text": "#e6edf6",
        "muted": "#9fb1cc",
        "border": "rgba(255,255,255,0.08)",
        "shadow": "0 10px 24px rgba(0,0,0,0.35)",
        "accent": "#3b82f6",
        "accent2": "#22c55e",
        "warn": "#f59e0b",
        "danger": "#ef4444",
}


def inject_css() -> None:
        css = f"""
        <style>
            html, body, [class*="css"]  {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
            .stApp {{ background: radial-gradient(1200px 600px at 10% 0%, rgba(59,130,246,0.18), transparent 60%),
                                                radial-gradient(1000px 600px at 90% 10%, rgba(34,197,94,0.12), transparent 60%),
                                                {THEME['bg']}; color: {THEME['text']}; }}

            /* Top padding */
            section.main > div {{ padding-top: 1.1rem; }}

            /* Hide Streamlit chrome */
            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}
            header {{ visibility: hidden; }}

            /* Inputs */
            .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
                background: {THEME['panel2']} !important;
                border: 1px solid {THEME['border']} !important;
                color: {THEME['text']} !important;
                border-radius: 12px !important;
            }}
            .stTextInput input:focus {{ border-color: rgba(59,130,246,0.65) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important; }}

            /* Buttons */
            .stButton button {{
                border-radius: 12px !important;
                border: 1px solid {THEME['border']} !important;
                background: {THEME['panel2']} !important;
                color: {THEME['text']} !important;
                padding: 0.65rem 0.9rem !important;
            }}
            .stButton button:hover {{ border-color: rgba(59,130,246,0.55) !important; transform: translateY(-1px); }}
            .stButton button[kind="primary"] {{
                background: linear-gradient(135deg, {THEME['accent']}, #2563eb) !important;
                border-color: rgba(59,130,246,0.6) !important;
                color: white !important;
            }}

            /* Metrics */
            div[data-testid="stMetric"] {{
                background: rgba(255,255,255,0.03);
                border: 1px solid {THEME['border']};
                border-radius: 14px;
                padding: 14px 14px 10px 14px;
            }}
            div[data-testid="stMetric"] * {{ color: {THEME['text']}; }}

            /* Cards */
            .card {{
                background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                border: 1px solid {THEME['border']};
                border-radius: 16px;
                box-shadow: {THEME['shadow']};
                padding: 16px;
            }}
            .card-title {{ font-size: 0.95rem; font-weight: 700; color: {THEME['text']}; margin-bottom: 4px; }}
            .card-sub {{ font-size: 0.86rem; color: {THEME['muted']}; margin-bottom: 10px; }}

            /* Header */
            .app-title {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; }}
            .app-sub {{ color: {THEME['muted']}; margin-top: 0.25rem; }}
            .divider {{ height: 1px; background: {THEME['border']}; margin: 14px 0 14px 0; }}

            /* Badges */
            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid {THEME['border']};
                background: rgba(255,255,255,0.03);
                font-weight: 700;
                font-size: 0.85rem;
            }}
            .dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{ background: rgba(15,27,51,0.65); border-right: 1px solid {THEME['border']}; }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
        st.markdown(
                f"""
                <div class="card" style="padding: 18px 18px 14px 18px;">
                    <div class="app-title">{title}</div>
                    <div class="app-sub">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
        )


def badge(label: str, kind: str) -> None:
        color = {
                "LOW": THEME["accent2"],
                "MEDIUM": THEME["warn"],
                "HIGH": THEME["danger"],
                "INFO": THEME["accent"],
        }.get(kind, THEME["muted"])

        st.markdown(
                f"""<span class="badge"><span class="dot" style="background:{color}"></span>{label}</span>""",
                unsafe_allow_html=True,
        )

# Optional heavy deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None  # type: ignore

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore


# ------------------------------
# Paths
# ------------------------------

def find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


ROOT = find_project_root(os.path.dirname(__file__))

PLATES_CSV = os.path.join(ROOT, "data", "raw", "plates_export.csv")
INTERSECTIONS_CSV = os.path.join(ROOT, "data", "raw", "intersections.csv")
COUNTY_STATS_CSV = os.path.join(ROOT, "data", "processed", "stats_by_judet.csv")

MODEL_PATH = os.path.join(ROOT, "data", "processed", "model.pth")
SCALER_PATH = os.path.join(ROOT, "data", "processed", "nn_scaler.json")

DRIVERS_CSV = os.path.join(ROOT, "data", "processed", "drivers.csv")


# ------------------------------
# Driver storage
# ------------------------------

DRIVER_FIELDS = ["full_name", "full_name_key", "plate", "created_at", "last_login"]


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_full_name(name: str) -> str:
    return " ".join((name or "").strip().split())


def full_name_key(name: str) -> str:
    return normalize_full_name(name).casefold()


PLATE_RE = re.compile(r"[A-Z0-9]+")


def clean_plate_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s).upper()
    parts = PLATE_RE.findall(s)
    return "".join(parts)


def load_drivers_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: List[Dict[str, str]] = []
        for r in reader:
            if not r:
                continue
            out.append({k: str(r.get(k, "") or "") for k in DRIVER_FIELDS})
        return out


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


def driver_exists(path: str, name: str, plate: str) -> bool:
    key = full_name_key(name)
    plate_clean = clean_plate_text(plate)
    if not key or not plate_clean:
        return False
    for r in load_drivers_csv(path):
        if r.get("full_name_key") == key and r.get("plate") == plate_clean:
            return True
    return False


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
        if r.get("full_name_key") == key and r.get("plate") == plate_clean:
            idx = i
            break

    if idx is None:
        rows.append(
            {
                "full_name": name_norm,
                "full_name_key": key,
                "plate": plate_clean,
                "created_at": now if set_created else "",
                "last_login": now if set_last_login else "",
            }
        )
    else:
        rows[idx]["full_name"] = name_norm
        rows[idx]["full_name_key"] = key
        rows[idx]["plate"] = plate_clean
        if set_created and not rows[idx].get("created_at"):
            rows[idx]["created_at"] = now
        if set_last_login:
            rows[idx]["last_login"] = now

    save_drivers_csv(path, rows)


# ------------------------------
# Data + model
# ------------------------------


def county_from_plate(plate: str) -> str:
    p = clean_plate_text(plate)
    if len(p) >= 2 and p[0].isalpha() and p[1].isalpha():
        return p[:2]
    if len(p) >= 1 and p[0].isalpha():
        return p[:1]
    return ""


@st.cache_data(show_spinner=False)
def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plates = pd.read_csv(PLATES_CSV)
    intersections = pd.read_csv(INTERSECTIONS_CSV)
    county = pd.read_csv(COUNTY_STATS_CSV)
    return plates, intersections, county


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=os.path.dirname(path))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if nn is not None:
    _nn = nn

    class RiskMLP(_nn.Module):
        def __init__(self, in_dim: int = 3):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(in_dim, 16),
                _nn.ReLU(),
                _nn.Linear(16, 8),
                _nn.ReLU(),
                _nn.Linear(8, 1),
                _nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)
else:
    RiskMLP = None  # type: ignore


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    if torch is None or nn is None or RiskMLP is None:
        return None, None
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    in_dim = len(scaler.get("feature_cols", ["acc_intersection", "acc_vehicle", "county_score"]))
    model = RiskMLP(in_dim=in_dim)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    return model, scaler


def normalize_features(features: List[float], scaler: dict) -> List[float]:
    mins = scaler.get("min", {})
    maxs = scaler.get("max", {})
    cols = scaler.get("feature_cols", [])

    out: List[float] = []
    for i, val in enumerate(features):
        col = cols[i] if i < len(cols) else f"f{i}"
        mn = float(mins.get(col, 0.0))
        mx = float(maxs.get(col, 1.0))
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        out.append((float(val) - mn) / denom)
    return out


def predict_risk(model, scaler, features: List[float]) -> float:
    # Fallback formula
    if model is None or scaler is None or torch is None:
        acc_i, acc_v, county_s = features
        raw = 0.55 * acc_i + 0.30 * acc_v + 0.15 * county_s
        return float(1.0 / (1.0 + np.exp(-raw / 10.0)))

    x_norm = normalize_features(features, scaler)
    xt = torch.tensor([x_norm], dtype=torch.float32)
    with torch.no_grad():
        return float(model(xt).item())


def risk_category(r: float) -> str:
    if r < 0.40:
        return "LOW"
    if r < 0.70:
        return "MEDIUM"
    return "HIGH"


# ------------------------------
# OCR (upload image)
# ------------------------------


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    if easyocr is None:
        return None
    return easyocr.Reader(["en"], gpu=False)


def ocr_from_bytes(image_bytes: bytes) -> List[Tuple[str, float]]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available")
    reader = get_ocr_reader()
    if reader is None:
        raise RuntimeError("EasyOCR not available")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    results = reader.readtext(gray)

    cands: List[Tuple[str, float]] = []
    for _, text, conf in results:
        t = clean_plate_text(text)
        if len(t) >= 5:
            cands.append((t, float(conf)))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:10]


# ------------------------------
# App
# ------------------------------


def ensure_session():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("full_name", "")
    st.session_state.setdefault("plate", "")


def login_screen():
    page_header("Risk App (Web)", "Login/Register: nume complet + număr de înmatriculare")

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(
            """<div class="card"><div class="card-title">Autentificare</div>
            <div class="card-sub">Conectează-te ca șofer sau creează un cont nou.</div>""",
            unsafe_allow_html=True,
        )

        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                name = st.text_input("Nume complet", key="login_name")
                plate = st.text_input("Număr înmatriculare", key="login_plate")
                submitted = st.form_submit_button("Login", type="primary")
            if submitted:
                if not normalize_full_name(name) or not clean_plate_text(plate):
                    st.warning("Completează numele și plăcuța.")
                elif not driver_exists(DRIVERS_CSV, name, plate):
                    st.error("Cont inexistent. Folosește Register.")
                else:
                    upsert_driver(DRIVERS_CSV, name, plate, set_created=False, set_last_login=True)
                    st.session_state.logged_in = True
                    st.session_state.full_name = normalize_full_name(name)
                    st.session_state.plate = clean_plate_text(plate)
                    st.rerun()

        with tab_reg:
            with st.form("register_form", clear_on_submit=False):
                name = st.text_input("Nume complet", key="reg_name")
                plate = st.text_input("Număr înmatriculare", key="reg_plate")
                submitted = st.form_submit_button("Register", type="primary")
            if submitted:
                if not normalize_full_name(name) or not clean_plate_text(plate):
                    st.warning("Completează numele și plăcuța.")
                else:
                    if driver_exists(DRIVERS_CSV, name, plate):
                        st.info("Contul există deja. Te loghez.")
                        upsert_driver(DRIVERS_CSV, name, plate, set_created=False, set_last_login=True)
                    else:
                        upsert_driver(DRIVERS_CSV, name, plate, set_created=True, set_last_login=True)
                        st.success("Cont creat. Te loghez.")

                    st.session_state.logged_in = True
                    st.session_state.full_name = normalize_full_name(name)
                    st.session_state.plate = clean_plate_text(plate)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def main_screen():
    plates_df, intersections_df, county_df = load_dataframes()
    model, scaler = load_model_and_scaler()

    with st.sidebar:
        st.subheader("Șofer")
        badge(st.session_state.full_name or "Conectat", "INFO")
        st.write(f"Plăcuță: **{st.session_state.plate}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.full_name = ""
            st.session_state.plate = ""
            st.rerun()

        st.divider()
        st.caption("Fișiere")
        st.code(os.path.relpath(DRIVERS_CSV, ROOT))
        st.code(os.path.relpath(MODEL_PATH, ROOT))
        st.code(os.path.relpath(SCALER_PATH, ROOT))

    page_header(
        "Risk Estimator",
        "Intersecție + interval + plăcuță + județ → scor de risc (0..1)",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""<div class="card"><div class="card-title">Date de intrare</div>
    <div class="card-sub">Selectează intersecția / intervalul și confirmă plăcuța.</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.3, 1.0, 1.1])
    with c1:
        inter_list = sorted(intersections_df["intersection"].astype(str).unique().tolist())
        selected_inter = st.selectbox("Intersecție", inter_list)
    with c2:
        intervals = (
            intersections_df[intersections_df["intersection"].astype(str) == selected_inter]["interval_label"]
            .astype(str)
            .unique()
            .tolist()
        )
        intervals = sorted(intervals)
        selected_interval = st.selectbox("Interval", intervals)
    with c3:
        plate_input = st.text_input("Plăcuță", value=st.session_state.plate)
        plate_clean = clean_plate_text(plate_input)

    st.markdown("</div>", unsafe_allow_html=True)

    # Compute features
    mask = (intersections_df["intersection"].astype(str) == selected_inter) & (
        intersections_df["interval_label"].astype(str) == selected_interval
    )
    acc_intersection = int(intersections_df.loc[mask, "accidents"].fillna(0).astype(int).iloc[0]) if mask.any() else 0

    p_mask = plates_df["plate"].astype(str).str.upper() == plate_clean
    if p_mask.any():
        acc_vehicle = int(plates_df.loc[p_mask, "accidents"].fillna(0).astype(int).iloc[0])
    else:
        acc_vehicle = 0

    county_code = county_from_plate(plate_clean)
    c_mask = county_df["county_code"].astype(str).str.upper() == county_code
    if c_mask.any():
        county_score = float(county_df.loc[c_mask, "scor_mediu_accidente"].astype(float).iloc[0])
    else:
        county_score = float(pd.to_numeric(county_df["scor_mediu_accidente"], errors="coerce").fillna(0).mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Accidente intersecție", acc_intersection)
    col2.metric("Accidente vehicul", acc_vehicle)
    col3.metric("Scor județ", round(county_score, 3))

    r = predict_risk(model, scaler, [acc_intersection, acc_vehicle, county_score])
    cat = risk_category(r)

    st.markdown(
        """<div class="card"><div class="card-title">Risc</div>
        <div class="card-sub">Scor estimat de model (0..1) + categorie.</div>""",
        unsafe_allow_html=True,
    )
    badge(f"{cat}  (r={r:.3f})", cat)
    st.progress(min(max(float(r), 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Actualizări rapide")
    cA, cB = st.columns(2)
    with cA:
        if st.button("+1 accident la intersecție/interval"):
            if mask.any():
                intersections_df.loc[mask, "accidents"] = int(acc_intersection) + 1
            else:
                intersections_df = pd.concat(
                    [
                        intersections_df,
                        pd.DataFrame(
                            [
                                {
                                    "intersection": selected_inter,
                                    "interval_label": selected_interval,
                                    "time_range": "",
                                    "accidents": int(acc_intersection) + 1,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            _atomic_write_csv(intersections_df, INTERSECTIONS_CSV)
            load_dataframes.clear()
            st.success("Salvat în intersections.csv")
            st.rerun()

    with cB:
        if st.button("+1 accident la plăcuță"):
            if plate_clean:
                if p_mask.any():
                    plates_df.loc[p_mask, "accidents"] = int(acc_vehicle) + 1
                else:
                    plates_df = pd.concat(
                        [
                            plates_df,
                            pd.DataFrame([{ "id": int(plates_df["id"].max()) + 1 if len(plates_df) else 1, "plate": plate_clean, "accidents": int(acc_vehicle) + 1 }]),
                        ],
                        ignore_index=True,
                    )
                _atomic_write_csv(plates_df, PLATES_CSV)
                load_dataframes.clear()
                st.success("Salvat în plates_export.csv")
                st.rerun()
            else:
                st.warning("Plăcuță invalidă")

    st.divider()
    st.subheader("OCR (upload imagine)")
    up = st.file_uploader("Încarcă o imagine cu plăcuța", type=["png", "jpg", "jpeg"])
    if up is not None:
        st.image(up.getvalue(), caption="Imagine încărcată", use_container_width=True)
        with st.spinner("Rulez OCR..."):
            try:
                cands = ocr_from_bytes(up.getvalue())
            except Exception as e:
                st.error(str(e))
                cands = []

        if not cands:
            st.info("N-am găsit candidați (OCR).")
        else:
            st.write("Candidați:")
            for t, conf in cands:
                st.write(f"- {t} (conf={conf:.2f})")
            best = cands[0][0]
            if st.button(f"Setează plăcuța: {best}"):
                st.session_state.plate = best
                st.rerun()


def main():
    st.set_page_config(page_title="Risk App", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    ensure_session()

    if not st.session_state.logged_in:
        login_screen()
    else:
        main_screen()


if __name__ == "__main__":
    main()
