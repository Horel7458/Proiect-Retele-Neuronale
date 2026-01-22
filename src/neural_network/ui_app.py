import os
import re
import json
import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import pandas as pd

# UI
import tkinter as tk
from tkinter import ttk, messagebox

# Camera + OCR
import cv2
import easyocr

# NN (optional)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# =========================
# Paths (edit if needed)
# =========================
PLATES_CSV = r"D:\Proiect retele neuronale\data\raw\plates_export.csv"
INTERSECTIONS_CSV = r"D:\Proiect retele neuronale\data\raw\intersections.csv"
COUNTY_STATS_CSV = r"D:\Proiect retele neuronale\data\processed\stats_by_judet.csv"

MODEL_PATH = r"D:\Proiect retele neuronale\data\processed\model.pth"
SCALER_PATH = r"D:\Proiect retele neuronale\data\processed\nn_scaler.json"


# =========================
# Helpers: normalization + fuzzy match
# =========================

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


def clean_plate_text(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    parts = PLATE_RE.findall(s)
    return "".join(parts)


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


def generate_substitution_candidates(plate: str, max_candidates: int = 1000) -> List[str]:
    p = clean_plate_text(plate)
    cands = {p}
    for (a, b) in SUBS:
        for i, ch in enumerate(p):
            if ch == a:
                cands.add(p[:i] + b + p[i + 1:])
            if ch == b:
                cands.add(p[:i] + a + p[i + 1:])
    out = list(cands)
    out.sort()
    return out[:max_candidates]


def match_plate_to_csv(ocr_plate: str, known_plates: List[str]) -> Tuple[Optional[str], float]:
    p = clean_plate_text(ocr_plate)
    if not p:
        return None, 0.0

    known_set = set(known_plates)

    if p in known_set:
        return p, 1.0

    sub_cands = generate_substitution_candidates(p)
    for c in sub_cands:
        if c in known_set:
            return c, 0.90

    same_len = [k for k in known_plates if abs(len(k) - len(p)) <= 1]
    best = None
    best_d = 10**9
    for k in same_len:
        d = levenshtein(p, k)
        if d < best_d:
            best_d = d
            best = k

    if best is not None and best_d <= 2:
        score = max(0.0, 0.85 - 0.2 * best_d)
        return best, score

    return None, 0.0


# =========================
# NN model + scaler loader
# =========================

class RiskMLP(nn.Module):
    def __init__(self, in_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def load_model_if_available(log_fn):
    if torch is None or nn is None:
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

    # checkpoint dict?
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
    # fallback formula if no torch/model
    if model is None or scaler is None or torch is None:
        acc_i, acc_v, county_s = features
        # more "balanced" so it doesn't always go HIGH
        raw = 0.40 * acc_i + 0.35 * acc_v + 0.25 * county_s
        # scale by typical ranges
        # acc_i can be ~0..30, acc_v ~0..10, county_s ~0..8
        raw_scaled = raw / 18.0
        return float(1.0 / (1.0 + pow(2.718281828, -raw_scaled)))

    x_norm = normalize_features(features, scaler)
    xt = torch.tensor([x_norm], dtype=torch.float32)
    with torch.no_grad():
        y = model(xt).item()
    return float(y)


def risk_category(r: float) -> str:
    # adjust thresholds a bit so results diversify
    if r < 0.40:
        return "LOW"
    if r < 0.70:
        return "MEDIUM"
    return "HIGH"


def compute_confidence(features: List[float], scaler: Optional[dict]) -> float:
    """
    Simple confidence proxy:
    - high if features are inside min/max learned ranges
    - lower if they are out of range (extrapolation)
    Returns 0..1
    """
    if not scaler:
        return 0.50

    mins = scaler.get("min", {})
    maxs = scaler.get("max", {})
    cols = scaler.get("feature_cols", ["acc_intersection", "acc_vehicle", "county_score"])

    penalties = []
    for i, v in enumerate(features):
        col = cols[i] if i < len(cols) else f"f{i}"
        mn = float(mins.get(col, 0.0))
        mx = float(maxs.get(col, 1.0))

        if v < mn:
            # how far below range
            penalties.append(min(1.0, (mn - v) / (mx - mn + 1e-9)))
        elif v > mx:
            penalties.append(min(1.0, (v - mx) / (mx - mn + 1e-9)))
        else:
            penalties.append(0.0)

    # confidence decreases with average penalty
    avg_pen = sum(penalties) / max(1, len(penalties))
    conf = 1.0 - avg_pen
    return max(0.0, min(1.0, conf))


def contribution_breakdown(acc_i: float, acc_v: float, county_s: float) -> Tuple[float, float, float]:
    """
    Explainability proxy:
    normalize values to compute % contribution.
    """
    # small smoothing to avoid division by zero
    a = max(0.0, acc_i)
    b = max(0.0, acc_v)
    c = max(0.0, county_s)
    total = a + b + c + 1e-9
    return (a / total, b / total, c / total)


# =========================
# Data structures
# =========================

@dataclass
class IntersectionKey:
    intersection: str
    interval_label: str


# =========================
# Main App
# =========================

class RiskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Risk App (Intersection + Plate + County)")
        self.geometry("1050x650")

        self.plates_df = None
        self.intersections_df = None
        self.county_df = None

        self.plate_to_acc: Dict[str, int] = {}
        self.known_plates: List[str] = []

        self.int_acc_map: Dict[Tuple[str, str], int] = {}
        self.county_score_map: Dict[str, float] = {}

        self.model = None
        self.scaler = None

        self.reader = None  # EasyOCR

        self._build_ui()
        self._load_all()

    # ---------- UI ----------
    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        ttk.Label(left, text="Intersection").grid(row=0, column=0, sticky="w", pady=4)
        self.cmb_intersection = ttk.Combobox(left, values=[], state="readonly", width=35)
        self.cmb_intersection.grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(left, text="Interval").grid(row=1, column=0, sticky="w", pady=4)
        self.cmb_interval = ttk.Combobox(left, values=[], state="readonly", width=35)
        self.cmb_interval.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(left, text="Plate (manual)").grid(row=2, column=0, sticky="w", pady=4)
        self.ent_plate = ttk.Entry(left, width=38)
        self.ent_plate.grid(row=2, column=1, sticky="w", pady=4)

        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)

        self.btn_scan = ttk.Button(btns, text="Scan plate (camera)", command=self.on_scan_camera)
        self.btn_scan.pack(side=tk.LEFT, padx=4)

        self.btn_calc = ttk.Button(btns, text="Calculate risk", command=self.on_calculate_risk)
        self.btn_calc.pack(side=tk.LEFT, padx=4)

        self.btn_add = ttk.Button(btns, text="Add accident (intersection+interval)", command=self.on_add_accident)
        self.btn_add.pack(side=tk.LEFT, padx=4)

        # Current data box
        box = ttk.LabelFrame(left, text="Current data")
        box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=10)
        left.grid_rowconfigure(4, weight=1)
        left.grid_columnconfigure(1, weight=1)

        self.lbl_int_acc = ttk.Label(box, text="Intersection accidents: -")
        self.lbl_int_acc.pack(anchor="w", pady=2)

        self.lbl_veh_acc = ttk.Label(box, text="Vehicle accidents: -")
        self.lbl_veh_acc.pack(anchor="w", pady=2)

        self.lbl_county = ttk.Label(box, text="County code: -")
        self.lbl_county.pack(anchor="w", pady=2)

        self.lbl_county_score = ttk.Label(box, text="County score: -")
        self.lbl_county_score.pack(anchor="w", pady=2)

        # New: Breakdown + confidence
        expl = ttk.LabelFrame(left, text="Explainability (approx)")
        expl.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=10)

        self.lbl_conf = ttk.Label(expl, text="Model confidence: -")
        self.lbl_conf.pack(anchor="w", pady=2)

        self.lbl_contrib = ttk.Label(expl, text="Contrib (int/veh/county): -")
        self.lbl_contrib.pack(anchor="w", pady=2)

        # Right: Risk result
        res = ttk.LabelFrame(right, text="Risk result")
        res.pack(fill=tk.X, pady=5)

        ttk.Label(res, text="Risk (0..1):").pack(anchor="w", pady=2)
        self.lbl_risk_val = ttk.Label(res, text="-", font=("Segoe UI", 22, "bold"))
        self.lbl_risk_val.pack(anchor="w", pady=2)

        ttk.Label(res, text="Category:").pack(anchor="w", pady=2)
        self.lbl_risk_cat = ttk.Label(res, text="-", font=("Segoe UI", 16, "bold"))
        self.lbl_risk_cat.pack(anchor="w", pady=2)

        # Right: Log
        logbox = ttk.LabelFrame(right, text="Log")
        logbox.pack(fill=tk.BOTH, expand=True, pady=5)

        self.txt_log = tk.Text(logbox, height=18, width=55)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # events
        self.cmb_intersection.bind("<<ComboboxSelected>>", lambda e: self.refresh_current_data())
        self.cmb_interval.bind("<<ComboboxSelected>>", lambda e: self.refresh_current_data())
        self.ent_plate.bind("<KeyRelease>", lambda e: self.refresh_current_data())

    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    # ---------- Load data ----------
    def _load_all(self):
        try:
            self.plates_df = pd.read_csv(PLATES_CSV)
            self.plates_df.columns = [c.strip().lower() for c in self.plates_df.columns]
            self.plates_df["plate_clean"] = self.plates_df["plate"].astype(str).map(clean_plate_text)
            self.plate_to_acc = dict(zip(self.plates_df["plate_clean"], self.plates_df["accidents"].astype(int)))
            self.known_plates = list(self.plate_to_acc.keys())
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

    # ---------- Data getters ----------
    def get_selected_intersection_interval(self) -> Tuple[str, str]:
        return self.cmb_intersection.get(), self.cmb_interval.get()

    def get_plate_clean(self) -> str:
        return clean_plate_text(self.ent_plate.get())

    def get_vehicle_accidents(self, plate_clean: str) -> int:
        return int(self.plate_to_acc.get(plate_clean, 0))

    def get_intersection_accidents(self, intersection: str, interval_label: str) -> int:
        return int(self.int_acc_map.get((intersection, interval_label), 0))

    def get_county_score(self, county: str) -> float:
        return float(self.county_score_map.get(county, 0.0))

    # ---------- UI updates ----------
    def refresh_current_data(self):
        inter, interval = self.get_selected_intersection_interval()
        plate_clean = self.get_plate_clean()

        acc_i = self.get_intersection_accidents(inter, interval)
        acc_v = self.get_vehicle_accidents(plate_clean)
        county = county_from_plate(plate_clean)
        county_score = self.get_county_score(county)

        self.lbl_int_acc.config(text=f"Intersection accidents: {acc_i}")
        self.lbl_veh_acc.config(text=f"Vehicle accidents: {acc_v}")
        self.lbl_county.config(text=f"County code: {county if county else '-'}")
        self.lbl_county_score.config(text=f"County score: {county_score:.3f}")

        # Update explainability preview too
        conf = compute_confidence([acc_i, acc_v, county_score], self.scaler)
        c1, c2, c3 = contribution_breakdown(acc_i, acc_v, county_score)
        self.lbl_conf.config(text=f"Model confidence: {conf:.2f}")
        self.lbl_contrib.config(text=f"Contrib (int/veh/county): {c1*100:.0f}% / {c2*100:.0f}% / {c3*100:.0f}%")

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

        acc_i = self.get_intersection_accidents(inter, interval)
        acc_v = self.get_vehicle_accidents(plate_clean)
        county = county_from_plate(plate_clean)
        county_score = self.get_county_score(county)

        self.log(f"[INFO] plate={plate_clean} acc_vehicle={acc_v} county={county} county_score={county_score:.3f}")
        self.log(f"[INFO] inter={inter} interval={interval} acc_intersection={acc_i}")

        r = predict_risk(self.model, self.scaler, [acc_i, acc_v, county_score])
        cat = risk_category(r)

        conf = compute_confidence([acc_i, acc_v, county_score], self.scaler)
        c1, c2, c3 = contribution_breakdown(acc_i, acc_v, county_score)

        self.lbl_risk_val.config(text=f"{r:.3f}")
        self.lbl_risk_cat.config(text=cat)

        self.lbl_conf.config(text=f"Model confidence: {conf:.2f}")
        self.lbl_contrib.config(text=f"Contrib (int/veh/county): {c1*100:.0f}% / {c2*100:.0f}% / {c3*100:.0f}%")

        self.refresh_current_data()
        self.log(f"[RISK] risk={r:.3f} category={cat} conf={conf:.2f} breakdown={c1:.2f}/{c2:.2f}/{c3:.2f}")

    def on_add_accident(self):
        inter, interval = self.get_selected_intersection_interval()
        if not inter or not interval:
            messagebox.showwarning("Warning", "Select intersection and interval first.")
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

    def on_scan_camera(self):
        t = threading.Thread(target=self._scan_camera_worker, daemon=True)
        t.start()

    def _scan_camera_worker(self):
        if self.reader is None:
            self.log("[INFO] Initializing EasyOCR (first run may take time)...")
            try:
                self.reader = easyocr.Reader(["en"], gpu=False)
                self.log("[OK] EasyOCR ready.")
            except Exception as e:
                self.log(f"[ERROR] EasyOCR init failed: {e}")
                messagebox.showerror("Error", f"EasyOCR init failed:\n{e}")
                return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("[ERROR] Cannot open camera.")
            messagebox.showerror("Error", "Cannot open camera.")
            return

        self.log("[INFO] Camera opened. Press 's' to OCR, 'q' to quit.")
        win = "Camera - press 's' to OCR, 'q' to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                self.log("[ERROR] Cannot read frame.")
                break
            last_frame = frame

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):
                self.log("[INFO] Running OCR on current frame...")
                plate, plate_score, ocr_raw = self._ocr_plate_from_frame(last_frame)

                if not plate:
                    self.log(f"[WARN] No plate found. OCR raw: {ocr_raw}")
                    continue

                best, match_score = match_plate_to_csv(plate, self.known_plates)

                if best is None:
                    self.log(f"[WARN] OCR={plate} (ocr_conf={plate_score:.2f}) not matched in CSV.")
                    self._set_plate_in_ui(plate)
                    continue

                self.log(f"[OK] OCR={plate} (ocr_conf={plate_score:.2f}) -> matched CSV plate={best} (match_score={match_score:.2f})")
                self._set_plate_in_ui(best)

        cap.release()
        cv2.destroyWindow(win)
        self.log("[INFO] Camera closed.")

    def _set_plate_in_ui(self, plate_clean: str):
        def _update():
            self.ent_plate.delete(0, tk.END)
            self.ent_plate.insert(0, plate_clean)
            self.refresh_current_data()
            acc_v = self.get_vehicle_accidents(plate_clean)
            self.log(f"[INFO] Plate {plate_clean} vehicle accidents (from CSV): {acc_v}")
        self.after(0, _update)

    def _ocr_plate_from_frame(self, frame) -> Tuple[Optional[str], float, str]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            results = self.reader.readtext(gray)

            if not results:
                return None, 0.0, ""

            cands = []
            raw_debug = []
            for bbox, text, conf in results:
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


def main():
    app = RiskApp()
    app.mainloop()


if __name__ == "__main__":
    main()
