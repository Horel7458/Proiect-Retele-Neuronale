import pandas as pd
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = REPO_ROOT / "data" / "processed" / "nn_dataset.csv"
COUNTY_STATS_CSV = REPO_ROOT / "data" / "processed" / "stats_by_judet.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "nn_dataset_labeled.csv"

# weights (adjust if you want)
W_INT = 0.55
W_VEH = 0.30
W_COUNTY = 0.15

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def find_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def main():
    df = pd.read_csv(IN_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Try to locate intersection accidents column (RO/EN variants)
    col_int = find_col(df.columns, [
        "acc_intersection",
        "accidente_intersectie",
        "accidente_intersectii",
        "intersection_accidents",
        "accidents_intersection",
        "accidents_intersectie",
        "accidente"
    ])

    # Try to locate vehicle accidents column
    col_veh = find_col(df.columns, [
        "acc_vehicle",
        "accidente_vehicul",
        "vehicle_accidents",
        "accidents_vehicle",
        "accidents",
        "accidente_vehiculului"
    ])

    # County code column (needed to attach county score if missing)
    col_county_code = find_col(df.columns, [
        "county_code",
        "judet",
        "judet_code",
        "county"
    ])

    # County score column (may already exist)
    col_county_score = find_col(df.columns, [
        "county_score",
        "scor_judet",
        "scor_mediu_accidente",
        "scor_mediu_accident",
        "score_county"
    ])

    # If county_score is missing, build it from stats_by_judet.csv using county_code
    if col_county_score is None:
        if col_county_code is None:
            raise ValueError(
                "Nu am gasit nici county_score, nici county_code in dataset.\n"
                f"Coloane existente: {list(df.columns)}"
            )

        stats = pd.read_csv(COUNTY_STATS_CSV)
        stats.columns = [c.strip() for c in stats.columns]

        stats_county = find_col(stats.columns, ["county_code", "judet", "county"])
        stats_score = find_col(stats.columns, ["scor_mediu_accidente", "scor_mediu_accident", "county_score", "scor"])

        if stats_county is None or stats_score is None:
            raise ValueError(
                "Fisierul stats_by_judet.csv nu are coloanele necesare.\n"
                f"Coloane stats: {list(stats.columns)}"
            )

        stats[stats_county] = stats[stats_county].astype(str).str.strip().str.upper()
        df[col_county_code] = df[col_county_code].astype(str).str.strip().str.upper()

        score_map = dict(zip(stats[stats_county], stats[stats_score].astype(float)))
        df["county_score"] = df[col_county_code].map(score_map).fillna(0.0)
        col_county_score = "county_score"

    # Validate required cols
    missing = []
    if col_int is None:
        missing.append("intersection_accidents (accidente_intersectie)")
    if col_veh is None:
        missing.append("vehicle_accidents (accidente_vehicul)")
    if col_county_score is None:
        missing.append("county_score")

    if missing:
        raise ValueError(
            f"Lipsesc coloane necesare: {missing}\n"
            f"Coloane existente: {list(df.columns)}"
        )

    # Compute label in 0..1
    raw = (
        W_INT * df[col_int].astype(float) +
        W_VEH * df[col_veh].astype(float) +
        W_COUNTY * df[col_county_score].astype(float)
    )

    # divide to avoid saturating at 1.0 (so you get LOW/MEDIUM/HIGH)
    df["label"] = raw.apply(lambda v: sigmoid(v / 10.0))

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved labeled dataset: {OUT_CSV}")
    print("Columns:", list(df.columns))
    print(df[[col_int, col_veh, col_county_score, "label"]].head(10))

if __name__ == "__main__":
    main()
