import os
import re
import pandas as pd


def find_project_root(start_path: str) -> str:
    """
    Finds project root by walking up until it sees a 'data' folder.
    This makes paths work no matter where you run the script from.
    """
    cur = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # reached filesystem root
            return os.path.abspath(start_path)
        cur = parent


def extract_county_code(plate: str) -> str:
    if plate is None:
        return ""
    plate = str(plate).strip().upper()
    m = re.match(r"^([A-Z]{1,2})", plate)
    return m.group(1) if m else ""


def normalize_01(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0.0
    return (s - mn) / (mx - mn)


def main():
    # --- robust paths ---
    # We detect the project root so this script can be run from anywhere.
    # This keeps paths relative and portable (important on Windows).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)

    plates_csv = os.path.join(project_root, "data", "raw", "plates_export.csv")
    intersections_csv = os.path.join(project_root, "data", "raw", "intersections.csv")

    # stats_by_judet can be in processed (recommended); fallback to raw if needed
    stats_processed = os.path.join(project_root, "data", "processed", "stats_by_judet.csv")
    stats_raw = os.path.join(project_root, "data", "raw", "stats_by_judet.csv")
    county_stats_csv = stats_processed if os.path.isfile(stats_processed) else stats_raw

    output_csv = os.path.join(project_root, "data", "processed", "nn_dataset.csv")

    # --- sanity checks ---
    # Fail fast if any required CSV is missing.
    # This prevents silent creation of an empty/broken dataset.
    missing = []
    for p in [plates_csv, intersections_csv, county_stats_csv]:
        if not os.path.isfile(p):
            missing.append(p)

    if missing:
        print("[ERROR] Missing file(s):")
        for p in missing:
            print("  -", p)
        print("\nTIP: Check your folder names and that you run inside the correct project.")
        return

    # 1) Read CSVs
    # plates_export.csv: known plates + accident counters per vehicle
    # intersections.csv: scenarios with accident counters per intersection
    # stats_by_judet.csv: county-level risk score (aggregated)
    plates = pd.read_csv(plates_csv)
    county = pd.read_csv(county_stats_csv)
    inters = pd.read_csv(intersections_csv)

    # 2) Standardize column names
    # Keep everything lowercase to avoid issues with CSV header casing.
    plates.columns = [c.strip().lower() for c in plates.columns]
    county.columns = [c.strip().lower() for c in county.columns]
    inters.columns = [c.strip().lower() for c in inters.columns]

    # Expected:
    # plates: id, plate, accidents
    # county: county_code, numar_vehicule, scor_mediu_accidente
    # inters: intersection, interval_label, time_range, accidents

    # 3) Extract county code from plate
    # Example: "B123ABC" -> "B", "AG44XYZ" -> "AG".
    plates["county_code"] = plates["plate"].apply(extract_county_code)

    county["county_code"] = county["county_code"].astype(str).str.strip().str.upper()
    plates["county_code"] = plates["county_code"].astype(str).str.strip().str.upper()

    # 4) Merge county score
    # Join plates with county score so we get 3 numeric features later.
    merged = plates.merge(
        county[["county_code", "scor_mediu_accidente"]],
        on="county_code",
        how="left"
    )

    # Fallback for missing county scores
    global_mean = float(county["scor_mediu_accidente"].mean())
    merged["scor_mediu_accidente"] = merged["scor_mediu_accidente"].fillna(global_mean)

    # 5) Cross join with intersection scenarios
    # Cross join generates final rows: plates x scenarios.
    # In this repo: 79 plates * 12 scenarios = 948 rows.
    inters_small = inters[["intersection", "interval_label", "time_range", "accidents"]].copy()
    inters_small = inters_small.rename(columns={"accidents": "accidente_intersectie"})

    merged["__key"] = 1
    inters_small["__key"] = 1
    nn = merged.merge(inters_small, on="__key", how="inner").drop(columns="__key")

    # 6) Rename for clarity
    nn = nn.rename(columns={
        "accidents": "accidente_vehicul",
        "scor_mediu_accidente": "scor_judet"
    })

    # 7) Normalize components and build heuristic label in [0,1]
    # Label is a simple weighted sum of normalized components.
    # This is a demo label (heuristic), not real ground-truth.
    nn["acc_inter_01"] = normalize_01(nn["accidente_intersectie"])
    nn["acc_veh_01"] = normalize_01(nn["accidente_vehicul"])
    nn["scor_judet_01"] = normalize_01(nn["scor_judet"])

    w_inter, w_veh, w_jud = 0.5, 0.3, 0.2
    nn["label_risk"] = (
        w_inter * nn["acc_inter_01"] +
        w_veh * nn["acc_veh_01"] +
        w_jud * nn["scor_judet_01"]
    ).clip(0.0, 1.0)

    # 8) Final dataset
    nn_final = nn[[
        "plate", "county_code",
        "intersection", "interval_label", "time_range",
        "accidente_intersectie", "accidente_vehicul", "scor_judet",
        "label_risk"
    ]].copy()

    # 9) Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    nn_final.to_csv(output_csv, index=False, encoding="utf-8")

    print("[OK] RN dataset created:", output_csv)
    print("Rows:", len(nn_final))
    print(nn_final.head(5))


if __name__ == "__main__":
    main()
