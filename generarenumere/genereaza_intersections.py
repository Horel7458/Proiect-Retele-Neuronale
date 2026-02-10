import pandas as pd
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "intersections.csv"

intersections = [
    "Arcul de Triumf",
    "Piata Romana",
    "Piata Unirii",
    "Piata Universitatii"
]

intervals = [
    ("Dimineata", "06:30-09:30", (5, 15)),   # trafic mare → accidente moderate
    ("Pranz", "12:00-14:00", (1, 5)),       # cele mai putine
    ("Seara", "15:30-19:00", (10, 25))       # cele mai multe
]

rows = []

for intersection in intersections:
    for label, time_range, (low, high) in intervals:
        rows.append({
            "intersection": intersection,
            "interval_label": label,
            "time_range": time_range,
            "accidents": random.randint(low, high)
        })

df = pd.DataFrame(rows)

# Creeaza folderul daca nu exista
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Fisier generat: {OUTPUT_PATH}")
print(df)
