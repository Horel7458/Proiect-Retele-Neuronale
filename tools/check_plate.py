import csv
import re

PLATE_RE = re.compile(r"[A-Z0-9]+")


def clean_plate_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).upper()
    parts = PLATE_RE.findall(s)
    return "".join(parts)


def main() -> None:
    path = "data/raw/plates_export.csv"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    plates_raw = [row.get("plate") for row in rows]
    plates_clean = [clean_plate_text(p) for p in plates_raw]
    known = set(plates_clean)

    q = "B555WHR"
    q_clean = clean_plate_text(q)

    print("query_clean:", q_clean)
    print("in_known:", q_clean in known)

    hits = [
        (i, plates_raw[i], plates_clean[i], rows[i].get("accidents"))
        for i in range(len(rows))
        if plates_clean[i] == q_clean
    ]

    print("hits_count:", len(hits))
    for i, raw, cl, acc in hits[:10]:
        print("hit", i, "raw_repr", repr(raw), "clean", cl, "acc", acc)

    if q_clean not in known:
        near = sorted({p for p in plates_clean if p.startswith("B")})[:30]
        print("sample_B_plates:", near)


if __name__ == "__main__":
    main()
