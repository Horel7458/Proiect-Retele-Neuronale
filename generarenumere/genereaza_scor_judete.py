import pandas as pd
import re
from pathlib import Path

# adaptează dacă e nevoie, dar la tine era exact așa:
CSV_INPUT = Path(r"D:\Proiect retele neuronale\data\raw\plates_export.csv")
CSV_OUTPUT = Path(r"D:\Proiect retele neuronale\data\processed\stats_by_judet.csv")


def extract_county_code(plate: str) -> str | None:
    """
    Extrage codul de județ din număr:
    ia toate literele de la începutul plăcuței (B, IF, CT, etc.).
    """
    if not isinstance(plate, str):
        plate = str(plate)

    plate = plate.upper().strip()
    m = re.match(r"^[A-Z]+", plate)
    return m.group(0) if m else None


def main():
    # 1. Citim CSV-ul de bază
    if not CSV_INPUT.exists():
        print(f"❌ Nu găsesc fișierul de intrare: {CSV_INPUT}")
        return

    df = pd.read_csv(CSV_INPUT)

    # Ne asigurăm că avem coloanele necesare
    if "plate" not in df.columns or "accidents" not in df.columns:
        print("❌ CSV-ul trebuie să conțină coloanele 'plate' și 'accidents'.")
        return

    # 2. Curățăm plăcuțele și extragem codul de județ
    df["plate"] = df["plate"].astype(str).str.upper().str.strip()
    df["judet"] = df["plate"].apply(extract_county_code)

    # 3. Eliminăm eventualele rânduri fără județ sau fără număr de accidente
    df_valid = df.dropna(subset=["judet", "accidents"]).copy()

    # 4. Calculăm scorul per județ (media accidentelor) + numărul de plăcuțe
    stats = (
        df_valid.groupby("judet")["accidents"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "judet": "county_code",
                "count": "numar_vehicule",
                "mean": "scor_mediu_accidente",
            }
        )
    )

    # Sortăm opțional după scor descrescător (cele mai „periculoase” primele)
    stats = stats.sort_values("scor_mediu_accidente", ascending=False)

    # 5. Creăm folderul processed dacă nu există
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 6. Salvăm în CSV
    stats.to_csv(CSV_OUTPUT, index=False, float_format="%.3f")

    print(f"✅ Scorurile pe județe au fost generate în:\n   {CSV_OUTPUT}")
    print(stats.head())


if __name__ == "__main__":
    main()
