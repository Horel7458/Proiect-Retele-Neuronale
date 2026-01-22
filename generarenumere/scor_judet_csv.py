import pandas as pd

CSV_PATH = r"D:\Proiect retele neuronale\data\raw\plates_export.csv"

def get_county_score(county_code):
    county_code = county_code.upper()

    # Citim CSV-ul
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ Nu găsesc fișierul CSV! Verifică calea.")
        return

    # Filtram plăcuțele care încep cu codul județului
    df_filtered = df[df["plate"].str.startswith(county_code)]

    if df_filtered.empty:
        print(f"❌ Nu există plăcuțe pentru județul '{county_code}' în CSV.")
        return

    media = df_filtered["accidents"].mean()

    print(f"\nJudeț: {county_code}")
    print(f"Număr vehicule: {len(df_filtered)}")
    print(f"Scor (media accidentelor): {media:.2f}\n")

if __name__ == "__main__":
    cod = input("Introduceți codul județului (ex: B, CJ, GL, IF): ").strip()
    get_county_score(cod)
