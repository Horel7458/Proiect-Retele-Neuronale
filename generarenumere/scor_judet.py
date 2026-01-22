import sqlite3

def get_county_score(county_code):
    county_code = county_code.upper()
    conn = sqlite3.connect("plates.db")
    cursor = conn.cursor()

    # selectăm accidentele pentru toate plăcuțele care încep cu codul de județ
    query = "SELECT accidents FROM vehicles WHERE plate LIKE ?"
    cursor.execute(query, (county_code + "%",))

    results = cursor.fetchall()
    conn.close()

    if len(results) == 0:
        print(f"Nu există plăcuțe în baza de date pentru județul '{county_code}'.")
        return

    total_acc = sum(row[0] for row in results)
    media = total_acc / len(results)

    print(f"\nJudeț: {county_code}")
    print(f"Număr vehicule: {len(results)}")
    print(f"Scor (media accidentelor): {media:.2f}")


if __name__ == "__main__":
    cod = input("Introduceți codul județului (ex: B, CJ, GL, IF): ").strip()
    get_county_score(cod)
