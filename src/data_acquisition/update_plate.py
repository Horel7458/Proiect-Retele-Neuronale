import sqlite3

def update_plate(old_plate, new_plate):
    conn = sqlite3.connect("plates.db")
    cursor = conn.cursor()

    sql = "UPDATE vehicles SET plate = ? WHERE plate = ?"
    cursor.execute(sql, (new_plate, old_plate))
    conn.commit()

    if cursor.rowcount == 0:
        print("Placa veche NU a fost găsită în baza de date!")
    else:
        print(f"Placa {old_plate} a fost schimbată cu {new_plate}.")

    conn.close()


if __name__ == "__main__":
    old_plate = input("Numărul vechi: ").strip().upper()
    new_plate = input("Numărul nou: ").strip().upper()

    update_plate(old_plate, new_plate)
