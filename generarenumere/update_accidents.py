import sqlite3

def update_accidents(plate, new_accidents):
    conn = sqlite3.connect("plates.db")
    cursor = conn.cursor()

    sql = "UPDATE vehicles SET accidents = ? WHERE plate = ?"
    cursor.execute(sql, (new_accidents, plate))
    conn.commit()

    if cursor.rowcount == 0:
        print("Număr de înmatriculare INEXISTENT în baza de date!")
    else:
        print(f"Accidentele pentru {plate} au fost actualizate la {new_accidents}.")

    conn.close()


if __name__ == "__main__":
    plate = input("Număr înmatriculare (ex: B123ABC): ").strip().upper()
    new_accidents = int(input("Număr nou de accidente: ").strip())

    update_accidents(plate, new_accidents)
