import sqlite3

DB_NAME = "plates.db"

def create_connection():
    """Creează / deschide baza de date SQLite."""
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_table(conn):
    """Creează tabelul 'vehicles' dacă nu există."""
    sql = """
    CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT UNIQUE NOT NULL,
        accidents INTEGER DEFAULT 0
    );
    """
    conn.execute(sql)
    conn.commit()

def insert_vehicle(conn, plate: str, accidents: int = 0):
    """Inserează un vehicul nou în tabel."""
    sql = "INSERT OR IGNORE INTO vehicles (plate, accidents) VALUES (?, ?);"
    conn.execute(sql, (plate, accidents))
    conn.commit()

def list_vehicles(conn):
    """Afișează toate vehiculele din bază."""
    cursor = conn.execute("SELECT id, plate, accidents FROM vehicles ORDER BY id;")
    rows = cursor.fetchall()
    if not rows:
        print("Baza de date este goală.")
        return
    print("\nVehicule înregistrate:")
    for row in rows:
        print(f"ID={row[0]} | Nr={row[1]} | Accidente={row[2]}")

def main():
    conn = create_connection()
    create_table(conn)
    print(f"Baza de date '{DB_NAME}' este pregătită.\n")

    while True:
        print("\nAlege o opțiune:")
        print(" 1 – Adaugă un număr de înmatriculare")
        print(" 2 – Afișează toate vehiculele")
        print(" 3 – Ieșire")
        opt = input("Opțiune: ").strip()

        if opt == "1":
            plate = input("Număr de înmatriculare (ex: B123ABC): ").strip().upper()
            if not plate:
                print("Nu ai introdus nimic.")
                continue

            accidents_str = input("Număr accidente (ex: 0, 1, 2...): ").strip()
            if accidents_str.isdigit():
                accidents = int(accidents_str)
            else:
                print("Valoare invalidă, setez număr accidente = 0.")
                accidents = 0

            insert_vehicle(conn, plate, accidents)
            print(f"Am salvat plăcuța {plate} cu {accidents} accidente.")

        elif opt == "2":
            list_vehicles(conn)

        elif opt == "3":
            print("La revedere!")
            break

        else:
            print("Opțiune invalidă, încearcă din nou.")

    conn.close()

if __name__ == "__main__":
    main()
