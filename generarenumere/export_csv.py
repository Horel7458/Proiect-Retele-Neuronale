import sqlite3
import csv

# Conectare la baza de date
conn = sqlite3.connect("plates.db")
cursor = conn.execute("SELECT id, plate, accidents FROM vehicles;")

# Numele fișierului CSV
csv_file = "plates_export.csv"

# Scriem în CSV
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Scriem header-ul (capul de tabel)
    writer.writerow(["id", "plate", "accidents"])

    # Scriem fiecare rând din baza de date
    for row in cursor:
        writer.writerow(row)

conn.close()

print(f"Export realizat cu succes în fișierul: {csv_file}")
