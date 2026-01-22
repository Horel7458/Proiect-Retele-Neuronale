import sqlite3

conn = sqlite3.connect("plates.db")
cursor = conn.execute("SELECT * FROM vehicles;")  # selectăm TOATE coloanele

# Aflăm numele coloanelor (id, plate, accidents, owner etc.)
column_names = [description[0] for description in cursor.description]

print("Coloane:", column_names)
print("-------------------------------")

for row in cursor:
    print(row)

conn.close()
