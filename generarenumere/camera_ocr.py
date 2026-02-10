import cv2
import easyocr
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Calea către CSV-ul cu numere + accidente
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "raw" / "plates_export.csv"


def clean_plate_text(text: str) -> str:
    """
    Curăță textul citit de OCR și păstrează doar litere și cifre.
    """
    text = text.replace(" ", "").upper()
    # păstrăm doar caractere A-Z și 0-9
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def levenshtein(a: str, b: str) -> int:
    """Calculează distanța Levenshtein (numărul minim de modificări între două string-uri)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # ștergere
                dp[i][j - 1] + 1,          # inserare
                dp[i - 1][j - 1] + cost    # înlocuire
            )
    return dp[m][n]


def get_accidents_for_plate(plate: str):
    """
    Caută în CSV plăcuța (curățată) și întoarce numărul de accidente.
    Dacă nu găsește exact, caută cea mai apropiată plăcuță (fuzzy match).
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ Nu găsesc fișierul CSV cu plăcuțe! Verifică calea CSV_PATH.")
        return None

    # Curățăm plăcuțele din CSV la fel ca pe cele din OCR
    df["plate_clean"] = df["plate"].astype(str).apply(clean_plate_text)

    plate_clean = clean_plate_text(plate)
    print(f"Caut în CSV numărul curățat: {plate_clean}")

    # 1) Căutare exactă
    match = df[df["plate_clean"] == plate_clean]
    if not match.empty:
        print("Rând găsit (match exact):", match.iloc[0].to_dict())
        return int(match.iloc[0]["accidents"])

    # 2) Căutare fuzzy: numărul cu distanța Levenshtein minimă
    df["distance"] = df["plate_clean"].apply(lambda x: levenshtein(plate_clean, x))
    best_row = df.sort_values("distance").iloc[0]

    print(
        "Cel mai apropiat număr din CSV:",
        best_row["plate_clean"],
        "(dist =", best_row["distance"],
        ")"
    )

    # Dacă e prea diferit (de ex. distanță > 2), îl considerăm că NU există
    if best_row["distance"] > 2:
        return None

    return int(best_row["accidents"])


def main():
    # inițializăm OCR-ul (prima dată poate dura câteva secunde)
    print("Pornesc EasyOCR... (așteaptă puțin la prima rulare)")
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nu pot deschide camera!")
        return

    print("Camera pornită.")
    print("Apasă 's' ca să încerci să citești numărul de înmatriculare.")
    print("Apasă 'q' ca să închizi fereastra.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu pot citi frame de la cameră!")
            break

        cv2.imshow("Camera - 's' pentru OCR, 'q' pentru iesire", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            # când apeși 's', aplicăm OCR pe frame-ul curent
            print("\n--- Pornesc OCR pe frame-ul curent ---")
            results = reader.readtext(frame)

            candidates = []

            for (bbox, text, conf) in results:
                cleaned = clean_plate_text(text)

                # Debug: vedem exact ce text citește OCR
                print(f"Text brut: '{text}'  -> curatat: '{cleaned}'  (conf={conf:.2f})")

                if not cleaned:
                    continue

                # desenăm conturul zonei detectate (doar vizual)
                pts = np.array(bbox, dtype=int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # afișăm textul și scorul pe imagine
                x, y = pts[0]
                cv2.putText(
                    frame,
                    f"{cleaned} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # FILTRU SIMPLU:
                if conf > 0.5 and 5 <= len(cleaned) <= 10:
                    candidates.append((cleaned, conf))

            if not candidates:
                print("Nu am găsit niciun număr de înmatriculare clar.")
            else:
                best = max(candidates, key=lambda c: c[1])
                plate = best[0]
                conf = best[1]
                print(f"\nNumăr detectat: {plate} (încredere: {conf:.2f})")

                # 🔍 Căutăm acest număr în CSV
                accidents = get_accidents_for_plate(plate)
                if accidents is None:
                    print(f"ℹ Numărul {plate} NU există (suficient de aproape) în fișierul CSV.")
                else:
                    print(f"✅ Pentru numărul {plate} sunt înregistrate {accidents} accidente în CSV.")

            cv2.imshow("Rezultat OCR", frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Rezultat OCR")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
