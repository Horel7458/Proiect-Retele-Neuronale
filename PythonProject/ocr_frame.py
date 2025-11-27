import easyocr
import cv2
import os

def main():
    # Calea către imagine (relativ la proiect)
    img_path = os.path.join("captured_frames", "frame_000.png")

    if not os.path.exists(img_path):
        print(f"Imaginea nu există: {img_path}")
        return

    # Citim imaginea cu OpenCV
    image = cv2.imread(img_path)

    if image is None:
        print("Nu am putut citi imaginea!")
        return

    # Inițializăm OCR-ul (limba engleză, fără GPU)
    reader = easyocr.Reader(['en'], gpu=False)

    # Aplicăm OCR pe imagine
    results = reader.readtext(image)

    print("\nRezultate OCR:")
    for (bbox, text, confidence) in results:
        print(f"Text: {text} | Încredere: {confidence:.2f}")

    # (Opțional) Afișăm imaginea
    cv2.imshow("Imagine originala", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
