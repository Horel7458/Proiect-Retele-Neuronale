import cv2
import os

def main():
    # se creeaza un folder pt poze
    output_dir = "captured_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nu pot deschide camera!")
        return

    print("Apasă 's' ca să salvezi un cadru, 'q' ca să ieși.")

    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu pot citi frame de la camera!")
            break

        cv2.imshow("Capture frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # când apeși 's', salvăm imaginea
            img_name = f"frame_{img_counter:03d}.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Am salvat imaginea: {img_path}")
            img_counter += 1

        elif key == ord('q'):
            # când apeși 'q', ieșim
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
