import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera nu poate fi deschisa!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu pot citi frame de la camera!")
            break

        cv2.imshow("Camera laptop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
