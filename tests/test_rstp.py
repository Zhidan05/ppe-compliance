import cv2

RTSP_URL = "rtsp://camppe:passwordppe@192.168.2.253:554/stream1"

print("Mencoba membuka RTSP...")
print(RTSP_URL.rsplit("@", 1)[-1])

cap = cv2.VideoCapture(RTSP_URL)

print("Is opened:", cap.isOpened())

if cap.isOpened():
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame")
            break

        cv2.imshow("Tapo Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()