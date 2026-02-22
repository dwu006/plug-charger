import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
cap.release()

if not ret:
    print("FAILED: could not read frame")
else:
    cv2.imwrite("test.png", frame)
    print(f"Saved test.png ({frame.shape[1]}x{frame.shape[0]})")
