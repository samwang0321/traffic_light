from ultralytics import YOLO
import cv2

# 載入訓練好的模型
model = YOLO(
    r"runs\detect\runs\traffic_light-4\weights\best.pt"
)

# 開啟攝影機
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO推論
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    # 繪製框框
    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Light Detection", annotated_frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 