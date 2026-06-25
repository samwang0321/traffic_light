from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# =========================
# 語音初始化
# =========================
engine = pyttsx3.init()
engine.setProperty("rate", 150)   # 語速，可調整
engine.setProperty("volume", 1.0) # 音量 0.0 ~ 1.0

is_speaking = False

def speak(text):
    global is_speaking

    if is_speaking:
        return

    def run():
        global is_speaking
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

    threading.Thread(target=run, daemon=True).start()


# =========================
# 載入訓練好的模型
# =========================
model = YOLO(
    r"weights\best.pt"
)

# 印出模型類別名稱，方便確認紅燈綠燈名稱
print("模型類別名稱：", model.names)

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(1)

# 語音播放控制
last_state = None
last_speak_time = 0
speak_interval = 3  # 至少間隔 3 秒才會再播一次，避免一直重複講

# 類別名稱設定
GREEN_LABELS = ["green", "greenlight", "green_light", "綠燈"]
RED_LABELS = ["red", "redlight", "red_light", "紅燈"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 推論
    results = model.predict(
        source=frame,
        conf=0.7,
        verbose=False
    )

    detected_state = None

    # 取得偵測結果
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()

        print("偵測到：", class_name)

        if class_name in GREEN_LABELS:
            detected_state = "green"

        elif class_name in RED_LABELS:
            detected_state = "red"

    # =========================
    # 語音判斷
    # =========================
    current_time = time.time()

    if detected_state == "green":
        if last_state != "green" or current_time - last_speak_time > speak_interval:
            speak("可以通行")
            last_state = "green"
            last_speak_time = current_time

    elif detected_state == "red":
        if last_state != "red" or current_time - last_speak_time > speak_interval:
            speak("禁止通行")
            last_state = "red"
            last_speak_time = current_time

    # 繪製框框
    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Light Detection", annotated_frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()