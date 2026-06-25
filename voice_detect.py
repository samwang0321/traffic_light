from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# =========================
# 語音初始化
# =========================
engine = pyttsx3.init()
engine.setProperty("rate", 150)    # 語速
engine.setProperty("volume", 1.0)  # 音量 0.0 ~ 1.0

is_speaking = False

def speak(text):
    global is_speaking

    # 如果正在講話，就不要重疊播放
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
model = YOLO(r"weights\best.pt")

print("模型類別名稱：", model.names)

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(1)

# =========================
# 語音播放控制
# =========================
last_voice_state = None
last_speak_time = 0
speak_interval = 3  # 每 3 秒播放一次

# 類別名稱設定
GREEN_LABELS = ["green", "greenlight", "green_light", "綠燈"]
RED_LABELS = ["red", "redlight", "red_light", "紅燈"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # YOLO 推論
    # =========================
    results = model.predict(
        source=frame,
        conf=0.6,
        verbose=False
    )

    detected_state = None
    best_conf = 0

    # =========================
    # 取得偵測結果
    # 如果同時偵測到多個物件，選信心值最高的紅燈或綠燈
    # =========================
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id].lower()

        print("偵測到：", class_name, "信心值：", conf)

        if class_name in GREEN_LABELS and conf > best_conf:
            detected_state = "green"
            best_conf = conf

        elif class_name in RED_LABELS and conf > best_conf:
            detected_state = "red"
            best_conf = conf

    # =========================
    # 語音廣播邏輯
    # 紅燈/綠燈持續偵測到時，每 3 秒重複播放
    # 燈號改變時，立刻播放新的提示
    # =========================
    current_time = time.time()

    if detected_state == "green":
        if last_voice_state != "green" or current_time - last_speak_time >= speak_interval:
            speak("可以通行")
            last_voice_state = "green"
            last_speak_time = current_time

    elif detected_state == "red":
        if last_voice_state != "red" or current_time - last_speak_time >= speak_interval:
            speak("禁止通行")
            last_voice_state = "red"
            last_speak_time = current_time

    else:
        # 沒有偵測到紅燈或綠燈時，不播放語音
        # 這行可以讓下一次重新偵測到燈號時馬上講
        last_voice_state = None

    # =========================
    # 顯示辨識畫面
    # =========================
    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Light Detection", annotated_frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()