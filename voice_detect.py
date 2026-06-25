from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# =========================
# 載入訓練好的模型
# =========================
model = YOLO(r"weights\best.pt")

print("模型類別名稱：", model.names)

# =========================
# 類別名稱設定
# =========================
GREEN_LABELS = ["green", "greenlight", "green_light", "綠燈"]
RED_LABELS = ["red", "redlight", "red_light", "紅燈"]

# =========================
# 語音狀態控制
# =========================
current_state = None       # 目前燈號狀態：green / red / None
last_detect_time = 0       # 最後一次偵測到紅綠燈的時間
running = True             # 控制語音執行緒是否繼續
speak_interval = 3         # 每 3 秒講一次


def normalize_label(label):
    """
    統一類別名稱格式，避免 green light / green_light / greenlight 對不起來
    """
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


def voice_loop():
    """
    語音廣播迴圈：
    只要 current_state 是 green 或 red，就每隔 3 秒講一次
    """
    global current_state, running

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    last_spoken_state = None
    last_speak_time = 0

    while running:
        now = time.time()

        if current_state == "green":
            if last_spoken_state != "green" or now - last_speak_time >= speak_interval:
                print("語音：可以通行")
                engine.say("可以通行")
                engine.runAndWait()

                last_spoken_state = "green"
                last_speak_time = time.time()

        elif current_state == "red":
            if last_spoken_state != "red" or now - last_speak_time >= speak_interval:
                print("語音：禁止通行")
                engine.say("禁止通行")
                engine.runAndWait()

                last_spoken_state = "red"
                last_speak_time = time.time()

        else:
            last_spoken_state = None

        time.sleep(0.1)


# 啟動語音執行緒
voice_thread = threading.Thread(target=voice_loop, daemon=True)
voice_thread.start()

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 推論
    results = model.predict(
        source=frame,
        conf=0.6,
        verbose=False
    )

    detected_state = None
    best_conf = 0

    # =========================
    # 取得偵測結果
    # =========================
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls_id]
        class_name_norm = normalize_label(class_name)

        print("偵測到：", class_name_norm, "信心值：", conf)

        if class_name_norm in ["green", "greenlight", "綠燈"] and conf > best_conf:
            detected_state = "green"
            best_conf = conf

        elif class_name_norm in ["red", "redlight", "紅燈"] and conf > best_conf:
            detected_state = "red"
            best_conf = conf

    # =========================
    # 更新語音狀態
    # =========================
    if detected_state is not None:
        current_state = detected_state
        last_detect_time = time.time()
    else:
        # 如果超過 1.5 秒都沒偵測到紅燈或綠燈，就停止語音
        if time.time() - last_detect_time > 1.5:
            current_state = None

    # =========================
    # 顯示辨識畫面
    # =========================
    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Light Detection", annotated_frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# 結束程式
# =========================
running = False
cap.release()
cv2.destroyAllWindows()