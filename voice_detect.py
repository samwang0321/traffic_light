from ultralytics import YOLO
import cv2
import time
import pygame

# =========================
# 載入訓練好的模型
# =========================
model = YOLO(r"weights\best.pt")

# 印出模型類別名稱，方便確認紅燈綠燈名稱
print("模型類別名稱：", model.names)

# =========================
# 初始化音檔播放
# =========================
pygame.mixer.init()

green_sound = pygame.mixer.Sound(r"audio\green_ch.wav")  # 可以通行
red_sound = pygame.mixer.Sound(r"audio\red_ch.wav")      # 禁止通行

# 使用同一個播放通道，避免聲音重疊
voice_channel = pygame.mixer.Channel(0)

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(1)

# =========================
# 語音播放控制
# =========================
last_state = None
last_speak_time = 0
speak_interval = 3  # 每 3 秒播放一次

# 類別名稱設定
GREEN_LABELS = ["green", "greenlight", "green_light", "綠燈"]
RED_LABELS = ["red", "redlight", "red_light", "紅燈"]


def normalize_label(label):
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


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
    # 語音播放邏輯
    # 持續偵測到同一燈號，每 3 秒播放一次
    # =========================
    current_time = time.time()

    if detected_state == "green":
        if last_state != "green" or current_time - last_speak_time >= speak_interval:
            print("語音：可以通行")

            # 如果上一段聲音還沒播完，先停止，避免重疊
            voice_channel.stop()
            voice_channel.play(green_sound)

            last_state = "green"
            last_speak_time = current_time

    elif detected_state == "red":
        if last_state != "red" or current_time - last_speak_time >= speak_interval:
            print("語音：禁止通行")

            voice_channel.stop()
            voice_channel.play(red_sound)

            last_state = "red"
            last_speak_time = current_time

    else:
        # 沒有偵測到紅燈或綠燈時，不播放
        last_state = None

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
pygame.mixer.quit()