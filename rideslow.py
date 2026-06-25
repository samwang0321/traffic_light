from ultralytics import YOLO
import cv2
import time
import pygame

# =========================
# 載入 YOLO 模型
# =========================
model = YOLO(r"weights\best.pt")
print("模型類別名稱：", model.names)

# =========================
# 初始化音檔播放
# =========================
pygame.mixer.init()

green_sound = pygame.mixer.Sound(r"audio\green_ch.wav")        # 可以通行
red_sound = pygame.mixer.Sound(r"audio\red_ch.wav")            # 禁止通行
reminder_sound = pygame.mixer.Sound(r"audio\kihonggang.wav")  # 另一個提示音檔

voice_channel = pygame.mixer.Channel(0)

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(1)

# =========================
# 類別名稱設定
# =========================
GREEN_LABELS = ["green", "greenlight", "green_light", "綠燈"]
RED_LABELS = ["red", "redlight", "red_light", "紅燈"]

def normalize_label(label):
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")

# =========================
# 語音控制參數
# =========================
last_state = None
last_main_voice_time = 0
last_reminder_time = 0

main_voice_interval = 6      # 紅燈/綠燈主要語音間隔
reminder_interval = 6        # 另一個音檔播放間隔

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

    current_time = time.time()

    # =========================
    # 語音播放邏輯
    # =========================
    if detected_state == "green":

        # 燈號剛變成綠燈，立刻播放可以通行
        if last_state != "green":
            print("語音：可以通行")
            voice_channel.stop()
            voice_channel.play(green_sound)

            last_state = "green"
            last_main_voice_time = current_time
            last_reminder_time = current_time

        # 綠燈持續存在，每 3 秒播放一次可以通行
        elif current_time - last_main_voice_time >= main_voice_interval:
            print("語音：可以通行")
            voice_channel.stop()
            voice_channel.play(green_sound)

            last_main_voice_time = current_time

        # 綠燈持續存在，每 5 秒播放另一個音檔
        if current_time - last_reminder_time >= reminder_interval:
            print("語音：播放另一個提示音檔")
            voice_channel.stop()
            voice_channel.play(reminder_sound)

            last_reminder_time = current_time


    elif detected_state == "red":

        # 燈號剛變成紅燈，立刻播放禁止通行
        if last_state != "red":
            print("語音：禁止通行")
            voice_channel.stop()
            voice_channel.play(red_sound)

            last_state = "red"
            last_main_voice_time = current_time
            last_reminder_time = current_time

        # 紅燈持續存在，每 3 秒播放一次禁止通行
        elif current_time - last_main_voice_time >= main_voice_interval:
            print("語音：禁止通行")
            voice_channel.stop()
            voice_channel.play(red_sound)

            last_main_voice_time = current_time

        # 紅燈持續存在，每 5 秒播放另一個音檔
        if current_time - last_reminder_time >= reminder_interval:
            print("語音：播放另一個提示音檔")
            voice_channel.stop()
            voice_channel.play(reminder_sound)

            last_reminder_time = current_time


    else:
        # 沒偵測到紅燈或綠燈
        last_state = None

    # =========================
    # 顯示畫面
    # =========================
    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Light Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()