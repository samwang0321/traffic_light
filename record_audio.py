import sounddevice as sd
from scipy.io.wavfile import write
import os

# =========================
# 錄音設定
# =========================
sample_rate = 44100  # 取樣率
duration = 3         # 錄音秒數，可自行修改

# 儲存資料夾
os.makedirs("audio", exist_ok=True)

# =========================
# 設定輸出音檔名稱
# =========================
# 錄綠燈語音就用 green.wav
# 錄紅燈語音就改成 red.wav
filename = "audio/rideslow.wav"

print("=" * 40)
print("準備錄音")
print(f"錄音時間：{duration} 秒")
print(f"儲存位置：{filename}")
input("按 Enter 開始錄音...")

print("開始錄音，請講話...")

audio_data = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="int16"
)

sd.wait()

write(filename, sample_rate, audio_data)

print("錄音完成")
print(f"已儲存：{filename}")
