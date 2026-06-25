# Traffic Light Detection with Voice Guidance

## 1. 專題簡介

本專題以 YOLOv8 物件偵測模型為核心，開發一套即時交通號誌辨識與語音提示系統。系統透過攝影機擷取即時影像，辨識畫面中的紅燈與綠燈，並依據辨識結果播放對應語音，提供視障者或行人通行輔助。

當系統偵測到綠燈時，會播放「可以通行」語音提示；當偵測到紅燈時，則播放「禁止通行」語音提示。透過影像辨識與語音廣播結合，本專題可應用於智慧交通、無障礙輔助系統與邊緣 AI 影像辨識等場景。

---

## 2. 系統需求

本系統主要需求如下：

* 即時讀取攝影機畫面
* 使用 YOLOv8 模型辨識交通號誌
* 判斷紅燈與綠燈狀態
* 顯示辨識框與即時影像畫面
* 根據燈號播放對應語音
* 支援自製音檔播放
* 可於本機端執行

---

## 3. 硬體與軟體環境

### 硬體環境

| 項目  | 說明                  |
| --- | ------------------- |
| 電腦  | Windows 筆電或桌機       |
| 攝影機 | USB Camera 或筆電內建攝影機 |
| 喇叭  | 播放語音提示              |
| 麥克風 | 錄製語音音檔使用            |
| GPU | 可選，用於加速模型推論         |

### 軟體環境

| 項目     | 說明                      |
| ------ | ----------------------- |
| 作業系統   | Windows 10 / Windows 11 |
| Python | 建議 Python 3.10          |
| 開發環境   | Visual Studio Code      |
| 影像辨識   | Ultralytics YOLOv8      |
| 影像處理   | OpenCV                  |
| 音檔播放   | pygame                  |
| 版本控制   | Git / GitHub            |

---

## 4. 安裝方式

### 4.1 建立虛擬環境

```bash
python -m venv yolov8
```

啟動虛擬環境：

```bash
yolov8\Scripts\activate
```

### 4.2 安裝套件

```bash
pip install ultralytics opencv-python pygame sounddevice scipy numpy
```

---

## 5. 執行方式

### 5.1 執行主程式

```bash
python test.py
```

### 5.2 操作說明

* 偵測到綠燈：播放「可以通行」
* 偵測到紅燈：播放「禁止通行」
* 按下 `q` 鍵：結束程式

### 5.3 攝影機設定

程式中預設使用：

```python
cap = cv2.VideoCapture(1)
```

若攝影機無法開啟，可改成：

```python
cap = cv2.VideoCapture(0)
```

---

## 6. 資料集說明

本專題資料集主要用於訓練交通號誌辨識模型，內容包含不同角度、距離與光線環境下的紅燈與綠燈影像。

### 類別標籤

| 類別名稱               | 說明 |
| ------------------ | -- |
| red / redlight     | 紅燈 |
| green / greenlight | 綠燈 |

### YOLO 資料集格式

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 7. 模型或演算法說明

本系統使用 YOLOv8 進行交通號誌辨識。YOLOv8 屬於即時物件偵測模型，可在單次推論中完成物件位置偵測與類別判斷，適合應用於即時影像辨識任務。

### 系統流程

```text
攝影機擷取影像
        ↓
YOLOv8 模型推論
        ↓
辨識紅燈 / 綠燈
        ↓
繪製辨識框
        ↓
判斷燈號狀態
        ↓
播放對應語音提示
```

### 語音提示邏輯

| 偵測結果 | 播放語音 |
| ---- | ---- |
| 綠燈   | 可以通行 |
| 紅燈   | 禁止通行 |

系統可設定語音播放間隔，避免語音過度重複播放。

---

## 8. 測試結果

| 測試項目            | 測試結果 |
| --------------- | ---- |
| 攝影機即時影像讀取       | 成功   |
| YOLOv8 模型載入     | 成功   |
| 紅燈辨識            | 成功   |
| 綠燈辨識            | 成功   |
| Bounding Box 顯示 | 成功   |
| 語音提示播放          | 成功   |
| 按 q 結束程式        | 成功   |

### 信心值設定

程式中可透過 `conf` 參數調整辨識信心值：

```python
conf=0.6
```

若誤判較多，可提高信心值；若偵測不到目標，可降低信心值。

---

## 9. Demo 影片

Demo 影片連結：

```markdown
[觀看 Demo 影片](https://github.com/samwang0321/traffic_light/tree/main/demo)
```


---

## 10. Team Members

| 姓名  | 負責項目          |
| --- | ------------- |
| 王順億 | YOLOv8 模型訓練、系統測試、報告製作  |
| 張智勝 | Python 程式開發、系統測試、報告製作  |
| 柯廷儒 | 資料集蒐集與標註、系統測試、報告製作 |

---

## 專案資料夾結構

```text
traffic_light/
├── README.md
├── test.py
├── record_one_audio.py
├── weights/
│   └── best.pt
└── audio/
    ├── green.wav
    ├── red.wav
    └── reminder.wav
```

---

## GitHub 更新方式

修改程式後，可使用以下指令更新至 GitHub：

```bash
git status
git add .
git commit -m "更新交通號誌辨識系統"
git push
```

---

## 後續改善方向

未來可進一步加入以下功能：

* 黃燈辨識
* 行人號誌辨識
* 倒數秒數辨識
* 語音播放優化
* Raspberry Pi 邊緣部署
* GPS 或地圖輔助定位
* 視障者智慧通行輔助系統整合
