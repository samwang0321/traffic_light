from ultralytics import YOLO
import torch


def main():
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=500,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        amp=True,
        cache=True,
        patience=20,
        project="runs",
        name="traffic_light"
    )


if __name__ == "__main__":
    main()