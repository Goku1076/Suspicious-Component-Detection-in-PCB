from ultralytics import YOLO
from pathlib import Path


DATASET = Path("C:/AIML Project/datasets/6cls/data.yaml")
MODEL = "yolov8n.pt"

PROJECT = Path("runs")
RUN_NAME = "pcb_detector"

EPOCHS = 150
IMAGE_SIZE = 640
BATCH = 8


def main():
    model = YOLO(MODEL)

    model.train(
        data=str(DATASET),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        project=str(PROJECT),
        name=RUN_NAME,
        device=0,
        workers=4,
        cache=True,
        pretrained=True,
        optimizer="auto",
        patience=30,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        verbose=True
    )

    print("\nTraining completed.\n")

    best_model = PROJECT / RUN_NAME / "weights" / "best.pt"

    if best_model.exists():
        print(f"Best model saved at:\n{best_model}")
    else:
        print("best.pt not found.")


if __name__ == "__main__":
    main()
