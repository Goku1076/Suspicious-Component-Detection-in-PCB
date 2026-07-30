from pathlib import Path
from ultralytics import YOLO
import cv2


MODEL_PATH = Path("runs/pcb_detector/weights/best.pt")
IMAGE_PATH = Path("test_images/test.jpg")
OUTPUT_DIR = Path("outputs")

CONFIDENCE = 0.25


def predict(image_path):
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        save=False,
        verbose=False
    )

    result = results[0]

    image = result.plot()

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), image)

    print(f"\nPrediction saved to:\n{output_path}")

    print("\nDetected Objects\n")

    if len(result.boxes) == 0:
        print("No defects detected.")
        return

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"{class_name:20} {confidence:.2%}")


if __name__ == "__main__":
    predict(IMAGE_PATH)
