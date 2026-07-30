from ultralytics import YOLO
import os
import torch

def main():
    # -----------------------------
    # 1️⃣ Check GPU availability
    # -----------------------------
    print("✅ GPU detected:" if torch.cuda.is_available() else "❌ No GPU detected.")
    if torch.cuda.is_available():
        print("Using:", torch.cuda.get_device_name(0))

    # -----------------------------
    # 2️⃣ Paths & Config
    # -----------------------------
    BASE_PATH = r"C:/AIML Project/datasets/6cls"
    DATA_YAML = os.path.join(BASE_PATH, "data.yaml")  # dataset YAML
    SAVE_DIR = os.path.join(BASE_PATH, "training_results")  # folder to save runs

    MODEL_YAML = "yolov8n.yaml"
    EPOCHS = 150
    BATCH_SIZE = 8
    IMG_SIZE = 640
    LR0 = 1e-4
    OPTIMIZER = "SGD"

    # -----------------------------
    # 3️⃣ Initialize model
    # -----------------------------
    print("Initializing YOLOv8 model...")
    model = YOLO(MODEL_YAML)

    # -----------------------------
    # 4️⃣ Train
    # -----------------------------
    print("Starting YOLOv8 training on GPU..." if torch.cuda.is_available() else "Starting YOLOv8 training on CPU...")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LR0,
        optimizer=OPTIMIZER,
        project=SAVE_DIR,
        name="pcb_scratch",
        exist_ok=True,
        save_period=10,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,  # 👈 fix multiprocessing crash on Windows
        amp=True
    )

    # -----------------------------
    # 5️⃣ Evaluate & Plot
    # -----------------------------
    print("Training complete. Evaluating on validation set...")
    metrics = model.val()
    print(metrics)

    model.plot_results()
    print(f"Plots saved at {os.path.join(SAVE_DIR, 'pcb_scratch')}")

    # -----------------------------
    # 6️⃣ Save final model
    # -----------------------------
    final_model_path = os.path.join(SAVE_DIR, "pcb_scratch", "weights", "best_from_scratch.pt")
    model.save(final_model_path)
    print(f"Trained model saved at: {final_model_path}")


if __name__ == "__main__":
    main()
