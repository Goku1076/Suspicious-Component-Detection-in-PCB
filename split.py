import random
import shutil
from pathlib import Path

random.seed(42)

DATASET = Path("../datasets/6cls")

IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

TRAIN_IMAGE_DIR = IMAGE_DIR / "train"
VAL_IMAGE_DIR = IMAGE_DIR / "val"

TRAIN_LABEL_DIR = LABEL_DIR / "train"
VAL_LABEL_DIR = LABEL_DIR / "val"

VAL_RATIO = 0.2

TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

images = list((IMAGE_DIR / "all").glob("*.*"))

random.shuffle(images)

split = int(len(images) * (1 - VAL_RATIO))

train_images = images[:split]
val_images = images[split:]


def copy_files(image_list, image_dest, label_dest):
    for image in image_list:
        shutil.copy(image, image_dest / image.name)

        label = LABEL_DIR / "all" / f"{image.stem}.txt"

        if label.exists():
            shutil.copy(label, label_dest / label.name)


copy_files(train_images, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
copy_files(val_images, VAL_IMAGE_DIR, VAL_LABEL_DIR)

print(f"Training Images : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print("Dataset split completed.")
