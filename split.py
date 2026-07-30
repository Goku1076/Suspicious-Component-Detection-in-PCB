import os
import random
import shutil
from tqdm import tqdm
from ultralytics import YOLO

# Base dataset path
base_path = r"C:/AIML Project/datasets/6cls"

# Automatically detect folders
images_all = None
labels_all = None
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(folder_path)):
            images_all = folder_path
        elif any(f.lower().endswith('.txt') for f in os.listdir(folder_path)):
            labels_all = folder_path

if not images_all or not labels_all:
    raise FileNotFoundError("Could not find images or labels folder in the dataset!")

# Target folders for split data
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

# Make output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

# Collect all images
all_images = [f for f in os.listdir(images_all) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split
random.shuffle(all_images)
train_split, val_split = 0.75, 0.15
train_end = int(train_split * len(all_images))
val_end = int((train_split + val_split) * len(all_images))

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

# Load YOLO model (auto-download if missing)
model_path = os.path.join(base_path, "yolov8n.pt")
model = YOLO(model_path)

def generate_label(img_path, label_path):
    """Generate YOLO label if missing."""
    results = model.predict(source=img_path, save=False)
    if results:
        with open(label_path, "w") as f:
            for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
                x, y, w, h = box.tolist()
                cls = int(cls.tolist())
                img_w, img_h = results[0].orig_size
                f.write(f"{cls} {x/img_w} {y/img_h} {w/img_w} {h/img_h}\n")

def move_files(file_list, split_name):
    for img in tqdm(file_list, desc=f"Moving {split_name} files"):
        base = os.path.splitext(img)[0]
        label = base + ".txt"

        src_img = os.path.join(images_all, img)
        src_lbl = os.path.join(labels_all, label)
        dst_img = os.path.join(images_path, split_name, img)
        dst_lbl = os.path.join(labels_path, split_name, label)

        # Move image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

        # Move label if exists, else generate it
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f"⚠️ Missing label for {img}, generating now...")
            generate_label(dst_img, dst_lbl)

# Process all splits
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("\n✅ Dataset split and label generation completed successfully!")
