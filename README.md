# PCB Defect Detection using YOLOv8

A deep learning-based PCB defect detection system built using YOLOv8.

## Defect Classes

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

---

## Dataset

The dataset contains annotated PCB images in YOLO format.
https://www.kaggle.com/datasets/akhatova/pcb-defects
```
datasets/
    images/
        train/
        val/

    labels/
        train/
        val/
```

---

## Installation

```bash
git clone https://github.com/yourusername/PCB-Defect-Detection.git

cd PCB-Defect-Detection

pip install -r requirements.txt
```

---

## Training

```bash
python train1.py
```

---

## Prediction

```bash
python predict.py
```

---

## GUI

```bash
python interface.py
```

---

## Results

The model predicts six PCB defect classes using YOLOv8.

Training metrics are automatically saved in

```
runs/detect/
```

---

## Technologies Used

- Python
- YOLOv8
- PyTorch
- OpenCV
- Tkinter

---

## Author

Gokul Krishnan J
B.Tech Electronics and Communication Engineering
VIT Vellore
