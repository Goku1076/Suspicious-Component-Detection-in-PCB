from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


MODEL_PATH = Path("runs/pcb_detector/weights/best.pt")
CONFIDENCE = 0.25


class PCBDetectorApp:

    def __init__(self, root):

        self.root = root
        self.root.title("PCB Defect Detection")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        self.model = YOLO(str(MODEL_PATH))

        self.image_path = None
        self.result_image = None

        self.create_widgets()

    def create_widgets(self):

        title = tk.Label(
            self.root,
            text="PCB Defect Detection using YOLOv8",
            font=("Arial", 20, "bold")
        )

        title.pack(pady=15)

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        upload_btn = tk.Button(
            button_frame,
            text="Upload Image",
            width=20,
            command=self.upload_image
        )

        upload_btn.grid(row=0, column=0, padx=10)

        detect_btn = tk.Button(
            button_frame,
            text="Detect Defects",
            width=20,
            command=self.detect
        )

        detect_btn.grid(row=0, column=1, padx=10)

        save_btn = tk.Button(
            button_frame,
            text="Save Result",
            width=20,
            command=self.save_result
        )

        save_btn.grid(row=0, column=2, padx=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        self.result_text = tk.Text(
            self.root,
            width=70,
            height=10,
            font=("Consolas", 11)
        )

        self.result_text.pack()

    def upload_image(self):

        file_path = filedialog.askopenfilename(

            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if not file_path:
            return

        self.image_path = Path(file_path)

        image = cv2.imread(str(self.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.show_image(image)

        self.result_text.delete("1.0", tk.END)

    def detect(self):

        if self.image_path is None:
            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )
            return

        results = self.model.predict(
            source=str(self.image_path),
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        self.result_image = result.plot()

        image_rgb = cv2.cvtColor(
            self.result_image,
            cv2.COLOR_BGR2RGB
        )

        self.show_image(image_rgb)

        self.result_text.delete("1.0", tk.END)

        if len(result.boxes) == 0:

            self.result_text.insert(
                tk.END,
                "No defects detected."
            )

            return

        self.result_text.insert(
            tk.END,
            "Detected Defects\n\n"
        )

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            self.result_text.insert(
                tk.END,
                f"{class_name:20} {confidence:.2%}\n"
            )

    def save_result(self):

        if self.result_image is None:

            messagebox.showwarning(
                "Warning",
                "Run detection first."
            )

            return

        save_path = filedialog.asksaveasfilename(

            defaultextension=".jpg",

            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png")
            ]
        )

        if not save_path:
            return

        cv2.imwrite(save_path, self.result_image)

        messagebox.showinfo(
            "Saved",
            "Result saved successfully."
        )

    def show_image(self, image):

        height, width = image.shape[:2]

        max_width = 850
        max_height = 450

        scale = min(
            max_width / width,
            max_height / height
        )

        width = int(width * scale)
        height = int(height * scale)

        image = cv2.resize(
            image,
            (width, height)
        )

        image = Image.fromarray(image)

        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo


def main():

    root = tk.Tk()

    PCBDetectorApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
