import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model/traffic_sign_model.h5')

class_mapping = {
    0: "A1", 1: "A17", 2: "A2", 3: "A21", 4: "A30",
    5: "A7", 6: "B1", 7: "B2", 8: "B20", 9: "B21",
    10: "B22", 11: "B23", 12: "B33", 13: "B36", 14: "B41",
    15: "C12", 16: "C2", 17: "C4", 18: "D1", 19: "D6",
    20: "Inny"
}

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        class_label = class_mapping[class_index]
        confidence = np.max(predictions)

        result_label.config(text=f"Znak: {class_label}, Pewność: {confidence:.2f}")

        image = Image.open(file_path).resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

root = tk.Tk()
root.title("Rozpoznawanie znaków drogowych")

frame = tk.Frame(root)
frame.pack(pady=20)

upload_button = tk.Button(frame, text="Wczytaj obraz", command=predict_image, font=("Arial", 14))
upload_button.pack(pady=10)

result_label = tk.Label(frame, text="Wynik: ", font=("Arial", 14))
result_label.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack(pady=10)

root.mainloop()
