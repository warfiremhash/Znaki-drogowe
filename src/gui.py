import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def predict_image():
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_label = np.argmax(prediction)
    result_label.config(text=f"Przewidywana klasa: {class_label}")

model = load_model('model/traffic_sign_model.h5')

root = tk.Tk()
root.title("Rozpoznawanie znaków drogowych")

upload_button = tk.Button(root, text="Wczytaj obraz", command=predict_image)
upload_button.pack()

result_label = tk.Label(root, text="Przewidywana klasa:")
result_label.pack()

root.mainloop()
