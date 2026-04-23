import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

image_folder = r"C:\Users\WEB\Desktop\train_v3\train"
excel_file = r"C:\Users\WEB\Desktop\recognized_names.xlsx"

model = load_model("ocr_model.h5")

image_size = (128, 128)

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Filename", "Recognized Name", "Status"])

files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

print(f"Найдено файлов: {len(files)}")

for filename in files:
    img_path = os.path.join(image_folder, filename)
    print(f"Обрабатываем файл: {filename}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Ошибка чтения файла: {filename}")
        df = pd.concat([df, pd.DataFrame([{
            "Filename": filename,
            "Recognized Name": "ERROR",
            "Status": "File Not Readable"
        }])], ignore_index=True)
        continue

    img = cv2.resize(img, image_size)
    img = img / 255.0
    img = img.reshape(1, image_size[0], image_size[1], 1)

    prediction = model.predict(img)
    recognized_text = str(np.argmax(prediction))

    print(f"Распознанный текст: {recognized_text}")

    if not recognized_text or any(char.isdigit() for char in recognized_text):
        print(f"Неверный формат, пропускаем: {recognized_text}")
        continue

    df = pd.concat([df, pd.DataFrame([{
        "Filename": filename,
        "Recognized Name": recognized_text,
        "Status": "OK"
    }])], ignore_index=True)

print("Сохраняем в Excel...")

with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
    df.to_excel(writer, index=False)

print(f"Данные сохранены в {excel_file}")
