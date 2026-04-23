import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

plt.ion()
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка захвата видео")
        break

    small_frame = cv2.resize(frame, (640, 480))

    result = DeepFace.analyze(
        small_frame,
        actions=['emotion'],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    emotions = result.get('emotion', {})
    emotion = result.get('dominant_emotion', 'Белгісіз')

    print(f"Эмоция: {emotion}")

    cv2.putText(
        frame,
        emotion,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    ax.clear()
    ax.bar(emotions.keys(), emotions.values(), color='blue')
    ax.set_ylim([0, 100])
    ax.set_title("Распределение эмоций")
    ax.set_ylabel("Процент")
    ax.set_xlabel("Эмоции")

    plt.pause(0.1)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
