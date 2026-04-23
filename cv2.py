from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import easyocr
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return image, gray

def recognize_plate(image_path):
    image, gray = preprocess_image(image_path)
    results = reader.readtext(gray, detail=1)
    plates = []

    for (bbox, text, prob) in results:
        if prob > 0.6:
            plates.append(text)

            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    processed_filename = 'processed_' + os.path.basename(image_path)

    if not processed_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        processed_filename += '.jpg'

    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, image)

    return plates, processed_path

def recognize_plate_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Видео ашылмады!")
        return []

    frame_skip = 5
    frame_count = 0
    plates_detected = set()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray, detail=1)

            for (bbox, text, prob) in results:
                if prob > 0.7:
                    plates_detected.add(text)

                    (top_left, _, bottom_right, _) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))

                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        text,
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

        frame_count += 1

    cap.release()
    return list(plates_detected)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл табылмады!'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Файл таңдалмады!'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        plates = recognize_plate_from_video(filepath)
        return jsonify({'plates': plates, 'video_path': filepath})

    plates, processed_path = recognize_plate(filepath)
    return jsonify({'plates': plates, 'image_path': processed_path})

@app.route('/camera')
def camera_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return 'Камера ашылмады!'

    plates_detected = set()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1)

        for (bbox, text, prob) in results:
            if prob > 0.7:
                plates_detected.add(text)

                (top_left, _, bottom_right, _) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Камерадан нөмір тану - 'q' басу үшін", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return f"Танылған нөмірлер: {', '.join(plates_detected) if plates_detected else 'табылмады'}"

if __name__ == '__main__':
    app.run(debug=True)
