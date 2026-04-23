import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

input_image_path = "images/img.png"
output_image_path = "images/output_objects.jpg"

image = cv2.imread(input_image_path)

if image is None:
    print(f"Ошибка: Не удалось загрузить изображение {input_image_path}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

models = ["Shoe", "Chair", "Cup", "Camera"]

detected_objects = []

for model_name in models:
    with mp_objectron.Objectron(
        static_image_mode=True,
        max_num_objects=5,
        model_name=model_name
    ) as objectron:

        results = objectron.process(image_rgb)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                detected_objects.append(model_name)

                mp_drawing.draw_landmarks(
                    image,
                    detected_object.landmarks_2d,
                    mp_objectron.BOX_CONNECTIONS
                )

                x = int(detected_object.landmarks_2d.landmark[0].x * image.shape[1])
                y = int(detected_object.landmarks_2d.landmark[0].y * image.shape[0])

                cv2.putText(
                    image,
                    model_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

if detected_objects:
    print("Найденные объекты:")
    for obj in detected_objects:
        print(obj)
else:
    print("Объекты не найдены")

cv2.imwrite(output_image_path, image)
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
