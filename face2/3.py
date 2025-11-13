import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

MODEL_PATH = 'trainer/cnn_face_model.h5'  # model CNN đã train
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (64, 64)

# Load model CNN
model = load_model(MODEL_PATH)

# Tạo bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Danh sách tên theo ID (phải khớp với file dataset)
names = {
    1: "cr7",
    2: "minh chim be",
    3: "thien",
    4: 'vy'
}

# Khởi tạo camera
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

print("\nBắt đầu nhận diện. Nhấn ESC để thoát.")

while True:
    ret, img = cam.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    num_face = len(faces)
    cv2.putText(img, f'Detected: {num_face}', (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = Image.fromarray(face_img).resize(IMG_SIZE)
        face_array = np.array(face_img, 'float32') / 255.0
        face_array = np.expand_dims(face_array, axis=(0, -1))  # shape: (1,64,64,1)

        # Dự đoán
        predictions = model.predict(face_array)
        id_ = np.argmax(predictions)
        confidence = predictions[0][id_] * 100  # xác suất %
        
        if confidence > 50:  # ngưỡng nhận diện
            name = names.get(id_ + 1, "Unknown")  # +1 nếu id bắt đầu từ 1
        else:
            name = "Unknown"

        # Vẽ khung và thông tin
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{name}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Acc: {round(confidence)}%", (x + 5, y + h + 25), font, 0.7, (255, 255, 0), 1)

    cv2.imshow("CNN Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\nThoát chương trình.")
cam.release()
cv2.destroyAllWindows()
