import cv2
import os
import numpy as np

TRAINER_PATH = 'trainer/trainer.yml'   # model đã train ở file 2
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"



#KHỞI TẠO 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

#NHẬP DANH SÁCH TÊN THEO ID

names = {
    1: "Thu",
    2: "Thoa",
    3: "MTP" }

#tạo cam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    num_face=len(faces)
    text=f'detected {num_face}'
    for (x, y, w, h) in faces:
        # Nhận diện khuôn mặt
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Xác định người và độ tin cậy
        if confidence < 90:
            name = names.get(id_, "Unknown")
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        # Vẽ khung và hiển thị thông tin
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{name}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Acc: {confidence_text}", (x + 5, y + h + 25), font, 0.7, (255, 255, 0), 1)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Face Recognition", img)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27: #0xFF=nút ecs
        break

print("\nThoát chương trình.")
cam.release()
cv2.destroyAllWindows()
