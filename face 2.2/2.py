import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split

DATASET_DIR = 'dataset'  # thư mục chứa ảnh đã chụp
IMG_SIZE = (64, 64)      # Kích thước chuẩn cho CNN

# Hàm đọc ảnh và nhãn
def load_data(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
    images = []
    labels = []
    for imagePath in imagePaths:
        # Chuyển ảnh sang grayscale và resize
        img = Image.open(imagePath).convert('L')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, 'float32') / 255.0  # chuẩn hóa [0,1]
        images.append(img_array)

        # Lấy ID từ tên file User.<id>.<count>.jpg
        filename = os.path.split(imagePath)[-1]
        id = int(filename.split('.')[1])
        labels.append(id)

    return np.array(images), np.array(labels)

# Load dữ liệu
images, labels = load_data(DATASET_DIR)

# Chỉnh nhãn về khoảng 0..(NUM_CLASSES-1) để phù hợp sparse_categorical_crossentropy
labels = labels - 1

# Đưa ảnh về dạng (số ảnh, height, width, channels)
images = np.expand_dims(images, axis=-1)

# Lấy số lượng class
NUM_CLASSES = len(np.unique(labels))

# Chia train/test
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Xây dựng mô hình CNN với Input layer để tránh warning
model = models.Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  # đầu ra: số lượng người trong dataset
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# Lưu mô hình
MODEL_PATH = 'trainer/cnn_face_model.h5'
os.makedirs('trainer', exist_ok=True)
model.save(MODEL_PATH)
print(f"\n Hoàn tất. Đã train {len(np.unique(labels))} khuôn mặt.")
print(f"\nHoàn tất. Mô hình CNN đã lưu tại {MODEL_PATH}")

