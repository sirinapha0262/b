# train.py - Updated for Eye3 Dataset (10 classes)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import albumentations as A
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

# --- ตั้งค่าใช้ GPU (ถ้ามี) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU not found, using CPU")

# --- สร้างโมเดลใหม่สำหรับ 10 คลาส ---
def create_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Augmentation pipeline ด้วย Albumentations ---
transform = A.Compose([
    A.Resize(128, 128),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.1),
])

# ฟังก์ชันการโหลดภาพ
def load_image(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            return np.zeros((128, 128, 3))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        image = augmented['image']
        image = image / 255.0  # normalize
        return image.astype(np.float32)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return np.zeros((128, 128, 3))

# ฟังก์ชันโหลดชุดข้อมูลจากโฟลเดอร์ Eye3
def load_dataset(data_dir):
    # คลาสทั้ง 10 คลาสจากรูปที่แสดง
    expected_classes = [
        'Central Serous Chorioretinopathy',
        'Diabetic Retinopathy', 
        'Disc Edema',
        'Glaucoma',
        'Healthy',
        'Macular Scar',
        'Myopia',
        'Pterygium',
        'Retinal Detachment',
        'Retinitis Pigmentosa'
    ]
    
    class_names = []
    file_paths = []
    labels = []
    
    # ตรวจสอบว่ามีโฟลเดอร์ train หรือไม่
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        data_dir = train_dir
    
    # สแกนโฟลเดอร์ทั้งหมด
    for item in os.listdir(data_dir):
        class_path = os.path.join(data_dir, item)
        if os.path.isdir(class_path):
            class_names.append(item)
    
    class_names = sorted(class_names)
    print(f"Found {len(class_names)} classes: {class_names}")
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        image_count = 0
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_paths.append(os.path.join(class_path, fname))
                labels.append(idx)
                image_count += 1
        print(f"Class '{class_name}': {image_count} images")
    
    return file_paths, labels, class_names

# กำหนดเส้นทางข้อมูลใหม่
data_directory = 'Eye3'  # เปลี่ยนจาก 'Eye Disease.v3i.folder' เป็น 'Eye3'
file_paths, labels, class_names = load_dataset(data_directory)

print(f"Total images: {len(file_paths)}")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# แบ่งข้อมูลเป็น train/val โดยประมาณ 80/20
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42)

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# ฟังก์ชัน data generator
def data_generator(file_paths, labels, batch_size=32, shuffle=True):
    dataset_size = len(file_paths)
    indices = np.arange(dataset_size)
    num_classes = len(class_names)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]
            images = [load_image(file_paths[i]) for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]
            yield np.array(images), tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)

batch_size = 32
train_gen = data_generator(train_paths, train_labels, batch_size=batch_size)
val_gen = data_generator(val_paths, val_labels, batch_size=batch_size, shuffle=False)

steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

# --- สร้างโมเดลใหม่ ---
num_classes = len(class_names)
model = create_model(num_classes=num_classes)

# --- คอมไพล์โมเดล ---
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # เริ่มด้วย learning rate ที่สูงกว่าเล็กน้อย
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# แสดงสถาปัตยกรรมของโมเดล
model.summary()

# --- Callbacks ---
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

checkpoint = ModelCheckpoint(
    'EyeModel_v3_best.h5', 
    monitor='val_accuracy',  # เปลี่ยนเป็น val_accuracy
    save_best_only=True, 
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# --- เริ่มเทรน ---
print("Starting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=50,  # เพิ่ม epochs เพราะเป็นโมเดลใหม่
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# --- บันทึกโมเดลที่เทรนเสร็จแล้ว ---
model.save('EyeModel_v3.h5')
print("Saved trained model as EyeModel_v3.h5")

# บันทึกรายชื่อคลาส
import json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Saved class names to class_names.json")

# แสดงผลการเทรน
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Training completed!")