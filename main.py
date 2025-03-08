import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Загрузка CIFAR-100
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')

# Нормализация данных (масштабирование значений пикселей в диапазон [0, 1])
train_images, test_images = train_images / 255.0, test_images / 255.0

# Список названий классов CIFAR-100
cifar100_class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# Создаем словарь для быстрого доступа к названиям классов
class_names = {i: name for i, name in enumerate(cifar100_class_names)}

# Путь для сохранения модели
model_path = "cifar100_cnn_model.h5"

# Проверяем, существует ли уже обученная модель
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded from disk.")
    except Exception as e:
        print(f"Error loading model: {e}. Training a new model...")
        model = None
else:
    model = None

# Если модель не загружена, создаем новую
if model is None:
    # Определение архитектуры модели
    model = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.BatchNormalization(),  # Добавляем Batch Normalization для стабилизации обучения
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),

            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(100)  # Выходной слой для 100 классов
        ]
    )

    # Компиляция модели
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Коллбэки для мониторинга и сохранения модели
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor="val_accuracy", verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Обучение модели
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=64),
        epochs=50,
        validation_data=(test_images, test_labels),
        callbacks=[checkpoint, early_stopping]
    )

    # Data Augmentation для улучшения качества изображений
    datagen = ImageDataGenerator(
        rotation_range=15,  # Рандомное вращение изображений
        width_shift_range=0.1,  # Горизонтальный сдвиг
        height_shift_range=0.1,  # Вертикальный сдвиг
        horizontal_flip=True,  # Отражение по горизонтали
        zoom_range=0.1  # Масштабирование
    )
    datagen.fit(train_images)

    # Функция для классификации изображения
def classify_image(image):
    if image.shape != (32, 32, 3):
        raise ValueError("Input image must have shape (32, 32, 3).")
    img_array = tf.expand_dims(image, 0)  # Создание пакета размером 1
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names.get(predicted_class, "Unknown")

# Функция для визуализации нескольких изображений с их предсказаниями
def show_images_with_predictions(images, true_labels, n=10):
    fig, axes = plt.subplots(2, n // 2, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(n):
        predicted_label = classify_image(images[i])
        true_label = class_names.get(true_labels[i][0], "Unknown")
        axes[i].imshow(images[i])
        axes[i].set_title(f"Pred: {predicted_label}\nTrue: {true_label}", fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Пример использования: показываем первые 10 изображений тестового набора
show_images_with_predictions(test_images[:10], test_labels[:10])