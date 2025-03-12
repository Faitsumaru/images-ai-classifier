from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Загрузка CIFAR-100
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')

# Список меток классов для CIFAR-100
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# Функция для изменения размера изображений с помощью TensorFlow
def resize_images_tf(images, target_size=(64, 64)):
    images = tf.cast(images, tf.float32)
    resized_images = tf.image.resize(images, target_size, method='bicubic')
    return resized_images / 255.0  # Нормализация до [0, 1]

# Увеличение разрешения изображений до 64x64
train_images_resized = resize_images_tf(train_images)
test_images_resized = resize_images_tf(test_images)

# Определение архитектуры модели
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(100)
])

# Компиляция модели
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomHeight(0.1),
    layers.RandomWidth(0.1),
    layers.RandomFlip("horizontal")
])

# Создание обучающего датасета с аугментацией
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_resized, train_labels))
train_dataset = train_dataset.shuffle(1000).batch(64).map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images_resized, test_labels)).batch(64)

# Коллбэки
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cifar100_cnn_model.h5", save_best_only=True, monitor="val_accuracy", verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Обучение модели
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=[checkpoint, early_stopping]
)

# Функция классификации
def classify_image(image):
    if image.shape != (64, 64, 3):
        raise ValueError("Input image must have shape (64, 64, 3).")
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]

# Функция визуализации
def show_images_with_predictions(images, true_labels, n=10):
    fig, axes = plt.subplots(2, n//2, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(n):
        pred = classify_image(images[i])
        true = class_names[true_labels[i][0]]
        axes[i].imshow(images[i])
        axes[i].set_title(f"Prediction: {pred}\nTrue picture: {true}", fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Пример использования
show_images_with_predictions(test_images_resized[:10], test_labels[:10])