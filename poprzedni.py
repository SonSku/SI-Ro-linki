import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# Ustawienie ziaren losowości dla powtarzalnych wyników
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

print("GPU dostępne:", tf.config.list_physical_devices('GPU'))

dataset_path = r"C:\house_plant_species"

# Parametry
img_size = (224, 224)
batch_size = 64
validation_split = 0.2

# Wczytanie danych z podziałem
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Klasy:", class_names, "\nLiczba klas:", num_classes)

# Augmentacja i normalizacja
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

normalization_layer = tf.keras.layers.Rescaling(1. / 255)


def prepare_ds(ds, training=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x), y))
    return ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


train_ds = prepare_ds(train_ds, training=True)
val_ds = prepare_ds(val_ds)

# Architektura modelu z Dropout i BatchNormalization
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Kompilacja
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacki
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("najlepszy_model.keras", save_best_only=True)
]

# Trening
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks
)

# Zapis ostatecznego modelu
model.save("model_rosliny_final.keras")

# Wizualizacja
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Strata')
plt.legend()

plt.show()