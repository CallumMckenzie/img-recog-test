from sklearn.model_selection import KFold
import tensorflow as tf
from keras import layers, models, regularizers
import numpy as np
import os
from pathlib import Path

# Parameters
k_folds = 5
batch_size = 32
img_height = 180
img_width = 180
l2_reg = regularizers.l2(0.02)
epochs = 20  # Set appropriate epochs for k-fold
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Load dataset
data_dir = Path("training images")
all_image_paths = list(data_dir.glob('*/*.png'))
all_labels = [p.parent.name for p in all_image_paths]
class_names = sorted(set(all_labels))
label_to_index = {name: i for i, name in enumerate(class_names)}
all_labels = [label_to_index[label] for label in all_labels]

# Shuffle data
np.random.seed(123)
shuffled_indices = np.random.permutation(len(all_image_paths))
all_image_paths = np.array(all_image_paths)[shuffled_indices]
all_labels = np.array(all_labels)[shuffled_indices]

# Define KFold
kf = KFold(n_splits=k_folds, shuffle=True, random_state=123)

fold_accuracies = []
fold_losses = []

for fold, (train_indices, val_indices) in enumerate(kf.split(all_image_paths)):
    print(f"\n### Training Fold {fold + 1}/{k_folds} ###")

    # Split data into training and validation
    train_paths = all_image_paths[train_indices]
    train_labels = all_labels[train_indices]
    val_paths = all_image_paths[val_indices]
    val_labels = all_labels[val_indices]

    # Create datasets
    def preprocess_image(file_path, label):
        img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array, label
    train_paths = [str(path) for path in train_paths]
    val_paths = [str(path) for path in val_paths]  # Do this for validation paths too

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(lambda x, y: tf.py_function(preprocess_image, [x, y], [tf.float32, tf.int64]))
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(lambda x, y: tf.py_function(preprocess_image, [x, y], [tf.float32, tf.int64]))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create model
    base_model = tf.keras.applications.VGG16(input_shape=(img_height, img_width, 3),
                                             include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze VGG16 layers

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model on validation data
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Fold {fold + 1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    fold_losses.append(val_loss)
    fold_accuracies.append(val_acc)

# Final Cross-Validation Results
print(f"\n### Cross-Validation Results ###")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Average Loss: {np.mean(fold_losses):.4f}")
