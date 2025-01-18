import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from keras import layers, models, regularizers

#Gaussian noise for data augmentation
class AddGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(AddGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, x, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.stddev)
            return x + noise
        return x
#-------------------------------------Image collection (To edit)
data_dir = "training images.tgz"
data_dir = pathlib.Path(data_dir).with_suffix('')
#-------------------------------------
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Create a dataset
batch_size = 32  # Reduced batch size due to smaller dataset
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# Performance tweaks
#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize pixel values to [0,1]
normalization_layer = layers.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
num_classes = len(class_names)
# Image Augmentation
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        AddGaussianNoise(0.05),
    ]
)

# Use MobileNetV2 as a feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False  # Freeze base model during training
l2_reg = regularizers.l2(0.05)
# Model architecture
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(8, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),

    #layers.Dropout(0.2),  

    layers.Flatten(),
    layers.Dense(147, activation='relu', kernel_regularizer=l2_reg), 
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


"""model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)  # Needed for global regularization

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=[l2_reg])  # Global regularization"""

model.summary()

# Train the model
epochs = 30  # Reduced due to the small dataset
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,  # Adjust the number of epochs as needed
    callbacks=[early_stopping]
)

# Plot training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['loss']))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Testing with a new image
sunflower_path = 'validation image\Screenshot 2024-11-24 164346.png'
img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Make prediction
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
