from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from keras import layers, models, regularizers

#-------------------------------------Image collection (To edit)
data_dir = "training images.tgz"
data_dir = pathlib.Path(data_dir).with_suffix('')
#-------------------------------------
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

# Create a dataset
batch_size = 32 # Reduced batch size due to smaller dataset
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# Performance tweaks
#AUTOTUNE = tf.data.AUTOTUNE
#uncomment these lines when model is performing, These lines are potentially causing large degree of variance between runs
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize pixel values to [0,1]
normalization_layer = layers.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# No data augmentation needed for VGG16 (optional)

# Use VGG16 as a feature extractor
base_model = tf.keras.applications.VGG16(input_shape=(img_height, img_width, 3),
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False  # Freeze base model during training

#for layer in base_model.layers[-4:]:  # Adjust the number of layers to unfreeze
    #layer.trainable = True


l2_reg = regularizers.l2(0.01)
# Model architecture
model = models.Sequential([
  layers.Rescaling(1./255),  # Include normalization here
  base_model,
  layers.Flatten(),
  #layers.Dropout(0.1),
  #layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
  #layers.Dense(1024, activation='relu', kernel_regularizer=l2_reg),
  #layers.Dropout(0.1),
  layers.Dense(294, activation='relu', kernel_regularizer=l2_reg), 
  layers.Dropout(0.1),
  layers.Dense(147, activation='relu', kernel_regularizer=l2_reg), # #must be larger than the number of classes: 147
  layers.Dropout(0.1),
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), #previously 3
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


model.summary()

# Train the model
epochs = 30  # Reduced due to the small dataset
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=batch_size,
  epochs=epochs,  # Adjust the number of epochs as needed
  callbacks=[early_stopping]
)
#print(len(history.history['loss']))
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

