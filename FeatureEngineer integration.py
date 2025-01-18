import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import layers
#from featureEngineer import ExtractFeatureTensor
from FeatureExtractor import ExtractAggregate
from PIL import Image
from img2vec_pytorch import Img2Vec
img2vec = Img2Vec()
# Function to preprocess images and extract features
def preprocess_images(image_dir):
    features_list = []
    labels_list = []
    class_mapping = {}  # Map class names to integers
    class_counter = 0
    
    # Loop through images in the directory
    for class_label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, class_label)
        
        if os.path.isdir(class_dir):
            if class_label not in class_mapping:
                class_mapping[class_label] = class_counter
                class_counter += 1
                
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                
                # Read and process image
                #image = cv2.imread(img_path)
                #if image is None:
                    #continue
                #img = Image.open(img_path).convert("RGB")
                #features = img2vec.get_vec(img)
                # Extract features for the image
                #features = ExtractFeatureTensor(image)
                features = ExtractAggregate(img_path)
                features_list.append(features)
                print (features)
                print(img_path)
                
                # Append corresponding numerical label
                labels_list.append(class_mapping[class_label])
               

    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)  # Ensure features are float32
    labels_array = np.array(labels_list, dtype=np.int32)       # Ensure labels are integers
    
    return features_array, labels_array, len(class_mapping)

# Path to the image directory
image_dir = r"C:\Users\callu\OneDrive\Desktop\programming\img recog test\training images"

# Preprocess images
features_array, labels, num_classes = preprocess_images(image_dir)

# Define the model
model = tf.keras.Sequential([
    layers.Input(shape=(features_array.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    features_array, 
    labels, 
    epochs=200, 
    batch_size=32, 
    validation_split=0.2
)

# Visualization of training results
def visualize_results(history):
    # Extract training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_results(history)
