import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def segment_tail_by_color(image):
    """Segment the whale tail based on its color (black) compared to the blue background."""
    # Convert the image to HSV color space using TensorFlow
    image_hsv = tf.image.rgb_to_hsv(image)

    # Debugging: Print or visualize the HSV image
    #print("HSV Image Range:")
    #print("Min HSV:", tf.reduce_min(image_hsv, axis=(0, 1)).numpy())
    #print("Max HSV:", tf.reduce_max(image_hsv, axis=(0, 1)).numpy())

    # Define color thresholds for the black tail in HSV space
    lower_black = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    upper_black = tf.constant([1.0, 1.0, 0.4], dtype=tf.float32)  # Adjusted thresholds

    # Create a mask for the black regions using TensorFlow's logical operations
    mask = tf.logical_and(
        tf.reduce_all(image_hsv >= lower_black, axis=-1),
        tf.reduce_all(image_hsv <= upper_black, axis=-1)
    )

    return tf.cast(mask, tf.float32)


def extract_tail_features(mask):
    features = []

    # 1. Number of non-zero pixels in the mask
    non_zero_pixels = tf.reduce_sum(mask)
    features.append(non_zero_pixels)

    # 2. Aspect ratio
    non_zero_indices = tf.where(mask > 0)
    min_x = tf.cast(tf.reduce_min(non_zero_indices[:, 1]), tf.int32)
    max_x = tf.cast(tf.reduce_max(non_zero_indices[:, 1]), tf.int32)
    min_y = tf.cast(tf.reduce_min(non_zero_indices[:, 0]), tf.int32)
    max_y = tf.cast(tf.reduce_max(non_zero_indices[:, 0]), tf.int32)

    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = tf.where(height > 0, tf.cast(width, tf.float32) / tf.cast(height, tf.float32), 0.0)
    features.append(aspect_ratio)

    # 3. Symmetry score
    total_width = tf.shape(mask)[1]
    half_width = tf.cast(total_width // 2, tf.int32)
    right_half = mask[min_y:max_y, min_x + half_width:max_x]
    left_half = mask[min_y:max_y, min_x:min_x + half_width]

    left_width = tf.shape(left_half)[1]
    right_width = tf.shape(right_half)[1]
    adjusted_right_half = tf.cond(
        tf.less(right_width, left_width),
        lambda: tf.pad(right_half, [[0, 0], [0, left_width - right_width]]),
        lambda: right_half[:, :left_width]
    )
    right_half_flipped = tf.reverse(adjusted_right_half, axis=[1])
    symmetry = tf.reduce_mean(tf.abs(tf.cast(left_half, tf.float32) - tf.cast(right_half_flipped, tf.float32)))
    features.append(symmetry)

    # 4. Notch count (with cleaned mask)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask.numpy().astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(cleaned_mask, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(edges)
    notch_count = 0
    if len(contours) > 0:
        hull = cv2.convexHull(contours[0], returnPoints=False)
        defects = cv2.convexityDefects(contours[0], hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                start_idx, end_idx, farthest_idx, depth = defects[i, 0]
                depth = depth / 256.0
                if depth > 10.0:  # Stricter threshold
                    notch_count += 1
    features.append(notch_count)

    return features

    


def ExtractFeatureTensor(image):
    # Check if the image is already a NumPy array
    if isinstance(image, np.ndarray):
        # If it's a NumPy array (from OpenCV), we directly process it
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # If it's a TensorFlow tensor, we convert it to a NumPy array
        image_rgb = image.numpy() if tf.executing_eagerly() else tf.image.convert_image_dtype(image, dtype=tf.uint8).numpy()

    # Resize the image to a fixed size for consistent processing
    img_height, img_width = 180, 180
    image_rgb = cv2.resize(image_rgb, (img_width, img_height))

    # Normalize pixel values to [0, 1] for TensorFlow compatibility
    image_tensor = tf.convert_to_tensor(image_rgb / 255.0, dtype=tf.float32)

    # Call the segmentation function to generate the mask
    mask = segment_tail_by_color(image_tensor)

    # Call the feature extraction function using the mask
    features = extract_tail_features(mask)

    # The mask is for testing tail identification, not for use outside this code
    if __name__ != "__main__":
        return features
    else:
        return mask, features


if __name__ == "__main__":
    # Path to the validation image
    image_path = r"C:\Users\callu\OneDrive\Desktop\programming\img recog test\validation image\Screenshot 2024-11-24 164346.png"

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} not found or could not be loaded.")
        exit()

    # Call ExtractFeatureTensor function
    mask, features = ExtractFeatureTensor(image)

    # 1 the mask and features
    print("Mask shape:", mask.shape)
    print("Extracted Features:", features)

    # Display the mask using matplotlib
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title("Tail Mask")
    plt.imshow(mask.numpy(), cmap="gray")

    plt.show()
