#imports
import numpy as np
import tensorflow as tf
import cv2

def Cropper(path):
    """This function removes the background from the image to allow for either the model
    to more easily process the image or the edge detection function to more easily detect the edges"""
    from PIL import Image
    from rembg import remove
    input = Image.open(path)
    CroppedImg = remove(input) #removes the background from the image
    output_array = np.array(CroppedImg) #converts the image to a numpy array for use in cv2
    if __name__ == "__main__": cv2.imshow("test",output_array)
    return output_array

def ExtractNotches(path):
    """Returns an integer value representing the number of notches that exist on the whale's tail"""
    #generates the edges 
    edges, img = GetEdges(path)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notchNum = 0
    notchDepths = []
    for contour in contours:
        length = cv2.arcLength(contour, closed=True)
        if length > 100: #checks the length of the arc lines
            area = cv2.contourArea(contour)
            if area > 10 and area < 400:  # Adjust thresholds based on image size
                notchNum += 1
                x, y, w, h = cv2.boundingRect(contour)
                depth = h  # Depth of the notch based on bounding box height
                notchDepths.append(depth)


    #for testing
    if __name__ == "__main__":
        print(f"Number of notches: {notchNum}")
        print(f"Notch depths: {notchDepths}")

    return notchNum,notchDepths
    
def ExtractShape(path):
    #generates the edges and turns them into a tensor
    edges, img = GetEdges(path)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # Adjust kernel size as needed
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) #removes edges found in water

    #edges_tensor = tf.convert_to_tensor(edges, dtype=tf.float32) 
    resized_features = cv2.resize(edges_cleaned, (360,435), interpolation=cv2.INTER_AREA)
    edges_tensor =  tf.convert_to_tensor(resized_features, dtype=tf.float32)

    #displays the edges over the image for testing
    if __name__ == "__main__":
        #print(resized_features.shape)  # Output: tensor shape (height, width, 1)
        #print(resized_features)
        #img[edges_cleaned == 255] = (255, 0, 0)
        #cv2.imshow('Edge Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (edges_tensor)

def ExtractAggregate(path):
    """Creates a 1D tensor containing the number of notches, 
    their average depth and the shape of the whale tail."""

    # Extract features
    notchNum, notchDepths = ExtractNotches(path)
    notchNumTensor = tf.convert_to_tensor(notchNum, dtype=tf.float32)  # Scalar tensor (0D)
    notchDepthsTensor = tf.convert_to_tensor(notchDepths, dtype=tf.float32)  # 1D tensor
    image_tensor = ExtractShape(path)  # 2D tensor

    # Process tensors
    image_flattened = tf.reshape(image_tensor, [-1])  # Flatten the 2D image tensor
    notchNumTensor_expanded = tf.reshape(notchNumTensor, [1])  # Convert 0D tensor to 1D

    # Create tensor list
    tensor_list = [image_flattened, notchNumTensor_expanded, notchDepthsTensor]

    # Find the maximum length among all tensors
    max_length = max(tensor.shape[0] for tensor in tensor_list)

    # Pad tensors to the same length
    padded_tensors = [tf.pad(tensor, [[0, max_length - tensor.shape[0]]]) for tensor in tensor_list]

    # Concatenate tensors into a single 1D tensor
    aggregated_features = tf.concat(padded_tensors, axis=0)
    return aggregated_features


#internal function to simplify code
def GetEdges(path):
    """Extracts the outline of the tail"""

    #img = cv2.imread(path)
    img = Cropper(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 300) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #testing
    try:
        contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = edges.copy() * 0
        print(contours)
        for contour in contours:
            length = cv2.arcLength(contour, closed=True)
            area = cv2.contourArea(contour)
            if length > 100 and area > 50:  # Adjust the length threshold
                cv2.drawContours(filtered_edges, [contour], -1, (255, 255, 255), 1)
            connected_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, kernel)
        assert 'connected_edges' in locals() == True #checks whether the tail has any edges
    except AssertionError:
        print("The tail had no detectable edges")
    cv2.imshow("Filtered Edges", filtered_edges)
    cv2.imshow("connected edges", connected_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return connected_edges, img


#Code for testing the module
if __name__ == "__main__":
    #path = r"validation image\BS-62 2021 (1).png"
    path = r"contrast_enhanced.png"
    #ExtractShape(path) #Returns the edges as a tensor
    #ExtractNotches(path)
    #GetEdges(path) #gets the image outline for other code pieces
    print(ExtractAggregate(path))



