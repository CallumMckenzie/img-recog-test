import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import torch
import sys
import os

using_colab = False

#Prerequisite functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  




def Cropper(path):
    def runPredict(input_point):
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        masks.shape
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()  
        
        return masks
    
    
    #image check
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    """

    #loading the SAM model
    sys.path.append("..")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[375, 325]])
    input_label = np.array([1])
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()  
    masks = runPredict(input_point)

    
    #lets the user select between the multiple masks
    TextLock = 1
    while TextLock == 1:
        choice = input("do you want to use mask 1, 2 or 3 or change the pointer 4\n")
        if int(choice) == 1:
            SelectedMask = masks[0]
            break
        elif int(choice) == 2:
            SelectedMask = masks[1]
            break
        elif int(choice) == 3:
            SelectedMask = masks[2]
            break
        elif int(choice) == 4:
            x = int(input("enter first coord\n"))
            y = int(input("enter second coord\n"))
            input_point = np.array([[x, y]])
            masks = runPredict(input_point)
        else:
            print("invalid input")

    # Convert the mask to uint8 for OpenCV operations
    SelectedMask_uint8 = (SelectedMask * 255).astype(np.uint8)

    # Apply the mask to the image (retain only the overlay region)
    masked_image = cv2.bitwise_and(image, image, mask=SelectedMask_uint8)

    # Save or display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_image)
    plt.axis('off')
    plt.title("Image with Mask Applied")
    plt.show()
    return masked_image


main_new_folder = r"training images(new)"
main_old_folder = r"training images"
for sub_folder in os.listdir(main_old_folder):
    sub_folder_path = os.path.join(main_old_folder, sub_folder) #joins so that the next line's path is valid
    for img_file in os.listdir(sub_folder_path):
        #print(img_file)
        img_file_path = os.path.join(sub_folder_path, img_file)
        #print(img_file_path)
        
        newpath = os.path.join(main_new_folder, sub_folder) 
        #print(newpath)
        
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        output_path = os.path.join(newpath, img_file) #creates the path for the new image
        

        #Only processes unprocessed images
        checker = 0
        for new_sub_folder in os.listdir(main_new_folder):
            new_sub_folder = os.path.join(main_new_folder, new_sub_folder) #joins so that the next line's path is valid
            for new_img_file in os.listdir(new_sub_folder):
                if new_img_file == img_file:
                    checker = 1
                

        if checker == 0:
            processed_image = Cropper(img_file_path) #processes the image
            print(newpath)
            print(output_path)
            cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)) #saves the new file

        
       

        
