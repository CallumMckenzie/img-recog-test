import csv
from FeatureExtractor import EdgeDetection
import os

directory = "training images"
with open('profiles1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["label", "number of notches", "average notch depth", "image shape"]
    writer.writerow(field)
    for subdir in os.listdir(directory):
        subdir = os.path.join(directory, subdir)
        for filename in os.listdir(subdir):
            label = filename
            print(filename)
            path = os.path.join(subdir, filename)
            features = EdgeDetection(path)
            print(features)
            #writer.writerow([label, notchnum, notchdepth, features[1]])

   

