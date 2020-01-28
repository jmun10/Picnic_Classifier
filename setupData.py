# runs first IF converting to jpeg

#import Convert_to_jpeg
#import OrganizeTrainImages
#import SortImages
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pathlib
from numpy import loadtxt
import random

IMG_SIZE = 64

localpath = r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"

types = []

with open("types.txt","r") as text:
        for line in text:
            currentPlace = line[:-1]
            types.append(currentPlace)
types[24] = "Pineapples, melons & passion fruit"
#print(types)


# trying out fashion mnset tutorial for picnic classifier
# first reformat images to 100 x 100
for type in types:
    path = os.path.join(localpath, type)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        #plt.imshow(img_array)
        #plt.show()
        break
    break
print("done reformatting images")

training_data = []


def createTrainingData():
    brokenFiles = 0
    for type in types:
        path = os.path.join(localpath, type)
        class_num = types.index(type)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                newArray = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([newArray,class_num])
            except Exception as e:
                brokenFiles+=1
                pass
    print("Amount of Broken Files: " + str(brokenFiles))


createTrainingData()
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

np.save("features.npy",X)
np.save("labels.npy",y)

print("all done")