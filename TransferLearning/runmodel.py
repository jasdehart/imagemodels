#!usr/bin/python3

    #to run: python3 remaketwo.py -d ./testing -m ./TransferLearning/testmodel.pt
    #You can put full paths or abbreviated if deriving from the same directory.
from __future__ import print_function, division

import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
from PIL import Image

'''
def main() calls getfiles() function and passes in directory from command line which
is set in if, name, main section
Since paths is what we need to send to running_model_prediction(), I set
the getfiles() = to paths because it returns the information we need.
paths can then be sent to the next function. In running_model_predictions(),
we send model because predict_images() needs that information but we
didn't call it in main, so the needed information is forwarded.
'''

def main(directory, model):
    paths = getfiles(directory)
    running_model_predictions(paths, model)


def getfiles(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(filename)
            fileList.append(str(filename))
        if filename.endswith(".gif"):
            print(filename)
            fileList.append(str(filename))
        if filename.endswith(".png"):
            sys.stderr.write("ERROR: Classifier only takes .jpg and .gif files\n")
        if filename.endswith(".jpeg"):
            sys.stderr.write("ERROR: Classifier only takes .jpg and .gif files\n")
    for filename in list(fileList):
        fn = directory + '/' + filename
        fullPathList.append(str(fn))

    return fullPathList


#prediction function
def predict_image(image, model):
        #boundaries for transforming the image
    test_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data = test_transforms(image)#transforms
    data = data.unsqueeze_(0)
    output = model(data) #predicting, outputs array
    prediction = int(torch.max(output.data, 1)[1].numpy()) #predicted number from category
    return prediction

#running predictions with files
def running_model_predictions(path, model):
    for filename in list(path):
        X = Image.open(filename)
        print('***********************************************************')
        print("Current file: " + filename)
        index = predict_image(X, model)
        print('Predicting.....')
        #converts index number to class name
        if (index== 0):
            index ='prediction: baby'
        if (index == 1):
            index = 'prediction: cards'
        if (index == 2):
            index = 'prediction: key'
        if (index == 3):
            index = 'prediction: license'
        if (index == 4):
            index = 'prediction: passport'

        print (index)



if __name__ == "__main__":
    #getting our information from command line, ALL are REQUIRED
    parser = argparse.ArgumentParser(description='Script for taking arguments')
    parser.add_argument('-d', '--directory',
                        help='directory for images',
                        required='True') #gets directory for images
    parser.add_argument('-m', '--model',
                        help='choose model you want to use',
                        required = 'True') #allows us to choose any model
    args = parser.parse_args()

    #if the argument for directory is input do this...
    if args.directory:
        directory = args.directory
        fileList = list() #store fileNames
        fullPathList = list() #store full filename paths
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #if the argument for model is input do this...
    if args.model:
        model = args.model
        model = torch.load(model)
        model.eval() #needed for normalization

    #calling and running main functions with needed inputs for other arguments
    main(directory, model)
