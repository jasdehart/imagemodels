#!usr/bin/python3
'''
to run: python3 remaketwo.py -d ./testing -m ./TransferLearning/testmodel.pt
You can put full paths or abbreviated if deriving from the same directory.
'''

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


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1 * 1 * 64, 100)
        self.fc2 = nn.Linear(100,64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = MyModel()




def getfiles(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(filename)
            fileList.append(str(filename))
        if filename.endswith(".gif"):
            print(filename)
            fileList.append(str(filename))
        if filename.endswith(".png"):
            sys.stderr.write("ERROR: Classifier only takes .jpg and .gif files")
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
        # original saved file with DataParallel
        checkpoint = torch.load('/Users/lisaegede/Documents/GitHub/imagemodels/TransferLearning/conv_net_model.ckpt')
        """
        import re
        pattern = re.compile(r'(?:layer1.0.weight|layer1.0.bias|layer2.0.weight|layer2.0.bias|fc1.weight|fc1.bias|fc2.weight|fc2.bias)\.?')
        state_dict = checkpoint
        for key, value in list(checkpoint.items()):
            res = pattern.match(key)
            if res:
                #new_key = res.group(1) + res.group(2)
                #state_dict[new_key] = state_dict[key]
                del checkpoint[key]
        torch.save(checkpoint, './newcheckpoint.ckpt')
        """
        model.load_state_dict(checkpoint)
        model.eval()



        #print(model)
        #checkpoint = torch.load('/home/leanna/Documents/Research/imagemodels/TransferLearning/conv_net_model.ckpt')
        #from pdb import set_trace; set_trace()
        #model.load_state_dict(checkpoint)



        #model.eval() #needed for normalization
        #model.load_state_dict(torch.load(model))
        #checkpoint.eval()



    #calling and running main functions with needed inputs for other arguments

    paths = getfiles(directory)
    running_model_predictions(paths, model)
