import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
import time
import gzip
import random
import time
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")

path = 'explained_images'
filenames = glob.glob(path + "/*.pickle")
explain_images = []
for filename in filenames:
    image = pickle.load( open(filename, "rb") )
    image = np.array(image[0])#, dtype="float64")
    np.append(image, image[1])
    explain_images.append(image)
    #training.append(np.array(data[0]))

class CustomTensorDataset(Dataset):
    
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        image = torch.tensor(self.tensors[0][index],dtype=torch.float32)

        y = self.tensors[1][index]

        return image, y

    def __len__(self):
        return len(self.tensors[0])

#Create Dataset for pytorch training 

num_sub_traj_seek = 70

#define your model
class Net(nn.Module):
    def __init__(self, perc1, perc2, perc3):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size = 3, stride = 2, padding = 1),#64
            nn.BatchNorm2d(12),
            nn.Dropout(perc1),
            nn.ReLU(True),
            nn.Conv2d(12, 24, kernel_size = 3, stride = 2, padding = 1),#128
            nn.BatchNorm2d(24),
            nn.Dropout(perc2),
            nn.ReLU(True),
            nn.Conv2d(24, 48, kernel_size = 3, stride = 2, padding = 1),#256
            nn.BatchNorm2d(48),
            nn.Dropout(perc3),
            nn.ReLU(True)
            )

        self.rnn = nn.RNN(12,12,batch_first=True)

        self.lin = nn.Sequential(
            nn.Linear(12*48, 2),
            nn.Sigmoid()
        )

    def forward(self, images):
        images = images.permute(0,3,1,2)
        outputs = self.conv(images)
        outputs = outputs.permute(1,0,2,3)
        rnn_output = [self.rnn(pixel)[1][0] for pixel in outputs]
        return self.lin(torch.cat(rnn_output,axis=1))

def predict(images):
    model.eval()

    testset = CustomTensorDataset(tensors=(images, torch.tensor(np.ones(images.shape[0]),dtype=torch.long)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

    predictions = []

    with torch.no_grad():
        for x,y in testloader:
            images = x.to(device)
            labels_y = y.to(device)

            scores = model(images)
            _, calcs = scores.max(1)
            #predictions = np.concatenate((predictions, calcs.detach().cpu().numpy()))
            predictions = scores.detach().cpu().numpy()
    
    #print(predictions)
    return predictions

        

model = Net(0.2, 0.1, 0.0).to(device)
model.load_state_dict(torch.load("final_cnn_rnn_model.pth"))

for i in range(4):
    interpreter = lime_image.LimeImageExplainer()
    explanation = interpreter.explain_instance(explain_images[i], predict, top_labels=2, hide_color=0,num_samples=4000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    img = Image.fromarray(image)
    plt.imshow(image)
    plt.imshow(img_boundry1)
    plt.savefig("img_boundary_" + str(i) +".png")
    plt.close()

print("Finished!")