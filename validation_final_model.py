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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")

start_time = time.perf_counter()

path = 'preprocessed_data'
filenames = glob.glob(path + "/*.pickle")
training = []
labels = []
for filename in filenames:
    data = pickle.load( open(filename, "rb") )
    training.append(np.array(data[0]))
    labels.append(data[1])

print("CNN model on the entire set of data with higher outputs (32, 64, 128), a linear layer, and 20 epochs.")
print(len(training))


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


data_length = len(training)
test_length = round(data_length / 5)

train_accuracies = []
test_accuracies = []
num_epochs = 30

for i in range(5):
    print(i)

    training_set = training[(i+1)*test_length:]
    labels_set = labels[(i+1)*test_length:]
    if i == 4:
        training_set = training[0:i*test_length]
        labels_set = labels[0:i*test_length]
    elif i != 0:
        training_set = np.concatenate((training[0:i*test_length],training[(i+1)*test_length:]),axis=0)
        labels_set = np.concatenate((labels[0:i*test_length],labels[(i+1)*test_length:]),axis=0)

    trainset = CustomTensorDataset(tensors=(training_set, torch.tensor(labels_set,dtype=torch.long)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    
    test_set = training[i*test_length:(i+1)*test_length]
    test_labels_set = labels[i*test_length:(i+1)*test_length]

    testset = CustomTensorDataset(tensors=(test_set, torch.tensor(test_labels_set,dtype=torch.long)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    net = Net(0.2, 0.1, 0.0).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#, momentum=0.9)

    #train your model

    def check_accuracy(testing, loader, model):
        if testing:
            print("Checking accuracy on test data")
        else:
            print("Checking accuracy on training data")

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                images = x.to(device)
                labels = y.to(device)

                scores = model(images)
                _, predictions = scores.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)

                #print(f'Label: {labels}')
                #print(f'Prediction: {predictions}')

            print(f'Got {num_correct} / {num_samples} with accuracy \ {float(num_correct)/float(num_samples)*100:.2f}')

            if testing:
                test_accuracies.append(float(num_correct)/float(num_samples))
            else:
                train_accuracies.append(float(num_correct)/float(num_samples))
            
        
        model.train()

    #check_accuracy(trainloader, net)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for x,y in trainloader:
            images = x.to(device)
            labels_y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)#.reshape(-1)
            loss = criterion(outputs, labels_y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss / 2000))

    check_accuracy(False, trainloader, net)
    check_accuracy(True, testloader, net)
    
    #save model
    model_name = "validation_model_optimal_" + str(i+1) + ".pth"
    torch.save(net.state_dict(),model_name)


end_time = time.perf_counter()
print(f"Training took {end_time - start_time:0.4f} seconds")
minutes = (end_time - start_time) / 60.0
print(f"Training took {minutes:0.4f} minutes")

new_name = "train_accuracy_optimal.pickle"
with open(new_name, 'wb') as f:
    pickle.dump(train_accuracies, f)

new_name = "test_accuracy_optimal.pickle"
with open(new_name, 'wb') as f:
    pickle.dump(test_accuracies, f)

epochs = range(5) + np.ones(num_epochs)
plt.plot(epochs, train_accuracies, color="purple", linewidth = 2)

plt.plot(epochs, test_accuracies, color="red", linewidth = 2)
 
plt.xlabel('Validation Fold')
plt.ylabel('Accuracy') 
plt.legend(["Training Set", "Testing Set"], loc = "upper right")
plt.title('Training and Testing Accuracy Over 5-Fold Cross Validation')

plt.savefig('validation_optimal_plot.png')
plt.show()