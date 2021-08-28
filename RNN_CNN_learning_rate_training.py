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

print("Differing Learning rates for the RNN-CNN.")
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
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size = 3, stride = 2, padding = 1),#64
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.Conv2d(12, 24, kernel_size = 3, stride = 2, padding = 1),#128
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 48, kernel_size = 3, stride = 2, padding = 1),#256
            nn.BatchNorm2d(48),
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
        #outputs = outputs.view(images.size(0), -1)
        #rnn_outputs = self.rnn(outputs)
        return self.lin(torch.cat(rnn_output,axis=1))
        #return self.lin(outputs)

data_length = len(training)
test_length = round(data_length / 5)

training_set = training[test_length:]
labels_set = labels[test_length:]

trainset = CustomTensorDataset(tensors=(training_set, torch.tensor(labels_set,dtype=torch.long)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

test_set = training[0:test_length]
test_labels_set = labels[0:test_length]

testset = CustomTensorDataset(tensors=(test_set, torch.tensor(test_labels_set,dtype=torch.long)))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
overall_test_accuracies = []
num_epochs = 20
epochs = range(num_epochs) + np.ones(num_epochs)

for i in range(len(learning_rates)):
    train_accuracies = []
    test_accuracies = []
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rates[i])#, momentum=0.9)

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
    model_name = "learning_rate_model_" + str(i) + "_20_epochs.pth"
    torch.save(net.state_dict(),model_name)

    new_name = "train_accuracy_learning_rate_model_" + str(i) + "_20_epochs.pickle"
    with open(new_name, 'wb') as f:
        pickle.dump(train_accuracies, f)

    new_name = "test_accuracy_learning_rate_model_" + str(i) + "_20_epochs.pickle"
    with open(new_name, 'wb') as f:
        pickle.dump(test_accuracies, f)
    overall_test_accuracies.append(test_accuracies)

    # Plots the number of sampled indexes against the estimation
    plt.plot(epochs, train_accuracies, color="purple", linewidth = 2)

    plt.plot(epochs, test_accuracies, color="red", linewidth = 2)

    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy') 
    plt.legend(["Training Set", "Testing Set"], loc = "upper right")
    plt.title('Training and Testing Accuracy Over 20 Epochs')

    plt.savefig("learning_rate_" + str(i) + "_accuracies_plot.png")
    plt.show()
    plt.close()


end_time = time.perf_counter()
minutes = (end_time - start_time) / 60.0
print(f"Training took {minutes:0.4f} minutes")


# Plots the number of sampled indexes against the estimation
plt.plot(epochs, overall_test_accuracies[0], color="purple", linewidth = 2)

plt.plot(epochs, overall_test_accuracies[1], color="blue", linewidth = 2)

plt.plot(epochs, overall_test_accuracies[2], color="red", linewidth = 2)

plt.plot(epochs, overall_test_accuracies[3], color="green", linewidth = 2)

plt.plot(epochs, overall_test_accuracies[4], color="green", linewidth = 2)

plt.xlabel('Number of Epochs')
plt.ylabel('Test Accuracy')
plt.legend(["Lr 0.0001", "Lr 0.001", "Lr 0.01", "Lr 0.1", "Lr 1"], loc = "upper right")
plt.title('Testing Accuracy with Varying Learning Rates Over 20 Epochs')

plt.savefig('learning_rates_all_models_plot.png')
plt.show()