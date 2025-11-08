from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CNN_Manager(Model):
    # this method does custom processing needed for the CNN
    def doCustomProcessing(self):
        # the CNN expects a 3D array; the MNIST is 2D as it has no dimension for color (its monochrome), so we
        # squeeze in another dimension
        self.mnistXTrain = np.expand_dims(self.mnistXTrain, axis = 1)
        self.mnistXTest = np.expand_dims(self.mnistXTest, axis = 1)

        # the cnn expects color dimension first, so just swap the dimensions for this
        self.cifarXTrain = self.cifarXTrain.transpose(0, 3, 1, 2)
        self.cifarXTest = self.cifarXTest.transpose(0, 3, 1, 2)

    # initial function that runs the common processing, then does what's special to the CNN
    def __init__(self):
        super().__init__()
        # now, custom stuff done
        self.doCustomProcessing()

        self.createTensorsAndSplitting()

    def doModelCreation(self, complexity, dataset, sampledParams):
        # firest dimension is # of points, so the one after is the number of features
        if(dataset == "mnist"):
            numChannels = 1
        else:
            numChannels = 3
        
        self.currentModel = CNN(complexity, numChannels, **sampledParams)

class CNN(nn.Module):
    def __init__(self, complexity, numChannels, learningRate, optimizer, dropoutRate, batchSize):
        super().__init__()
        layers = []
        
        finalNumFilters = 0

        if(complexity == "low"):
            numFiltersFirstLayer = 32
            convStepSize = 1
            firstKernelSize = 5
            layers.append(nn.Conv2d(numChannels, numFiltersFirstLayer, firstKernelSize, padding = 1))
            layers.append(nn.ReLU())

            # dimension of each filter after will be 26 x 26 for MNIST, and 30 x 30 for CIFAR. SO, we can use the following pooling fine

            poolKernelSize = 2
            poolStepSize = poolKernelSize
            layers.append(nn.MaxPool2d(poolKernelSize, poolStepSize))

            # after this, dimension of each filter will be 13 x 13 for MNIST, and 15 x 15 for CIFAR
            numFiltersSecondLayer = 64
            secondKernelSize = 3
            layers.append(nn.Conv2d(numFiltersFirstLayer, numFiltersSecondLayer, secondKernelSize, padding = 1))
            layers.append(nn.ReLU())

            layers.append(nn.AdaptiveAvgPool2d(1))
            
            finalNumFilters = numFiltersSecondLayer
        elif(complexity == "medium"):
            numFiltersFirstLayer = 32
            convStepSize = 1
            firstKernelSize = 5
            layers.append(nn.Conv2d(numChannels, numFiltersFirstLayer, firstKernelSize, padding = 1))
            layers.append(nn.BatchNorm2d(numFiltersFirstLayer))
            layers.append(nn.ReLU())

            # dimension of each filter after will be 26 x 26 for MNIST, and 30 x 30 for CIFAR. SO, we can use the following pooling fine

            poolKernelSize = 2
            poolStepSize = poolKernelSize
            layers.append(nn.MaxPool2d(poolKernelSize, poolStepSize))

            # after this, dimension of each filter will be 12 x 12 for MNIST, and 14 x 14 for CIFAR
            numFiltersSecondLayer = 64
            secondKernelSize = 3
            layers.append(nn.Conv2d(numFiltersFirstLayer, numFiltersSecondLayer, secondKernelSize, padding = 1))
            layers.append(nn.BatchNorm2d(numFiltersSecondLayer))
            layers.append(nn.ReLU())

            layers.append(nn.AdaptiveAvgPool2d(1))
            
            finalNumFilters = numFiltersSecondLayer
        elif(complexity == "high"):
            numFiltersFirstLayer = 32
            convStepSize = 1
            firstKernelSize = 3
            layers.append(nn.Conv2d(numChannels, numFiltersFirstLayer, firstKernelSize, padding = 1))
            layers.append(nn.BatchNorm2d(numFiltersFirstLayer))
            layers.append(nn.ReLU())

            # dimension of each filter after will be 28 x 28 for MNIST, and 32 x 32 for CIFAR. SO, we can use the following pooling fine

            poolKernelSize = 2
            poolStepSize = poolKernelSize
            layers.append(nn.MaxPool2d(poolKernelSize, poolStepSize))

            # after this, dimension of each filter will be 14 x 14 for MNIST, and 16 x 16 for CIFAR
            numFiltersSecondLayer = 64
            secondKernelSize = 3
            layers.append(nn.Conv2d(numFiltersFirstLayer, numFiltersSecondLayer, secondKernelSize, padding = 1))
            layers.append(nn.BatchNorm2d(numFiltersSecondLayer))
            layers.append(nn.ReLU())

            layers.append(nn.MaxPool2d(poolKernelSize, poolStepSize))

            numFiltersThirdLayer = 128
            secondKernelSize = 2
            layers.append(nn.Conv2d(numFiltersSecondLayer, numFiltersThirdLayer, secondKernelSize, padding = 1))
            layers.append(nn.BatchNorm2d(numFiltersThirdLayer))
            layers.append(nn.ReLU())

            layers.append(nn.AdaptiveAvgPool2d(1))

            finalNumFilters = numFiltersThirdLayer

        layers.append(nn.flatten())
        layers.append(nn.Linear(finalNumFilters, finalNumFilters * 2))
        layers.append(nn.ReLU())

        # first, remove on hidden layers
        layers.append(nn.Dropout(dropoutRate))

        numClasses = 10
        layers.append(nn.Linear(finalNumFilters * 2, numClasses))

        self.model = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 50

        if(optimizer == "adam"):
            self.optimizer = optim.Adam(self.parameters(), lr = learningRate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr = learningRate)
    
    def forward(self, x):
        return self.model(x)
    
    def trainAndEvaluate(self, trainSet, testSet):
        for epoch in range(self.num_epochs):
            for batchX, batchY in trainSet:
                # zero out gradients
                self.optimizer.zero_grad()

                rawOutputs = self(batchX)

                loss = self.criterion(rawOutputs, batchY)

                # do backward step
                loss.backward()

                self.optimizer.step()

        numCorrect = 0
        numTotal = 0
        with torch.no_grad():
            for batchX, batchY in testSet:
                output = self(batchX)
                predictedOutputs = torch.argmax(output, dim = 1)

                for i in range(0, len(predictedOutputs), 1):
                    if(predictedOutputs[i] == batchY[i]):
                        numCorrect += 1
                    numTotal += 1

        return numCorrect / numTotal