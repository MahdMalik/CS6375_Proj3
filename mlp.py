from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP_Manager(Model):
    def doCustomProcessing(self):
        # a mlp just expects a 1 dimensional array as its input, so we must flatten our 2D array to a 1D one.
        self.mnistXTrain = self.mnistXTrain.reshape(self.mnistXTrain.shape[0], self.mnistXTrain.shape[1] * self.mnistXTrain.shape[2])
        self.mnistXTest = self.mnistXTest.reshape(self.mnistXTest.shape[0], self.mnistXTest.shape[1] * self.mnistXTest.shape[2])

        # may not make a difference, but just in case we transpose before flattening. Do this because for the cnn, we have to do this as
        # it expects channel first
        self.cifarXTrain = self.cifarXTrain.transpose(0, 3, 1, 2)
        self.cifarXTest = self.cifarXTest.transpose(0, 3, 1, 2)

        # a mlp just expects a 1 dimensional array as its input, so we must flatten our 3D array to a 1D one.
        self.cifarXTrain = self.cifarXTrain.reshape(self.cifarXTrain.shape[0], 
                                                    self.cifarXTrain.shape[1] * self.cifarXTrain.shape[2] * self.cifarXTrain.shape[3])
                
        self.cifarXTest = self.cifarXTest.reshape(self.cifarXTest.shape[0], 
                                                    self.cifarXTest.shape[1] * self.cifarXTest.shape[2] * self.cifarXTest.shape[3])
    
    # do the common pre processing before doing what s special to the MLP
    def __init__(self):
        super().__init__()
        # now, custom stuff done
        self.doCustomProcessing()

        self.createTensorsAndSplitting()

    def doModelCreation(self, complexity, dataset, sampledParams):        
        # firest dimension is # of points, so the one after is the number of features
        if(dataset == "mnist"):
            numFeatures = self.mnistXTrain.shape[1]
        else:
            numFeatures = self.cifarXTrain.shape[1]
        
        self.currentModel = MLP(complexity, numFeatures, **sampledParams).to(self.device)


class MLP(nn.Module):
    def __init__(self, complexity, numFeatures, learningRate, optimizer, dropoutRate, batchSize):
        super().__init__()
        layers = []

        
        # do all the layers except for the finaln lyaer
        beforeHiddenLayerLength = -1
        if(complexity == "low"):
            numPerceptrons = 128
            layers.append(nn.Linear(numFeatures, numPerceptrons))
            layers.append(nn.ReLU())
            beforeHiddenLayerLength = numPerceptrons
        elif(complexity == "medium"):
            numPerceptronsFirst = 512
            numPerceptronsSecond = 256
            numPerceptronsThird = 128

            layers.append(nn.Linear(numFeatures, numPerceptronsFirst))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsFirst, numPerceptronsSecond))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsSecond, numPerceptronsThird))
            layers.append(nn.ReLU())

            beforeHiddenLayerLength = numPerceptronsThird
        elif(complexity == "high"):
            numPerceptronsFirst = 1024
            numPerceptronsSecond = 512
            numPerceptronsThird = 256
            numPerceptronsFourth = 128
            numPerceptronsFifth = 64
            numPerceptronsSixth = 32

            layers.append(nn.Linear(numFeatures, numPerceptronsFirst))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsFirst, numPerceptronsSecond))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsSecond, numPerceptronsThird))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsThird, numPerceptronsFourth))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsFourth, numPerceptronsFifth))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(numPerceptronsFifth, numPerceptronsSixth))
            layers.append(nn.ReLU())

            beforeHiddenLayerLength = numPerceptronsSixth
    
        # apply dropout before the final layer
        layers.append(nn.Dropout(dropoutRate))
        
        # now do the final layer
        numClasses = 10
        layers.append(nn.Linear(beforeHiddenLayerLength, numClasses))

        self.model = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()

        if(optimizer == "adam"):
            self.optimizer = optim.Adam(self.parameters(), lr = learningRate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr = learningRate)

    def forward(self, x):
        return self.model(x)

    def trainAndEvaluate(self, trainSet, testSet, finalTrain, device):
        if(finalTrain):
            numEpochs = 50
        else:
            numEpochs = 30
        
        for epoch in range(numEpochs):
            for batchX, batchY in trainSet:
                batchX, batchY = batchX.to(device), batchY.to(device)
                
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
                batchX, batchY = batchX.to(device), batchY.to(device)
                output = self(batchX)
                predictedOutputs = torch.argmax(output, dim = 1)

                for i in range(0, len(predictedOutputs), 1):
                    if(predictedOutputs[i] == batchY[i]):
                        numCorrect += 1
                    numTotal += 1

        return numCorrect / numTotal