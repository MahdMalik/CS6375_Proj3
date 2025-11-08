from abc import ABC, abstractmethod
from tensorflow.keras.datasets import mnist, cifar10
import random
import time
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch


class Model(ABC):
    @abstractmethod
    def doCustomProcessing(self):
        pass
    def retrieveDataset(self, setType):
        if setType == "mnist":
            (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        else:
            (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

        # normalizing them. Converting pixel values between 0 - 255 to 0 - 1
        xTrain /= 255.0
        xTest /= 255.0

        return (xTrain, yTrain, xTest, yTest)
            
    # initialize method
    def __init__(self):
        (self.mnistXTrain, self.mnistYTrain, self.mnistXTest, self.mnistYTest) = self.retrieveDataset("mnist")
        (self.cifarXTrain, self.cifarYTrain, self.cifarXTest, self.cifarYTest) = self.retrieveDataset("cifar")

        self.parameters = {
            "learningRate": [0.01, 0.001, 0.0001],
            "batchSize": [32, 64, 128],
            "optimizer": ["sgd", "adam"],
            "dropoutRate": [0.2, 0.5]
        }

    def sampleParams(self):
        return {key: random.choice(values) for key, values in self.parameters.items()}
    
    def createTensorsAndSplitting(self):
        self.mnistXTrain = torch.tensor(self.mnistXTrain, dtype=torch.float32)
        self.mnistXTest = torch.tensor(self.mnistXTest, dtype=torch.float32)
        self.mnistYTrain = torch.tensor(self.mnistYTrain, dtype=torch.long).squeeze()
        self.mnistYTest = torch.tensor(self.mnistYTest, dtype=torch.long).squeeze()

        self.cifarXTrain = torch.tensor(self.cifarXTrain, dtype=torch.float32)
        self.cifarXTest = torch.tensor(self.cifarXTest, dtype=torch.float32)
        self.cifarYTrain = torch.tensor(self.cifarYTrain, dtype=torch.long).squeeze()
        self.cifarYTest = torch.tensor(self.cifarYTest, dtype=torch.long).squeeze()

        
        mnistTrain = TensorDataset(self.mnistXTrain, self.mnistYTrain)
        cifarTrain = TensorDataset(self.cifarXTrain, self.cifarYTrain)

        self.mnistTrain, self.mnistValid = random_split(mnistTrain, [50000, 10000])

        self.cifarTrain, self.cifarValid = random_split(cifarTrain, [45000, 5000])

    @abstractmethod
    def doModelCreation(self, complexity, dataset, sampledParams):
        pass

    def createBatches(self, batchSize, dataset):
        
        if(dataset == "mnist"):
            self.trainBatch = DataLoader(self.mnistTrain, batch_size = batchSize, shuffle = True)
            self.testBatch = DataLoader(self.mnistValid, batch_size = batchSize)
        else:
            self.trainBatch = DataLoader(self.cifarTrain, batch_size = batchSize, shuffle = True)
            self.testBatch = DataLoader(self.cifarValid, batch_size = batchSize)

    def createFullModel(self, complexity, dataset):
        startTime = time.time() 
        numValuesUsed = 16
        self.finalParams = {}
        highestAccuracy = 0

        for _ in range(0, numValuesUsed, 1):
            sampledParams = self.sampleParams()

            self.doModelCreation(complexity, dataset, sampledParams)

            self.createBatches(sampledParams["batchSize"], dataset)

            accuracy = self.currentModel.trainAndEvaluate(self.trainBatch, self.testBatch)

            if(accuracy > highestAccuracy):
                highestAccuracy = accuracy
                self.finalParams = sampledParams
        
        self.doModelCreation(complexity, dataset, self.finalParams)
        self.createBatches(self.finalParams["batchSize"], dataset)
        accuracy = self.currentModel.trainAndEvaluate(self.trainBatch, self.testBatch)

        endTime = time.time()

        numMinutes = (endTime - startTime) / 60
        numSeconds = (endTime - startTime) % 60

        print(f"Accuracy for complexity of {complexity} and dataset of {dataset} was {accuracy}, while total elapsed time was {numMinutes} minutes and {numSeconds} seconds. Additionally, the parameters chosen were: {self.finalParams}")