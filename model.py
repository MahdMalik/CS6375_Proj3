from abc import ABC, abstractmethod
from tensorflow.keras.datasets import mnist, cifar10
import random
import time
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
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
        xTrain = xTrain.astype('float32') / 255.0
        xTest = xTest.astype('float32') / 255.0

        return (xTrain, yTrain, xTest, yTest)
            
    # initialize method
    def __init__(self):
        (self.mnistXTrain, self.mnistYTrain, self.mnistXTest, self.mnistYTest) = self.retrieveDataset("mnist")
        (self.cifarXTrain, self.cifarYTrain, self.cifarXTest, self.cifarYTest) = self.retrieveDataset("cifar")

        # all the parameters we can choose form
        self.parameters = {
            "learningRate": [0.01, 0.001, 0.0001],
            "batchSize": [32, 64, 128],
            "optimizer": ["sgd", "adam"],
            "dropoutRate": [0.2, 0.5]
        }

        # trying to use the GPU if possible, ependeing on where we run it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # this allows us to randomly sample from each parameter
    def sampleParams(self):
        return {key: random.choice(values) for key, values in self.parameters.items()}
    
    # create the tensors for each array, as well as TensorDatasets
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

        self.mnistTest = TensorDataset(self.mnistXTest, self.mnistYTest)
        self.cifarTest = TensorDataset(self.cifarXTest, self.cifarYTest)

        # after we create the tensors we can do this
        self.mnistTrain, self.mnistValid = random_split(mnistTrain, [50000, 10000])

        self.cifarTrain, self.cifarValid = random_split(cifarTrain, [45000, 5000])

    # abstract method the subclasses implement
    @abstractmethod
    def doModelCreation(self, complexity, dataset, sampledParams):
        pass

    # We need to do this each hypertuning because the batch size is a hyeprparameter, so we cneed to create batches dynamically
    def createBatches(self, batchSize, dataset, doingFinalTrain):
        if(dataset == "mnist"):
            if(doingFinalTrain):
                mergedTrain = ConcatDataset([self.mnistTrain, self.mnistValid])
                self.trainBatch = DataLoader(mergedTrain, batch_size = batchSize, shuffle = True)
                self.testBatch = DataLoader(self.mnistTest, batch_size = batchSize)
            else:
                self.trainBatch = DataLoader(self.mnistTrain, batch_size = batchSize, shuffle = True)
                self.testBatch = DataLoader(self.mnistValid, batch_size = batchSize)
        else:
            if(doingFinalTrain):
                mergedTrain = ConcatDataset([self.cifarTrain, self.cifarValid])
                self.trainBatch = DataLoader(mergedTrain, batch_size = batchSize, shuffle = True)
                self.testBatch = DataLoader(self.cifarTest, batch_size = batchSize)
            else:
                self.trainBatch = DataLoader(self.cifarTrain, batch_size = batchSize, shuffle = True)
                self.testBatch = DataLoader(self.cifarValid, batch_size = batchSize)

    # this creates the model, calls the function for tuning, and does evaluation and returns the rutnime.
    def createFullModel(self, complexity, dataset, modelName):
        startTime = time.time() 
        tuneIterations = 11
        self.finalParams = {}
        highestAccuracy = 0

        # hypertune using random search with about 11 iterations
        for _ in range(0, tuneIterations, 1):
            #get a random search sample
            sampledParams = self.sampleParams()

            # crate our model according to sampled parameters
            self.doModelCreation(complexity, dataset, sampledParams)

            # create the batches
            self.createBatches(sampledParams["batchSize"], dataset, False)

            # train and valuate this current model, get accuracy
            accuracy = self.currentModel.trainAndEvaluate(self.trainBatch, self.testBatch, False, self.device)

            # keep the parameter set with the highest scores
            if(accuracy > highestAccuracy):
                highestAccuracy = accuracy
                self.finalParams = sampledParams
        
        # do model createion again with the final parameters
        self.doModelCreation(complexity, dataset, self.finalParams)

        # create batches again, but this time pass in doingFinalTrain = True to concatenate train and validation sets togethaa
        self.createBatches(self.finalParams["batchSize"], dataset, True)
        # get accuracy to print
        accuracy = self.currentModel.trainAndEvaluate(self.trainBatch, self.testBatch, True, self.device)

        # get end time from start of tuning to end of final model's training
        endTime = time.time()
        numMinutes = (endTime - startTime) / 60
        numSeconds = (endTime - startTime) % 60

        print(f"Accuracy for {modelName} model with complexity of {complexity} and dataset of {dataset} was {accuracy}, while total elapsed time was {numMinutes} minutes and {numSeconds} seconds. Additionally, the parameters chosen were: {self.finalParams}")