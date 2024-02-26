# Maitar Asher
# ma4265
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR #for KSVM
from sklearn.metrics import mean_squared_error #to calcualte MSE
import torch #for NN
import torch.nn as nn #for NN
import torch.optim as optim #for NN
from torch.utils.data import DataLoader,TensorDataset #for NN

"""
(i) Plot datasets 
"""

sets = ['complexity0', 'complexity1', 'complexity2', 'complexity3', 'complexity4']
num_samples = [50, 1000]

"""
for dataset in sets:
    for num in num_samples:
        train_file = f"{dataset}/train.csv"
        test_file = f"{dataset}/test.csv"
        df_train = pd.read_csv(train_file, nrows=num)
        x_train = df_train['x']
        y_train = df_train['y']
        plt.scatter(x_train, y_train, alpha=0.5, s=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"{dataset} ({num} Training Samples)")
        plt.show()
"""

"""
(j) Train an RBF KSVM regressor and a NN for each function complexity with varying number
of training samples

You must use the MSE loss.
Mean squared error (MSE) loss is a widely-used loss function in machine learning and statistics 
that measures the average squared difference between the predicted values and the actual target values.

"""

num_samples = [50, 100, 300, 500, 750, 1000]

"""
Training RBF KSVM regressor with varying number of training samples

In machine learning, the radial basis function kernel, or RBF kernel, 
is a popular kernel function used in various kernelized learning algorithms.
In particular, it is commonly used in support vector machine classification.
"""

KSVM_MSE_dict = {} #making a dict, so each data set will have its own errors based on num of samples
for dataset in sets:
    KSVM_MSE = []
    for num in num_samples:
        train_data = pd.read_csv(f"{dataset}/train.csv", nrows=num).to_numpy()
        test_data = pd.read_csv(f"{dataset}/test.csv").to_numpy()
        x_train, y_train = train_data[:, 0].reshape(-1, 1), train_data[:, 1]
        x_test, y_test = test_data[:, 0].reshape(-1, 1), test_data[:, 1]
        ksvm_rbf = SVR(kernel="rbf") #rbf is the kernal we're using
        ksvm_rbf.fit(x_train, y_train) #fit the SVM model according to the given training data.
        y_pred = ksvm_rbf.predict(x_test) #make prediction
        mse = mean_squared_error(y_test, y_pred) #calcuate error
        KSVM_MSE.append(mse) #add error to list
    KSVM_MSE_dict[dataset] = KSVM_MSE

"""
Training NN regressor with varying number of training samples

Sources: 
https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/
https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
"""

"""
Define the class 
We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__.
Every nn.Module subclass implements the operations on input data in the forward method.
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        """
        The model expects rows of data with 1 variable 
        The first hidden layer has 24 neurons
        The second hidden layer has 12 neurons
        The output layer has one neuron
        """
        self.hidden1 = nn.Linear(1, 24)
        self.hidden2 = nn.Linear(24, 12)
        self.output = nn.Linear(12, 1)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.act_fn(self.hidden1(x))
        x = self.act_fn(self.hidden2(x))
        x = self.output(x)
        return x

NN_MSE_dict = {} #making a dict, so each data set will have its own errors based on num of samples
for dataset in sets:
    NN_MSE = []
    for sample in num_samples:
        train_data = pd.read_csv(f"{dataset}/train.csv", nrows=sample)
        test_data = pd.read_csv(f"{dataset}/test.csv")
        x_train = torch.tensor(train_data['x'].values, dtype=torch.float32).reshape(-1, 1) #A PyTorch Tensor is basically the same as a numpy array
        y_train = torch.tensor(train_data['y'].values,  dtype=torch.float32).reshape(-1, 1)
        mydataloader = DataLoader(list(zip(x_train,y_train)),shuffle=True,batch_size=25)
        input_size = x_train.shape[1]
        output_size = y_train.shape[1]
        model = NeuralNetwork()
        nn_loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        """
        An epoch is a measure of the number of times all training data is used once to update the parameters.
        Epoch: Passes the entire training dataset to the model once
        
        Batch: One or more samples passed to the model, from which the gradient descent algorithm
        will be executed for one iteration
        """
        for epoch in range(100):
            for batch_x, batch_y in mydataloader:
                y_pred = model(batch_x)
                loss = nn_loss_function(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        NN_MSE.append(loss.item())

    NN_MSE_dict[dataset] = NN_MSE

for dataset in sets:
    fig, ax = plt.subplots()
    ax.plot(num_samples, KSVM_MSE_dict[dataset], label='KSVM',color='g')
    ax.plot(num_samples, NN_MSE_dict[dataset], label='NN',color='b')
    ax.set_xlabel('Num of Samples')
    ax.set_ylabel('Mean squared error (MSE) loss')
    ax.set_title(f'{dataset} MSE error')
    ax.legend()
    plt.show()

