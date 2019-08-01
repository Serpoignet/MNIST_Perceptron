#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn,functional
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='datasets/')

y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

print(y.value_counts())
print(X.describe())

print("Here we goooooo !")

train_batch_size = 42
test_batch_size = 42

scaler = MinMaxScaler()
X = scaler.transform(X)

features_train, features_test, targets_train, targets_test = train_test_split(X,Y,test_size=0.2, random_state=42)

X_train = torch.from_numpy(features_train)
X_test = torch.from_numpy(features_test)

Y_train = torch.from_numpy(targets_train).type(torch.LongTensor) 
Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(X_train,Y_train)
test = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = train_batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = test_batch_size, shuffle = False)


