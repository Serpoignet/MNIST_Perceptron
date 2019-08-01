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
from sklearn import MinMaxScaler

import pandas as pd

mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='datasets/')

y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

print(y.value_counts())
print(X.describe())

print("Here we goooooo !")

train_batch_size = 42
test_batch_size = 42

scale = MinMaxScaler()
X = scale(X)

