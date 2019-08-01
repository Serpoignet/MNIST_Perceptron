# %%

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import pandas as pd

# %%
mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='datasets/')

# %%
y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

# %%
print(y.value_counts())
print(X.describe())