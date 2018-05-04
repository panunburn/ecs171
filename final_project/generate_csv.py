
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from io import StringIO


# In[2]:


X_train = np.load("data/ecs171train.npy")
X_test = np.load("data/ecs171test.npy")
X_train = [x.decode() for x in X_train]
X_test = [x.decode() for x in X_test]


# In[ ]:


X_str = ""
Y_str = ""
i = 0
for x_str in X_train:
    x_str = x_str.strip()
    if (i == 0):
            Y_str += (x_str[:-5] + "\n")
            i+=1
    X_str += (x_str + "\n")

print("collect X_train string.")
 
TRAIN_DATA=StringIO(X_str)
df = pd.read_csv(TRAIN_DATA, sep=",")
df.to_csv('data/ecs171train.csv')

print("generate X_train csv.")


for x_str in X_test:
    x_str = x_str.strip()
    Y_str += (x_str + "\n")

print("collect X_test string.")

TEST_DATA=StringIO(Y_str)
df = pd.read_csv(TEST_DATA, sep=",", low_memory=False)
df.to_csv('data/ecs171test.csv')

print("generate X_test csv.")
