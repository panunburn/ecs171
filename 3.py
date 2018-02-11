
# coding: utf-8

# In[ ]:


import numpy as np
import os
Train_X = np.loadtxt('mnist_2_vs_7/mnist_X_train.dat')
Train_Y = np.loadtxt('mnist_2_vs_7/mnist_Y_train.dat')
Test_X = np.loadtxt('mnist_2_vs_7/mnist_X_test.dat')
Test_Y = np.loadtxt('mnist_2_vs_7/mnist_Y_test.dat')
N = 10000
D = 780
StepSize = 0.1
def train(w):
    w_t = gradient_descent(w)
    for step in range(0,200):
        w_t = gradient_descent(w_t)
    return w_t

def gradient_descent(w):
    summation = 0
    for n in range(0,N):
        x_n = Train_X[n]
        y_n = Train_Y[n]
        #print (x_n)
        #print (y_n)
        summation += y_n * x_n / (1 + np.exp(y_n * np.dot(w,x_n)))
    gt = (-1/N)*summation
    return w - StepSize * gt

def test(w):
    match = 0
    
    for i in range(N):
        if (np.dot(w,Test_X[i]) < 0) and (Test_Y[i] == -1):
            match+= 1
        elif (np.dot(w,Test_X[i]) > 0) and (Test_Y[i] == 1):
            match+= 1
    return match/N

def main():
    w0 = np.zeros(D)
    w_star = train(w0)
    accuracy = test(w_star)
    print (accuracy)


# In[19]:


main()

