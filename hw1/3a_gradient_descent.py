
# coding: utf-8

# In[24]:


import numpy as np
import os
Train_X = np.loadtxt('mnist_2_vs_7/mnist_X_train.dat')
Train_Y = np.loadtxt('mnist_2_vs_7/mnist_y_train.dat')
Test_X = np.loadtxt('mnist_2_vs_7/mnist_X_test.dat')
Test_Y = np.loadtxt('mnist_2_vs_7/mnist_y_test.dat')
N = 10000
D = 780
StepSize = 0.0001
def train(w):
    w_t = gradient_descent(w)
    for step in range(0,200):
        w_t = gradient_descent(w_t)
    return w_t

def gradient_descent(w):
    summation = 0
    error = 0
    for n in range(0,N):
        x_n = Train_X[n]
        y_n = Train_Y[n]
        #error += np.log(1 + np.exp(-1 * y_n * np.dot(w, x_n)))
        summation += y_n * x_n / (1 + np.exp(y_n * np.dot(w,x_n)))
    #in_error = error/N
    #print(in_error)
    gt = (-1/N)*summation
    return w - StepSize * gt

def test(w):
    match = 0
    
    for i in range(len(Test_X)):
        if (np.dot(w,Test_X[i]) < 0) and (Test_Y[i] == -1):
            match+= 1
        elif (np.dot(w,Test_X[i]) > 0) and (Test_Y[i] == 1):
            match+= 1
    return match/len(Test_X)

def main():
    w0 = np.zeros(D)
    w_star = train(w0)
    accuracy = test(w_star)
    print (accuracy)


# In[25]:


main()

