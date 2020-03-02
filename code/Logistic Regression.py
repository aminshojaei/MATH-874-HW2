# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:21:17 2020

@author: Amin Shojaeighadikolaei
"""
#
############################################################## Import Libraries
import numpy as np
import matplotlib.pyplot as plt

##################################################################### Functions
def sigmoid(z):
    output = []
    for s in z:
        output.append(1 / (1 + np.exp(-s)))
    return np.asarray(output)
################################################################## Load dataset
dataset= np.load(r'C:\Users\a335s717\Desktop\HW2\mnist148.npz')
new_dataset= dataset.files
X = dataset['arr_0']
Y = dataset['arr_1']
Test = dataset['arr_2']

for c in range(4):
    plt.subplot(2,2,c+1)
    plt.imshow(X[c])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('MNIST'+str(c+1))

############################################################# Preparing Dataset
Input = []
Output=[]
count = np.zeros((10))
w = np.random.random((28 * 28, 3))
for x, y in zip(X,Y):
    if y in [1, 4, 8]:
        Input.append(x.reshape((28 * 28)) / 255)
        count[y] += 1
        
        if y == [1]:
            y = [1, 0, 0]
        elif y == [4]:
            y = [0, 1, 0]
        elif y == [8]:
            y = [0, 0, 1]   
        Output.append(y)
x_test=[]
for xx in Test :
    x_test.append(xx.reshape((28 * 28)) / 255)

samples = np.asarray(Input)
labels = np.asarray(Output)
test = np.asarray(x_test)

X_train = samples
Y_train = labels
X_Test = test


################################################################ Main algotithm

alpha = 0.0001
while True:
    yBar = np.dot(X_train, w)
    yBar = sigmoid(yBar)
    error = Y_train - yBar
    delta = np.dot(X_train.T, error)
    w += alpha * delta
    Error = np.abs(np.mean(error))
    print('Error: %.8f' % Error)
    if Error < 0.05:
        break


################################################################ Test the Model

for c in range(3):
    plt.subplot(2,2,c+1)
    plt.imshow(Test[c])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Sample test'+str(c+1))
    
Y_Test = np.dot(X_Test[0] , w)
print(' predicted class for first sample is:  %d' % ( np.argmax(Y_Test)))
Y_Test = np.dot(X_Test[1] , w)
print(' predicted class for second sample is:  %d' % ( np.argmax(Y_Test)))
Y_Test = np.dot(X_Test[2] , w)
print(' predicted class for third sample is:  %d' % ( np.argmax(Y_Test)))
Y_Test = np.dot(X_Test[2] , w)



