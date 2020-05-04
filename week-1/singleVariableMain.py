import numpy as np                                          # used for mathematical computation
import pandas as pd                                         # used for reading data from text or excel file
import matplotlib.pyplot as plt                             # used for visualizing data

data = pd.read_csv('ex1data1.txt', header=None)             # read from dataset
X = data.iloc[:, 0]                                         # read first column, automatically as a numpy array
y = data.iloc[:, 1]                                         # read second column
m = len(y)                                                  # number of training example
print(data.head())                                          # view first few rows of the data

X = X[:,np.newaxis]                                         # converting X from shape (m,) to (m,1)......converting x to a column vector from row vector
y = y[:,np.newaxis]                                         # converting y from shape (m,) to (m,1)
ones = np.ones((m,1))                                       # initializing an array of 1s as value for intercept terms
X = np.hstack((ones, X))                                    # adding the intercept term to X
theta = np.zeros([2,1])                                     # Initializing parameters (theta 0 and theta1)......... two row, one column
iterations = 1500                                           # number of iterations to run
alpha = 0.01                                                # value of alpha

input("Press enter if you have completed computeCost file, else Ctrl+C then enter to exit")

'''Functions defined in other files will be imported here'''

from computeCost import computeCost
from gradientDescent import gradientDescent

J= computeCost(X,y,theta)                                   #calling function computeCost from computeCost.py file
print('Cost function J value : ',J)
input("Press enter if you have completed gradient descent file, else Ctrl+C then enter to exit")
theta = gradientDescent(X, y, theta, alpha, iterations)     #calling function gradientDescent from gradientDescent.py file
J= computeCost(X,y,theta)
print('New Cost function value: ',J)

plt.style.use('dark_background')
plt.axis([4.8,25,-5,25])
plt.grid(True, color = 'c', ls = 'dotted')
plt.scatter(X[:,1], y, color = 'g', marker = '.', linewidth = 1)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

plt.plot(X[:,1], np.dot(X, theta), color = 'r', linewidth = 1, label = "Regression Line")
plt.show()

