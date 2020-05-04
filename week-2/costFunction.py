import numpy as np
from sigmoid import sigmoid
epilson = 1e-5


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -((np.log(h + epilson).T).dot(y)+np.log(1-h + epilson).T.dot(1-y))/m

    return J


def gradient(theta, X, y):
	'''' calculate gradient descent for logistic regression'''
	m = y.size
	theta=theta.reshape(-1,1)
	h = sigmoid(X.dot(theta)) 
	grad = ((1/m)* np.dot(X.T,(h-y)))

	return(grad.flatten())			# returns copy of array in one dimension
