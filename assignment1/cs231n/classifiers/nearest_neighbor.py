# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 16:05:02 2016

@author: t-zhhan
"""
import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, Y):
        """
        X is N x D where each row is an example. Y is 1-dimension of size N
        """
        # the nearest neighbor classifier simply remembers all the training data.
        self.Xtr = X
        self.Ytr = Y
        
    def predict(self, X):
        """
        X is N x D where each row is an example we wish to predict label for.
        """
        # get the first dimension of X
        num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)  # generate a 1-dimension ndarray of size num_test with initial value 0
        
        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.Xtr - X[i ,:]), axis = 1)    # X[i , :] equals to X[i]: get the ith row of X.
            
            # # using the L2 distance
            # distances = np.sum(np.square(self.Xtr - X[i , :]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]
        
        return Ypred