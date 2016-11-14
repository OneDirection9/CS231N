# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:33:36 2016

@author: t-zhhan

Linear Classifier
    - Loss function

Here is the loss function (without regularization) implemented in Python,
in both unvectorized and half-vectorized form.
"""
import numpy as np

def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x, y)
    - x is a column vector representing an image (e.g. 3073 * 1 in CIFAR-10)
      with an appended bias dimension in the 3074-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 * 3073 in CIFAR-10)
    """
    delta = 1.0 # see notes about delta later in this section
    scores = W.dot(x) # scores becomes of size 10 * 1, the scores for each class
    correct_class_score = scores[y]
    D = W.shape[0] # number of classes, e.g. 10
    loss_i = 0.0
    
    for j in xrange(D): # iterator over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the i-th example 
        loss_i += max(0, scores[j] - correct_class_score + delta)
    
    return loss_i
    
def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y]- scores[y] cancled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    
    return loss_i
    
def L(X, y, W):
    """
    fully-vectorized implementation:
    - X holds all the training examples as columns (e.g. 3073 * 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 * 3073)
    """
    delta = 1.0
    Scores = W.dot(X) # scores becomes of size 10 * 50,000
    
    # compute correct scores for each image
    # Correct_Class_Score is 1 * 50,000
    Correct_Class_Score = Scores[y, np.arange(Scores.shape[1])]
    # compute all margins for all classes
    # Margins is 10 * 50,000
    Margins = np.maximum(0, Scores - Correct_Class_Score + delta)
    # on j-th column, i-th row was gave delta.
    # We want to ignore the delta.
    Margins[y, np.arange(Margins.shape[1])] = 0
    Loss = np.sum(Margins, axis = 0)
    
    return Loss
    