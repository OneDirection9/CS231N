# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 23:52:39 2016

@author: t-zhhan

K nearest neighbors.
"""
import sys
sys.path.append("..")

import numpy as np
import cs231n.data_utils as data_utils
import cs231n.classifiers.k_nearest_neighbor as kNN
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (10.0, 8.0)    # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('../cs231n/datasets/cifar-10-batches-py/')   # a magic funtion we provide to load the CIFAR-10 dataset 
    # flatten out all images to be one-dimensional
    # Xtr : 50000 * 32 * 32 * 3; Ytr : 50000
    
    # As a sanity check, we print out the size of the training and test data
    
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 * 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xtr_roes becomes 10000 * 3072
    print "********** Loading Dataset Success **********"
    
    print "========== Start Training =========="
    kNearestN = kNN.KNearestNeighbor() # create a K Nearest Neighbor classifier class
    kNearestN.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    print "********** Training Success **********"
    
    print "========== Start Predicting =========="
    YtePredict = kNearestN.predict(Xte_rows, k = 5, num_loops = 0) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted(i.e. label matches)
    print "accuracy:%f" % (np.mean(YtePredict == Yte)) # YtePredict == Yte will generate a bool array. and np.mean calcuate the average value.
    print "********** End of Program **********"

    #no loops:accuracy is 33.98%