# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 23:52:39 2016

@author: t-zhhan

K nearest neighbors.
"""

import numpy as np
import data_utils
import classifiers.k_nearest_neighbor as kNN

if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('datasets/cifar10/')   # a magic funtion we provide to load the CIFAR-10 dataset 
    # flatten out all images to be one-dimensional
    # Xtr : 50000 * 32 * 32 * 3; Ytr : 50000
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