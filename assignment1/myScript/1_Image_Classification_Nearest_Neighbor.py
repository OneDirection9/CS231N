# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 14:30:04 2016

@author: t-zhhan

Train the Nearest Neighbor classifier on CIFAR-10.
We choose the L1 diatance as the measure.
"""
import sys
sys.path.append("..")

import numpy as np
import cs231n.data_utils as data_utils
import cs231n.classifiers.nearest_neighbor as nearestNeb

if __name__ == "__main__":
    print "========== Start Loading CIFAR-10 =========="
    Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('../cs231n/datasets/cifar10/')   # a magic funtion we provide to load the CIFAR-10 dataset 
    # flatten out all images to be one-dimensional
    # Xtr : 50000 * 32 * 32 * 3; Ytr : 50000
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 * 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xtr_roes becomes 10000 * 3072
    print "********** Loading Dataset Success **********"
    
    print "========== Start Training =========="
    nearestN = nearestNeb.NearestNeighbor() # create a Nearest Neighbor classifier class
    nearestN.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    print "********** Training Success **********"
    
    print "========== Start Predicting =========="
    YtePredict = nearestN.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted(i.e. label matches)
    print "accuracy:%f" % (np.mean(YtePredict == Yte)) # YtePredict == Yte will generate a bool array. and np.mean calcuate the average value.
    print "********** End of Program **********"
    
    # The accuracy is 38.59% in the CIFAR-10 dataset.