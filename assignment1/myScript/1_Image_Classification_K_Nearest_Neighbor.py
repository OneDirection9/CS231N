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
    
    print "========== Start Loading Dataset =========="
    Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('../cs231n/datasets/cifar-10-batches-py/')   # a magic funtion we provide to load the CIFAR-10 dataset 
    # As a sanity check, we print out the size of the training and test data
    print "Training data shape: ", Xtr.shape
    print "Training labels shape: ", Ytr.shape
    print "Test data shape: ", Xte.shape
    print "Test labels shape: ", Yte.shape
    
    ###########################################################################
    # take a sample to short run time.
    num_training = 5000
    mask = xrange(num_training)
    Xtr = Xtr[mask]
    Ytr = Ytr[mask]

    num_test = 500
    mask= range(num_test)
    Xte = Xte[mask]
    Yte = Yte[mask]

    # flatten out all images to be one-dimensional
    # Xtr : 50000 * 32 * 32 * 3; Ytr : 50000
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 * 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xtr_roes becomes 10000 * 3072
    print "********** Loading Dataset Success **********\n"

    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(Ytr == y) # return the samples that is in the class of y
        idxs = np.random.choice(idxs, samples_per_class, replace=False) # choose samples_pre_class of them
        for i, idx in enumerate(idxs):
            plt_idx = num_classes * i + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(Xtr[idx])
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    print "========== Start Training =========="
    kNearestN = kNN.KNearestNeighbor() # create a K Nearest Neighbor classifier class
    kNearestN.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    print "********** Training Success **********\n"
    
    print "========== Cross-validation =========="
    # cross validation to set k
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    Xtr_folds = []
    Ytr_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    Xtr_folds = np.split(Xtr_rows, num_folds, axis = 0)
    Ytr_folds = np.split(Ytr, num_folds, axis = 0)
    pass
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}
    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    for k in k_choices:
        k_to_accuracies[k] = np.zeros(num_folds)
        for i in xrange(num_folds):
            x_tr = np.concatenate([f for j, f in enumerate(Xtr_folds) if j != i])
            y_tr = np.concatenate([f for j, f in enumerate(Ytr_folds) if j != i])
            kNearestN.train(x_tr, y_tr)
            
            y_test_pred = kNearestN.predict(Xtr_folds[i], k)
            iAccuracy = np.mean(Ytr_folds[i] == y_test_pred)
            k_to_accuracies[k][i] = iAccuracy
    pass
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print 'k = %d, accuracy = %f' % (k, accuracy)       
    print "********** End of Cross-validation **********\n"
    
    print "========== Start Predicting =========="
    YtePredict = kNearestN.predict(Xte_rows, k = 5, num_loops = 0) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted(i.e. label matches)
    print "accuracy:%f" % (np.mean(YtePredict == Yte)) # YtePredict == Yte will generate a bool array. and np.mean calcuate the average value.
    print "********** End of Program **********"

    #no loops:accuracy is 33.98%