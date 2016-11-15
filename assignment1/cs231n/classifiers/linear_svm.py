import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)    # scores: a numpy array of shape (1, C)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # calculate the dW,
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # we need to add reg * W :
  # L = 1/N * Li + 0.5 * reg * W**2
  # do partial derivatives on this formula
  dW += reg * W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C. (N,) means 1 * N
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  Scores = X.dot(W) # Scores: N * C
  # correct_class_scores: 1 * N
  correct_class_scores = Scores[np.arange(num_train), y]
  # correct_class_scores.T equals to itself.
  # so we need to transpose Scores to statify the dimension. and Margins is C * N
  Margins = np.maximum(0, Scores.T - correct_class_scores + 1) # delta set to 1
  Margins = Margins.T # Margins: N * C
  Margins[np.arange(num_train), y] = 0
          
  loss = np.sum(Margins)
  loss /= num_train
  # add regularization
  loss += 0.5 * reg * np.sum(W*W)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  Binary = Margins
  # Binary[i][j] = 1 means: for i-th sample, Sj - Syi + 1 > 0
  Binary[Margins > 0] = 1

  # (1) for all j: dW[j, :] sum_{i, j produces positive margin with i} X[:, i].T
  # (2) for all i: dW[y[i], :] sum_{y != y_i, j produces positive margin with i} -X
  # col_sum respond to how much x_i need to sub.
  col_sum = np.sum(Binary, axis = 1)
  Binary[np.arange(num_train), y] = -col_sum[xrange(num_train)]
  dW = np.dot(X.T, Binary)
  # mean
  dW /= num_train
  # Regularize
  dW += reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
