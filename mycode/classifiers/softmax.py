import numpy as np
from random import shuffle

from matplotlib.dates import num2date


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores = scores - np.max(scores)
    correct_class_score = scores[y[i]]
    ef = np.exp(correct_class_score)
    sum_ef = np.sum(np.exp(scores))
    loss+= -np.log(ef/sum_ef)

    for j in range(num_classes):
      p = np.exp(scores[j]) / sum_ef
      dW[:,j] += (p - (j == y[i])) * X[i, :]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores,axis=1)[:,np.newaxis]
  correct_classes_scores = scores[np.arange(num_train),y]
  ef = np.exp(correct_classes_scores)
  sum_ef = np.sum(np.exp(scores),axis=1)[:,np.newaxis]
  p = np.exp(scores) / sum_ef
  loss = np.sum(-np.log(p[np.arange(num_train),y]))
  mask = np.zeros(p.shape)
  mask[np.arange(num_train), y] = 1

  dW = X.T.dot(p - mask)


  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

