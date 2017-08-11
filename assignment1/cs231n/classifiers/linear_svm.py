import numpy as np
from random import shuffle
from past.builtins import xrange

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
    
  # initialize gradient parameters
  ###############################################################################################
  #                  Numerical Gradient Implementation - Part I                                 #
  ##  This part is mainly for parameter initialization. Though unnecessary in python, but       #
  ##  initization do help untangling the messy knots in our mind.                               #
  ###############################################################################################
  #h = 0.00001 # step size
  #scores_h = np.zeros((W.shape[0],W.shape[1],num_classes)) # a tensor of shape of D*C*C
  # loss value matrix of shape (D, C), each cell indicates the loss value after adding step size to corresponding weight
  #loss_h = np.zeros(W.shape) 
  #################################################################################################
  ##                                 End of Part I                                                #
  #################################################################################################

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #################################################################################################
    ####               Numerical Gradient Implementation - Part II                                 ###
    ##                                 Calculate scores                                              #
    #################################################################################################
    #for m in xrange(W.shape[0]):
    #    for n in xrange(W.shape[1]):
    #        W_h = W
    #        W_h[m,n] += h
    #        scores_h[m,n,:] = X[i].dot(W_h)
    #################################################################################################
    ##                                   End of Part II                                             #
    #################################################################################################
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        # compute gradient analytically
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
      #################################################################################################
      ####               Numerical Gradient Implementation - Part III                                 ###
      ##                                 Add up scores                                              #
      #################################################################################################
      #margin_h = scores_h[:,:,j] - correct_class_score + 1
      # zero out low values
      #margin_h[margin_h<.0] = 0
      #loss_h += margin_h
      #################################################################################################
      ##                                   End of Part III .                                          #
      #################################################################################################
      
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #################################################################################################
  ####               Numerical Gradient Implementation - Part IV                                 ###
  ##                                 Calculate gradient                                          #
  #################################################################################################
  # Calculate gradients
  #loss_h /= num_train
  #loss_h += reg * np.sum(W * W)
  #dW = (loss_h - loss)/h
  #################################################################################################
  ##                                   End of Part IV .                                          #
  #################################################################################################


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
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  num_train = y.shape[0]
  # N * C
  scores_all = X.dot(W)
  # N * 1, value of each row = the correct class score w.r.t corresponding input x_i  
  correct_class_scores_all = scores_all[range(num_train),y]  
  # raw loss matrix of shape (N, C)
  loss_matrix = scores_all - correct_class_scores_all[:, None] + 1
  # set loss value of correct class to 0
  loss_matrix[range(num_train),y] = 0
  # zero out negative values (which indicates no loss)
  loss_matrix[loss_matrix<.0] = 0 # zero out negative values
  loss = np.sum(loss_matrix) / num_train + reg * np.sum(W*W)
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
  #pass
  # find cells where value is greater than 0 and set them to 1
  coeff = loss_matrix
  coeff[coeff > 0] = 1
  # set the cell of correct class to negative of the sum of its row
  coeff[range(num_train), y] = -np.sum(coeff, axis=1)

  dW = X.T.dot(coeff)
  # average and incorporate regularization terms
  dW = dW/num_train + 2 * reg * W
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
