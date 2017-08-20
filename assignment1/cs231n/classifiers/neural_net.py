from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import pdb

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    
    self.params['W1_m'] = np.zeros((input_size, hidden_size))
    self.params['W1_v'] = np.zeros((input_size, hidden_size))
    self.params['W2_m'] = np.zeros((hidden_size, output_size))
    self.params['W2_v'] = np.zeros((hidden_size, output_size))
    self.params['b1_m'] = np.zeros(hidden_size)
    self.params['b1_v'] = np.zeros(hidden_size)    
    self.params['b2_m'] = np.zeros(output_size)
    self.params['b2_v'] = np.zeros(output_size)
    
  def loss(self, X, y=None, reg=0.0, dropout_ratio=0.):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    
    ReLU = lambda x: np.maximum(0,x) # ReLU activation function
    a_2 = X.dot(W1)+b1
    h_2 = ReLU(a_2)
    
    #--------implement dropout
    if dropout_ratio != 0.:
        mask = (np.random.rand(*h_2.shape) < dropout_ratio)/dropout_ratio # indicate where to dropout
        h_2 *= mask # dropout!
    #--------dropout finish
    
    scores = h_2.dot(W2)+b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    pass
    scores_exp = np.exp(scores)
    softmax = scores_exp[range(N), y]/np.sum(scores_exp,axis=1)
    loss = 1 / float(N) * np.sum(-np.log(softmax)) + reg*(np.sum(W1*W1)+np.sum(W2*W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    #------------------------------- The verbose form ---------------------------
    d_loss = 1.
    d_aver = 1 / float(N) * d_loss  
    d_minus_1 = -1 * d_aver
    d_log = 1./softmax * d_minus_1
    
    # for classes that k != y 
    d_scores = -scores_exp / np.sum(scores_exp, axis=1, keepdims=True) * softmax[:,None] * d_log[:,None]
    # for classes that k=y
    d_scores[range(N), y] += softmax * d_log
    dW2 = h_2.T.dot(d_scores)
    dh_2 = d_scores.dot(W2.T)
    db2 = np.sum(d_scores, axis=0)
    da_2 = np.maximum(0, a_2)/a_2 * dh_2
    dW1 = X.T.dot(da_2)
    db1 = np.sum(da_2, axis=0)
    #--------------------------------------------------------------------
    
    #============================== The compact form ============================== 
    #d_scores = scores_exp/np.sum(scores_exp,axis=1, keepdims=True)
    #d_scores[range(N), y] -= 1.
    #d_scores /= float(N)
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    #dW2 = np.dot(h_2.T, d_scores)
    #db2 = np.sum(d_scores, axis=0, keepdims=True)
    # next backprop into hidden layer
    #dhidden = np.dot(d_scores, W2.T)
    # backprop the ReLU non-linearity
    #dhidden[h_2 <= 0] = 0
    # finally into W,b
    #dW1 = np.dot(X.T, dhidden)
    #db1 = np.sum(dhidden, axis=0, keepdims=True)
    #============================================================================== 
    
    dW2 += 2 * reg * W2
    dW1 += 2 * reg * W1
    grads['W2'] = dW2
    grads['W1'] = dW1
    grads['b2'] = db2
    grads['b1'] = db1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False,
           d_r=0., opt_alg='SGD'): # append the dropout ratio
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    # parameters for adam algorithm
    W1_m = self.params['W1_m']
    W1_v = self.params['W1_v']
    W2_m = self.params['W2_m']
    W2_v = self.params['W2_v']
    b1_m = self.params['b1_m']
    b1_v = self.params['b1_v']   
    b2_m = self.params['b2_m']
    b2_v = self.params['b2_v']
    
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      indices_batch = np.random.choice(num_train, batch_size, replace = True)
      X_batch = X[indices_batch]
      y_batch = y[indices_batch]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout_ratio=d_r)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
    
      if opt_alg == 'SGD':
        self.params['W2'] -= learning_rate*grads['W2']
        self.params['W1'] -= learning_rate*grads['W1']
        self.params['b2'] -= learning_rate*grads['b2']
        self.params['b1'] -= learning_rate*grads['b1']
      if opt_alg == 'Adam':
        t=it+1
        
        W2_m = 0.9*W2_m + 0.1*grads['W2']
        W2_mt = W2_m / (1-0.9**t)
        W2_v = 0.999*W2_v + 0.001*(grads['W2']**2)
        W2_vt = W2_v / (1-0.999**t)
        self.params['W2'] += - learning_rate * W2_mt / (np.sqrt(W2_vt) + 1e-8)
        
        b2_m = 0.9*b2_m + 0.1*grads['b2']
        b2_mt = b2_m / (1-0.9**t)
        b2_v = 0.999*b2_v + 0.001*(grads['b2']**2)
        b2_vt = b2_v / (1-0.999**t)
        self.params['b2'] += - learning_rate * b2_mt / (np.sqrt(b2_vt) + 1e-8)

        W1_m = 0.9*W1_m + 0.1*grads['W1']
        W1_mt = W1_m / (1-0.9**t)
        W1_v = 0.999*W1_v + 0.001*(grads['W1']**2)
        W1_vt = W1_v / (1-0.999**t)
        self.params['W1'] += - learning_rate * W1_mt / (np.sqrt(W1_vt) + 1e-8)
        
        b1_m = 0.9*b1_m + 0.1*grads['b1']
        b1_mt = b1_m / (1-0.9**t)
        b1_v = 0.999*b1_v + 0.001*(grads['b1']**2)
        b1_vt = b1_v / (1-0.999**t)
        self.params['b1'] += - learning_rate * b1_mt / (np.sqrt(b1_vt) + 1e-8)

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    a2 = X.dot(self.params['W1']) + self.params['b1']
    ReLU = lambda x: np.maximum(0,x) # ReLU activation function
    h2 =ReLU(a2)
    scores_matrix = h2.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores_matrix, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


