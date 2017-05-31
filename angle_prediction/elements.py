# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:23:37 2017

integrate fundemantals of deep learning models

Only the following are used in ResNet:
MultiLogisticRegression: 


@author: ygao
"""

import theano  
import numpy as np  
import cPickle
import os
import gzip   
import theano.tensor as T  
#from theano.tensor.nnet import conv


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
	"""
	input: (sample_size, n_in) 
	n_in:  the num of feats
	n_out: the num of clusters/labels
	p_y_given_x: (sample_size, n_out)
	y_pred: (sample_size, )
	"""

        self.W = theano.shared(
            value = np.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = np.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]
        self.paramL1 = abs(self.W).sum() + abs(self.b).sum()
        self.paramL2 = (self.W**2).sum() + (self.b**2).sum()

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def TestLogisticRegression():
    N = 400     # training sample size
    feats = 784    # number of input variables
    rng = np.random
    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randint(size = N, low = 0, high = 2))
    training_steps = 10000
    # Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.ivector("y")
    LR = LogisticRegression(x, n_in = feats, n_out = 2) 
    
    prediction = LR.y_pred    
    cost = LR.negative_log_likelihood(y) + 0.01 * LR.paramL2  # The cost to minimize
    gparams = T.grad(cost, LR.params)     # Compute the gradient of the cost
    
    # Compile
    updates = [(param, param - 0.1 * gparam)  
        for param, gparam in zip(LR.params, gparams)]
    
    train = theano.function(
        inputs = [x,y],
        outputs = [prediction, LR.negative_log_likelihood(y)],
        updates = updates,
        allow_input_downcast = True)
    predict = theano.function(inputs = [x], outputs = prediction)
    
    for i in range(training_steps):
        pred, err = train(D[0], D[1])
    
    print("Final model:")
    print(LR.W.get_value())
    print(LR.b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))
    print(D[1] == predict(D[0]))

class MultiLogisticRegression(object):
    def __init__(self, input, n_in, n_out, sel = None):
	"""
	input: ( batch_size, n_sites, n_in ), where n_sites is the max protein length 
	n_in:  the num of feats
	n_out: the num of clusters/labels
	p_y_given_x: (batch_size * n_sites, n_out)
	y_pred: (batch_size, n_sites)
        sel: total selected site indexes (n_sel_sites, )
        pp: selected probability matrix (n_sel_sites, n_out)
	"""
	input2 = input.reshape( (-1, n_in))
	self.input = input
	self.n_out = n_out
	if sel is not None:
	    sel = sel.dimshuffle(0, 'x')
	self.sel = sel
        self.W = theano.shared(
            value = np.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = np.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

	self.p_y_given_x = T.nnet.softmax(T.dot(input2, self.W) + self.b)
	self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
	if self.sel is not None:
	    self.pp = self.p_y_given_x[ sel[T.arange(sel.shape[0]),0], : ]
        self.params = [self.W, self.b]
        self.paramL1 = abs(self.W).sum() + abs(self.b).sum()
        self.paramL2 = (self.W**2).sum() + (self.b**2).sum()

    def negative_log_likelihood(self, y):
	
	if self.sel is not None:
	    yy = y[ self.sel[T.arange(self.sel.shape[0]),0] ]
	    return -T.mean(T.log(self.pp)[T.arange(yy.shape[0]), yy])
	else:
	    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
	    if self.sel is not None:
		yy = y[ self.sel[T.arange(self.sel.shape[0]),0] ]
		y_pred = self.y_pred[ self.sel[T.arange(self.sel.shape[0]),0] ]
        	return T.mean(T.neq(y_pred, yy))
	    else:
        	return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def TestMultiLogisticRegression():
    batch_size = 2
    N = 10000                                   # training sample size
    feats = 66                               # number of input variables
    rng = np.random
    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(batch_size, N, feats), 
         rng.randint(size = (batch_size * N), 
         low = 0, high = 2))
    D[1][:5] = -1
    training_steps = 100
    print D[0].shape
    # Declare Theano symbolic variables
    x = T.tensor3("x")
    y = T.ivector("y")
#    mask = T.bvector("mask")
    sel = T.ivector("sel")
    LR = MultiLogisticRegression(x, n_in = feats, n_out = 2, sel = sel) 
    
    prediction = LR.y_pred
   
    cost = LR.negative_log_likelihood(y) + 0.01 * LR.paramL2# The cost to minimize
    gparams = T.grad(cost, LR.params)             # Compute the gradient of the cost
    debug = theano.function([x,sel], LR.pp, on_unused_input = 'warn', 
                            allow_input_downcast = True)
    # Compile
    updates = [(param, param - 0.1 * gparam)  
               for param, gparam in zip(LR.params, gparams)]
    
    train = theano.function(
              inputs = [x,y,sel],
              outputs = [prediction, LR.errors(y), LR.negative_log_likelihood(y)],
              updates = updates,
              allow_input_downcast = True)
    predict = theano.function(inputs = [x], outputs = prediction)
    test = theano.function(inputs = [x,y,sel], 
                           outputs = LR.negative_log_likelihood(y))
    
#    print debug(D[0],D[1],mask)[:10,:]
    sel = np.asarray(range(5,400))
#    print sel
#    print debug(D[0],D[1],sel)[:10,:]
#    print debug(D[0],D[1],sel).shape
    
    for i in range(training_steps):
        pred, error, nll = train(D[0], D[1], sel)
#        print nll
	
#    print("Final model:")
#    print(LR.W.get_value())
#    print(LR.b.get_value())
#    print("target values for D:")
#    print(D[1])
#    print("prediction on D:")
#    print(predict(D[0]))
   # print(D[1]==predict(D[0]))
    print error

    
class HiddenLayer(object):  
    def __init__(self, rng, input, n_in, n_out, W = None, b=None,  
                 activation=None):  
        self.input = input  
        if W is None:
            W_values = np.asarray(  
                rng.uniform(  
                    low = -np.sqrt(6. / (n_in + n_out)),  
                    high = np.sqrt(6. / (n_in + n_out)),  
                    size = (n_in, n_out)  
                ),  
                dtype = theano.config.floatX  
            )  
            if activation == theano.tensor.nnet.sigmoid:  
                W_values *= 4  
  
            W = theano.shared(value = W_values, name = 'W', borrow = True)  
  
        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)  
            b = theano.shared(value = b_values, name = 'b', borrow = True)  
  
        self.W = W  
        self.b = b  
        lin_output = T.dot(input, self.W) + self.b  
	if activation is None:
	    self.output = lin_output
	else:
	    self.output = activation(lin_output)
        self.params = [self.W, self.b]  
        self.paramL1 = abs(self.W).sum() + abs(self.b).sum()
        self.paramL2 = (self.W**2).sum() + (self.b**2).sum()

def TestHiddenLayer():
    N = 400                                   # training sample size
    feats = 784                               # number of input variables
    rng = np.random
    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randn(N))
    x = T.dmatrix("x")
    y = T.dvector("y")
    HL = HiddenLayer(rng, x, n_in = feats, n_out = 3, activation = T.tanh)
    h = theano.function([x], HL.output, on_unused_input = 'warn')
    out = h(D[0])
    print out.shape
    #params = h(D[0])
#    params = h(D[0][0,:])
#    for g in params:
#	print g.shape

if __name__ == "__main__":
    #TestLogisticRegression()
    TestMultiLogisticRegression()
