
"""
The MIT License (MIT)

Copyright (c) 2015 Alec Radford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""  

# with only Adam used (recording other related shared variables)

import theano
import theano.tensor as T
import numpy as np
  
def Adam(cost, params, lr = 0.0002, b1 = 0.1, b2 = 0.001, e = 1e-8):
    updates = []
    other_params = []

    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.).astype(theano.config.floatX))
    i_t = i + 1.

    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
	other_params.append(m)
	other_params.append(v)

        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates, other_params


# Please ignore the followings
def Adam_ori(cost, params, lr = 0.0002, b1 = 0.1, b2 = 0.001, e = 1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.).astype(theano.config.floatX))
    i_t = i + 1.

    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)

        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates

def TestAdam_ori():
    from elements import LogisticRegression
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
    # The cost to minimize
    cost = LR.negative_log_likelihood(y) + 0.01 * LR.paramL2
    updates = Adam(cost, LR.params)     # Compute the gradient of the cost
    
    # Compile
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
    print(D[1]==predict(D[0]))


if __name__ == "__main__":
    TestAdam_ori()
