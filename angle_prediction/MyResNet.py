import numpy as np
import theano
import theano.tensor as T

"""
abstract from Jinbo's ResNet.py 
"""

##a new implementation of 1D convolution 
class ResConv1DLayer(object):

    def __init__(self, rng, input, n_in = 0, n_out = 0, 
                 halfWinSize = 0, activation = T.nnet.relu, mask = None):
        """ 
        The input has shape (batchSize, n_in, seqLen)
	The output will have shape (batchSize, n_out, seqLen)
	mask has shape (batchSize, #positions_to_be_masked)
	#positions_to_be_masked = maxLen - minLen
        """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
	self.halfWinSize = halfWinSize

        windowSize = 2*halfWinSize + 1
        self.filter_size = windowSize

        # reshape input to shape (batchSize, n_in, nRows=1, nCols=seqLen) 
        in4conv2D = input.dimshuffle(0, 1, 'x', 2)

        # initialize the filter
        w_shp = (n_out, n_in, 1, windowSize)
	if activation == T.nnet.relu:
            W_values = np.asarray(
                rng.normal(scale = np.sqrt(2. / (n_in*windowSize + n_out)),
                           size = w_shp), 
                dtype = theano.config.floatX )
	else:
            W_values = np.asarray(
                rng.uniform(low = - np.sqrt(6. / (n_in*windowSize + n_out)), 
                            high = np.sqrt(6. / (n_in*windowSize + n_out)), 
                            size = w_shp),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
            	W_values *= 4

        self.W = theano.shared(value=W_values, name='ResConv1d_W', borrow=True)

        b_shp = (n_out,)
        self.b = theano.shared(
            np.asarray(rng.uniform(low = -.0, high = .0, size = b_shp), 
                       dtype=input.dtype), 
            name ='ResConv1d_b', 
            borrow=True)

        # conv_out and conv_out_bias have shape (batch_size, n_out, 1, nCols)
        conv_out = T.nnet.conv2d(in4conv2D, self.W, 
                                 filter_shape=w_shp, border_mode='half')
        if activation is not None:
            conv_out_bias = activation(conv_out + 
                                       self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            conv_out_bias = (conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

	## out2 has shape (batchSize, n_out, nCols)
        out2 = conv_out_bias.dimshuffle(0, 1, 3, 2)[:, :, :, 0]

        if mask is not None:
            ## since we did zero padding at left side of the input tensor
            ## we need to reset these positions to 0 again after convolution 
            ## to avoid introducing noise
            ## mask has shape (batchSize, #positions_to_be_masked)

            ##take the subtensor of out2 that needs modification
            out2_sub = out2[:, :, :mask.shape[1] ]
            mask_new = mask.dimshuffle(0, 'x', 1)
            self.output = T.set_subtensor(out2_sub, T.mul(out2_sub, mask_new))
        else:
            self.output = out2

	##self.output has shape (batchSize, n_out, nCols)

        # parameters of the model
        self.params=[self.W, self.b]

        self.paramL1 = abs(self.W).sum() + abs(self.b).sum()
        self.paramL2 = (self.W**2).sum() + (self.b**2).sum()


##note that here we do not consider mask, so the estimated mean and variance 
##may not be very accurate when there are too many padding zeros in x
def batch_norm(x, n_in, eps = 1e-6):

    ## x shall have shape (batchSize, n_in, nCols) 
    ## or (batchSize, n_in, nRows, nCols)
    gamma = theano.shared(
        np.asarray(np.ones((n_in,)), dtype = theano.config.floatX), 
        borrow = True)
    bias = theano.shared(
        np.asarray(np.zeros((n_in,)), dtype = theano.config.floatX), 
        borrow = True)
    if x.ndim == 4:
        x_mean = x.mean(axis = [0,2,3], keepdims = True)
        x_std = (x.var(axis = [0,2,3], keepdims = True) + eps ).sqrt()
        y = T.nnet.bn.batch_normalization(x, gamma[None,:,None,None], 
                                          bias[None,:,None,None], x_mean, 
                                          x_std, mode = "low_mem")

    elif x.ndim == 3:
        x_mean = x.mean(axis = [0,2], keepdims = True)
        x_std = (x.var(axis = [0,2], keepdims = True) + eps ).sqrt()
        y = T.nnet.bn.batch_normalization(x, gamma[None,:,None], 
                                          bias[None,:,None], x_mean, 
                                          x_std, mode = "low_mem")
    else:
	print 'the ndim of input for batch_norm can only be 3 or 4!'
	sys.exit(-1)

    return y, [gamma, bias]

class BatchNormLayer:
    def __init__(self, input, n_in):
        self.input = input
	self.n_in = n_in
	bnout, bnparams = batch_norm(input, n_in)
	self.output = bnout
	self.params = bnparams
	self.n_out = n_in
	self.paramL1 = abs(bnparams[0]).sum() + abs(bnparams[1]).sum()
	self.paramL2 = (bnparams[0]**2).sum() + (bnparams[1]**2).sum()



class ResBlockV2:

    def __init__(self, rng, input, n_in, halfWinSize = 0, mask = None, 
                 mask_2 = None, n_out = None, activation = T.nnet.relu, 
                 dim_inc_method = 'partial_projection', batchNorm = False, 
                 dropout = False):
	## The input has shape (batchSize, n_in, nRows, nCols) 
        ## or (batchSize, n_in, nCols)

        ## When input.ndim=4, mask and mask_2 shall be interpreted as
	## rowmask and colmask, respectively

	## If n_out is not None, then it shall be no smaller than n_in
        if n_out is not None:
            assert n_out >= n_in
            self.n_out = n_out
        else:
            self.n_out = n_in

        self.n_in = n_in
        self.input = input
        self.mask = mask
        self.halfWinSize = halfWinSize

	if input.ndim == 3:
	    ConvLayer = ResConv1DLayer
	else:
	    print 'the ndim of input can only be 3'
	    sys.exit(-1)

        if batchNorm:
	    bnlayer1 = BatchNormLayer(input, n_in)
	    input1 = activation(input)
	    if input.ndim == 3:
	        l1 = ConvLayer(rng, input = input1, n_in = n_in, n_out = self.n_out, 
                               halfWinSize = halfWinSize, mask = mask, activation = None)
	    else:
	        l1 = ConvLayer(rng, input = input1, n_in = n_in, n_out = self.n_out, 
                               halfWinSize = halfWinSize, rowmask = mask, 
                               colmask = mask_2, activation = None)

	    bnlayer2 = BatchNormLayer(l1.output, l1.n_out)
	    input2 = activation(bnlayer2.output)

	    l2 = ConvLayer(rng, input = input2, n_in = l1.n_out, n_out = self.n_out, 
                           halfWinSize = halfWinSize, mask = mask, activation = None)

	    self.layers = [bnlayer1, l1, bnlayer2, l2]
	else:
	    #input1 = input
	    input1 = activation(input)

	    l1 = ConvLayer(rng, input = input1, n_in = n_in, n_out = self.n_out, 
                           halfWinSize = halfWinSize, mask = mask, activation = None)

	    input2 = activation(l1.output)

	    l2 = ConvLayer(rng, input = input2, n_in = l1.n_out, n_out = self.n_out, 
                           halfWinSize = halfWinSize, mask = mask, activation = None)

	    self.layers = [l1, l2]

	## intermediate has shape (batchSize, n_out, nRows, nCols) 
        ## or (batchSize, n_out, nCols)
	intermediate = l2.output

	if dim_inc_method == 'full_projection':
	    ## we do 1*1 convolution here without any nonlinear transformation
	    linlayer = ConvLayer(rng, input = input, n_in = n_in, n_out = self.n_out, 
                                 halfWinSize = 0, mask = mask, activation = None)
	    intermediate = intermediate + linlayer.output
	    self.layers.append(linlayer)
	    #print 'projection is True'

	elif dim_inc_method == 'identity':
	    if input.ndim == 3:
	    	intermediate = T.inc_subtensor(intermediate[:, :n_in, :], input)
	    else:
	    	print 'the ndim of input can only be 3'
	    	sys.exit(-1)
	    print 'only projection is supported'
	    sys.exit(-1)

	elif dim_inc_method == 'partial_projection':
	    if self.n_out == n_in:
		intermediate = intermediate + input
	    else:
		linlayer = ConvLayer(rng, input = input, n_in = n_in, 
                                     n_out = self.n_out - n_in, halfWinSize=0, 
                                     mask=mask, activation=None)
		self.layers.append(linlayer)

		intermediate = (intermediate + 
                                T.concatenate([input, linlayer.output], axis=1))
	else:
	    print 'unsupported dimension increase method: ', dim_inc_method
	    sys.exit(-1)

	## self.output has shape (batchSize, n_out, nRows, nCols)
	self.output = intermediate

	self.params = []
	self.paramL1 = 0
	self.paramL2 = 0

	for layer in self.layers:
	    self.params += layer.params
	    self.paramL1 += layer.paramL1
	    self.paramL2 += layer.paramL2


class ResNet:
    def __init__(self, rng, input, n_in, halfWinSize = 0, mask = None, 
                 mask_2 = None, n_hiddens = None, n_repeats = None, 
                 activation = T.nnet.relu, dim_inc_method = 'partial_projection', 
                 batchNorm = False, batchNorm2 = False, version = 'ResNetv1'):
	"""
	This ResNet consists of the following components:
	1) a start layer with input as input, output has n_hiddens[0] features
	2) in total there are len(n_repeats) stacks
	3) each stack has 1 + n_repeats[i] blocks 
	4) the first block of each stack has n_hiddens[i-1] or n_in input features 
           and n_hiddens[i] output features
	5) the other blocks of each stack has #inputfeatures = #outputfeatures = n_hiddens[i]

	input has shape (batchSize, nRows, nCols, n_in) or (batchSize, nCols, n_in)
	output has shape (batchSize, nRows, nCols, n_hiddens[-1]) or (batchSize, nCols, n_hiddens[-1])
	n_hiddens needs to be an increasing sequence
	n_repeats shall be non-negative
	"""

	assert n_hiddens is not None
	assert n_repeats is not None
	assert len(n_hiddens) > 0
	assert len(n_hiddens) == len(n_repeats)

	if input.ndim == 3:
	    ConvLayer = ResConv1DLayer
	    input2 = input.dimshuffle(0, 2, 1)
	else:
	    print 'the ndim of input can only be 3!'
	    sys.exit(-1)

	ResBlock = ResBlockV2

	blocks = []

	startLayer = ConvLayer(rng, input = input2, n_in = n_in, 
                                   n_out = n_hiddens[0], halfWinSize = halfWinSize,
                                   mask = mask, activation = activation)

	blocks.append(startLayer)

	## repeat a block n_repeats[i] times
	curr_block = startLayer
	for j in range(n_repeats[0]):
	    assert (curr_block.n_out == n_hiddens[0])
	    new_block = ResBlock(rng, input = curr_block.output, n_in = n_hiddens[0], 
                                 mask = mask, mask_2 = mask_2, halfWinSize = halfWinSize, 
                                 activation = activation, dim_inc_method = dim_inc_method, 
                                 batchNorm = batchNorm)
	    blocks.append(new_block)
	    curr_block = new_block

	for i in range(1, len(n_hiddens)):
	    ## the start block is in charge of dimension increase
	    assert (curr_block.n_out == n_hiddens[i-1])

	    #bnflag = batchNorm
	    new_block = ResBlock(rng, input = curr_block.output, n_in = n_hiddens[i-1], 
                                 n_out = n_hiddens[i], mask = mask, mask_2 = mask_2, 
                                 halfWinSize = halfWinSize, activation = activation, 
                                 dim_inc_method = dim_inc_method, batchNorm = batchNorm)
	    blocks.append(new_block)
	    curr_block = new_block

	    ## repeat a block n_repeats[i] times
	    for j in range(n_repeats[i]):
		assert (curr_block.n_out == n_hiddens[i])
		new_block = ResBlock(rng, input = curr_block.output, n_in = n_hiddens[i], 
                                     mask = mask, mask_2 = mask_2, halfWinSize = halfWinSize, 
                                     activation = activation, dim_inc_method = dim_inc_method, 
                                     batchNorm = batchNorm)
		blocks.append(new_block)
	        curr_block = new_block

	out2 = curr_block.output
	self.n_out = curr_block.n_out

	## out2 has shape (batchSize, n_out, nRows, nCols) or (batchSize, n_out, nCols)
	## change the output shape back to (batchSize, nRows, nCols, n_out) 
        ## or (batchSize, nCols, n_out)
	if input.ndim == 3:
	    self.output = out2.dimshuffle(0, 2, 1)
	else:
	    print 'the ndim of input can only be 3!'
	    sys.exit(-1)

	self.params = []
	self.paramL1 = 0
	self.paramL2 = 0
	for block in blocks:
	    self.params += block.params
	    self.paramL1 += block.paramL1
	    self.paramL2 += block.paramL2

	self.layers = blocks


def TestResNet():
    rng = np.random.RandomState()
    n_in = 3
    n_out = 4
    nRows = 20
    m = T.fmatrix('m')
    x = T.tensor3('x')
    net = ResNet(rng, input = x, n_in = n_in, halfWinSize = 1, 
                 n_hiddens = [30, 35, 40, 45], n_repeats = [5, 1, 0, 2], 
                 activation = T.tanh, mask = m)
    #net2 = ResNet(rng, input = net1.output, n_in = 3, halfWinSize = 1, 
    #              n_hiddens = [4], n_repeats = [1], activation = T.tanh, mask = m)
    y = net.output
    f = theano.function([x, m], y, on_unused_input='warn')

    bSize = 2
    a = np.random.uniform(0, 1, (bSize, nRows, n_in))
    m_value = np.ones( (bSize, 2) )

    """
    b = f(a, m_value)
    print b
    print b.shape
    """

    paramL2 = net.paramL2

    loss = T.mean( (y)**2 )
    cost = loss + 0.001 * paramL2
    #params = net1.params + net2.params
    params = net.params 
    gparams = T.grad(cost, params)
    h = theano.function([x, m], gparams, on_unused_input = 'warn')
    #gs = h(a, m_value)

#    for g in gs:
#	print g.shape
#	#print g

    updates = [ (p, p - 0.03 * g) for p, g in zip(params, gparams) ]
    train = theano.function([x, m], [cost, loss, paramL2], updates = updates, allow_input_downcast=True)

    for i in xrange(10):
	c, los, l2 = train(a, m_value)
	print c, los, l2

if __name__ == "__main__":

    #TestConvLayers()
    TestResNet()
