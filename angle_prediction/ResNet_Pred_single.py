# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:11:14 2017

ResNet Angle Prediction for Single target (without split files or label files)

@author: ygao
"""

import numpy as np
import theano
import theano.tensor as T
import os,sys
import cPickle

def LoadModel(model_file):
    if os.path.isfile(model_file):
        f = open(model_file, 'rb')
        chkpoint = cPickle.load(f)
        f.close()
    else:
        print 'Please provide a valid model path!'
        sys.exit(-1)
    return chkpoint

## Build the ResNet model and define function to calculate marginal probability
def ResNet_Build(modelSpecs, chkpoint, n_feat):
    from elements import MultiLogisticRegression
    from MyResNet import ResNet
    rng = np.random.RandomState()

    n_layers, n_nodes, halfWinSize, reg_fac, bestParams, bestOtherParams = chkpoint
    print 'PREDICTING...', 'n_layers=', n_layers, 
    print 'halfWinSize=', halfWinSize, 'n_nodes=', n_nodes

    x = T.tensor3('x', dtype=theano.config.floatX)   
    y = T.ivector('y') 
   # mask = T.bmatrix('mask')
   # sel = T.ivector('sel')
    K = modelSpecs['K']
    
    net = ResNet(rng, input = x, n_in = n_feat, halfWinSize = halfWinSize, 
                 n_hiddens = [n_nodes for i in range(n_layers)], 
                 n_repeats = [0 for i in range(n_layers)], 
                 activation = modelSpecs['activation'], 
                 batchNorm = modelSpecs['batchNorm'])
    classifier = MultiLogisticRegression(input = net.output, n_in = n_nodes, 
                                         n_out = K)
    params = classifier.params + net.params
    for param, value in zip(params, bestParams):
        param.set_value(value)
    calMAP = theano.function(inputs = [x], outputs = classifier.p_y_given_x, 
                             on_unused_input = 'ignore', allow_input_downcast = True) 
    return calMAP

## Predict marginal probability for testData
def ResNet_Pred(modelSpecs=None, testData=None, model_file=None):
    from util import InteSeqData

    if modelSpecs is None:
        print 'Please provide a model specification for training'
        sys.exit(-1)
    if model_file is None:
        print 'Please specify a model path to do prediction'
        sys.exit(-1)
    ## Prepare testData
    if testData is None:
        print 'Please provide test data to do prediction'
        sys.exit(-1)
    else:
        feat_test = testData
#                                                                 split_test, index)
    ## Build Model
    n_feat = feat_test.shape[1] # num of feats
    chkpoint = LoadModel(model_file)
    
    calMAP = ResNet_Build(modelSpecs, chkpoint, n_feat)
    
    ## Testing on testData
    test_feat = feat_test[np.newaxis, ...]
    #test_mask = None
    pp = calMAP(test_feat)
    nrow, ncol = pp.shape
    assert ncol == modelSpecs['K']

    return pp

import getopt

def Usage():
    print 'python ResNet_Pred.py -m model_file  -p testfile  -o outfile'

    print '-m: specify a model path to output or load model parameters'
    print '-p: specify files containing data to be predicted'
    print '-o: specify file path for predicted marginal probability'

def main(argv):
    from util import LoadFiles
    modelSpecs = dict()

    testFile = None
    modelfile = None

    modelSpecs['K'] = 20
    modelSpecs['activation'] = T.nnet.relu
    modelSpecs['batchNorm'] = True

    try:
        opts, args = getopt.getopt(
            argv,"m:p:o:",["model_file=", "testfile=", "outfile="])
        print opts, args
    except getopt.GetoptError:
        Usage()
        sys.exit(-1)

    if len(opts) < 3:
        Usage()
        sys.exit('Wrong number of input parameters!')

    for opt, arg in opts:
        if opt in ("-m", "--modelfile"):
            modelfile = arg

        elif opt in ("-p", "--testfile"):
            testFile = [ f.strip() for f in arg.split(',') ]

        elif opt in ("-o", "--outfile"):
            outfile = arg

        else:
            sys.exit('Undefined input parameters!')

    if modelfile is None:
	print "Please specify the model_file path!"
	sys.exit()
    if testFile is None:
	print "Please specify the file path to test!"
	sys.exit()
    if outfile is None:
	print "Please specify the file path to output!"
	sys.exit()

    if True:
	if testFile is not None:
            testData = LoadFiles(testFile)
	    print '#testData with', testData.shape[0]
	else:
	    testData = None
        ## Calculate the marginal probability matrix for testData
        pp = ResNet_Pred(modelSpecs = modelSpecs, 
                         testData = testData, 
                         model_file = modelfile)
        ## Save marginal probability matrix to file
        nrow, K = pp.shape
        fout = open(outfile, 'w')
        for i in range(nrow):
            for r in range(K):
                #fout.write('%f\t' % tranpp[i,r])
                fout.write('%f\t' % pp[i,r])
            fout.write('\n')
        fout.close()

if __name__ == "__main__":
    #sys.setrecursionlimit(10000) 
    recursionlimit = sys.getrecursionlimit()
    #print 'recursionlimit = ', recursionlimit
    sys.setrecursionlimit(3*recursionlimit)  # to solve Maximum recursion depth problem
    print 'recursionlimit = ', sys.getrecursionlimit()
    main(sys.argv[1:])
