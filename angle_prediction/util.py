
import sys
import numpy as np
import theano
import theano.tensor as T
import os
rng = np.random

def LoadFiles(files):
  # Used to load feat_file,label_file
    if len(files) != 1:
        print 'Wrong number of files to be loaded!'
        sys.exit()
    else:
        feat = np.loadtxt(files[0])
#        label = np.int_(np.loadtxt(files[1]))
#        split = np.int_(np.loadtxt(files[2]))
#        assert len(label) == feat.shape[0]
    return feat
    #return feat, label, split

def InteSeqData(data, label, split, index):
    """
    Used to generate sequence-based data, i.e., each sequence is a sample
    abs_feat: (n_targets, Lmax, n_feat)
    labels: (n_targets * Lmax, )
    mask: (n_targets, Lmax-Lmin)
    sel: (n_sites, )
    
    """    
    total_site, n_feat = data.shape
    Lmax = split[index].max()
    Lmin = split[index].min()
    abs_feat = np.zeros((len(index), Lmax, n_feat))
    abs_label = np.zeros((len(index), Lmax)) - 1
    abs_label = np.int_(abs_label)
    count = 0
    for idx in index:
	begin = 0
	if idx > 0:
	    begin = split[:idx].sum()	
	end = split[:(idx+1)].sum()
	#print begin, end
	pos = Lmax - split[idx]
	for r in range(begin, end):
	    abs_feat[count, pos, :] = data[r, :]
	    abs_label[count, pos] = label[r]
	    pos += 1
	count += 1
    labels = abs_label.reshape(-1)
    mask = np.zeros(shape = (len(index), Lmax-Lmin), dtype = np.int8)
    for j in xrange(len(index)):
   	seqLen = split[index[j]] 
        mask[j, Lmax - seqLen : ].fill(1)
    sel = []
    for i in range(len(labels)):
	if labels[i] != -1:
	    sel.append(i)
    sel = np.asarray(sel)
    return abs_feat, labels, mask, sel
    #return abs_feat, abs_label, mask, sel
    
def TestInteSeqData():

    os.chdir('/mnt/home/ygao/DeepLearning')
    feat_train = np.loadtxt('test_trainingfeat')
    label_train = np.loadtxt('test_traininglabel')
    label_train = np.int_(label_train)
    split_train = np.int_( np.loadtxt('split_train') )
    
    index = [0,1]
    abs_feat, abs_label, mask, sel = InteSeqData(feat_train, label_train, split_train, index)
    print split_train[index]
#    print abs_feat[1, 324:327, :5]
#    #print abs_feat[1, 593:600, :5]
#    print feat_train[427:430, :5]
#    print abs_feat.shape
    print abs_label.shape
    print mask.shape
    print len(sel)

if __name__ == "__main__":
    #TestInteData()
    TestInteSeqData()
