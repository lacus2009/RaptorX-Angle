#!/home/ygao/anaconda2/bin/python2.7
###### Used to generate predicted angles from the combination of KMeans and predicted marginal probability

import os,sys
import numpy as np
from sklearn.externals import joblib

if len(sys.argv) != 7:
    print 'usage:<1>MAP_path <2>k_cents_file  <3>k_vars_file <4>fasta_file <5>std2err_model <6>out_dir'
    sys.exit();

MAP = np.loadtxt(sys.argv[1])
npre, K = MAP.shape

data = np.loadtxt(sys.argv[2])
k_cents = data[:,3:]

k_vars = np.loadtxt(sys.argv[3])

fasta_file = sys.argv[4]
target = fasta_file[-11:-6]

reg = joblib.load(sys.argv[5])

def tran(tri):
    if len(tri) != 2:
        print 'Input must be (cos,sin)!'
    else:
        theta = np.arctan(tri[1]/tri[0])
    if tri[0] >= 0:
        ang = theta
    elif tri[0] < 0 and tri[1] > 0:
        ang = theta + np.pi
    else:
        ang = theta - np.pi
    return ang

f = open(fasta_file, 'r')
lines = f.readlines()
seq = lines[1].strip()
f.close()

out_path = sys.argv[6] + '/' + target + '.pred'
fout = open(out_path, 'w')
fout.write('Site\tRes\tPrePhi\t\tPrePsi\tPreStdErr\tPreStdErr\n')

for i in range(len(seq)):
    # Get the certain residue
    res = seq[i]

    # Generate predicted angles
    PrePhiVec = np.zeros(2)
    PrePsiVec = np.zeros(2)
    for r in range(K):
        PrePhiVec += MAP[i,r] * k_cents[r,:2]
        PrePsiVec += MAP[i,r] * k_cents[r,2:]
    PrePhi = tran(PrePhiVec)/np.pi*180
    PrePsi = tran(PrePsiVec)/np.pi*180

    # Calculate intuitive variance
    PrePhiVar = 0.
    PrePsiVar = 0.
    for r in range(K):
        PrePhiVar += MAP[i,r] * k_vars[r,0]
        PrePsiVar += MAP[i,r] * k_vars[r,1]
    PrePhiStd = np.sqrt(PrePhiVar)
    PrePsiStd = np.sqrt(PrePsiVar)
    test = np.array([PrePhiStd, PrePsiStd])
    pred = reg.predict(test[:, np.newaxis])
    
    PrePhiErr = pred[0]
    PrePsiErr = pred[1]

    fout.write('%d\t%c\t%f\t%f\t%f\t%f\n' % (i+1, res, PrePhi, PrePsi, PrePhiErr, PrePsiErr) )
fout.close()

