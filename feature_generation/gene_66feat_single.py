#!/home/ygao/anaconda2/bin/python2.7
###### Used to generate 66feat file for a target from its TGT file

import os,sys
import numpy as np

if len(sys.argv) != 4:
    print 'usage:<1>targetFasta <2>tgt_file <3>feat_file'
    sys.exit();

fasta_file = sys.argv[1]
target = fasta_file[-11:-6]
tgt_file = sys.argv[2]
feat_file = sys.argv[3]

AA1Coding=[0,4,3,6,13,7,8,9,11,10,12,2,14,5,1,15,16,19,17,18,20]

gonnet = np.array(
[ [ 1.7378,  0.870964,0.933254,0.933254, 1.12202,  0.954993, 1,        1.12202,  0.831764, 0.831764,  0.758578, 0.912011, 0.851138, 0.588844, 1.07152,  1.28825,  1.14815,   0.436516,  0.60256,  1.02329],
  [ 0.870964,2.95121, 1.07152, 0.933254, 0.60256,  1.41254,  1.09648,  0.794328, 1.14815,  0.57544,   0.60256,  1.86209,  0.676083, 0.47863,  0.812831, 0.954993, 0.954993,  0.691831,  0.660693, 0.630957],
  [ 0.933254,1.07152, 2.39883, 1.65959,  0.660693, 1.1749,   1.23027,  1.09648,  1.31826,  0.524807,  0.501187, 1.20226,  0.60256,  0.489779, 0.812831, 1.23027,  1.12202,   0.436516,  0.724436, 0.60256],
  [ 0.933254,0.933254,1.65959, 2.95121,  0.47863,  1.23027,  1.86209,  1.02329,  1.09648,  0.416869,  0.398107, 1.12202,  0.501187, 0.354813, 0.851138, 1.12202,  1,         0.301995,  0.524807, 0.512861],
  [ 1.12202, 0.60256, 0.660693,0.47863, 14.1254,   0.57544,  0.501187, 0.630957, 0.74131,  0.776247,  0.707946, 0.524807, 0.812831, 0.831764, 0.489779, 1.02329,  0.891251,  0.794328,  0.891251, 1],
  [ 0.954993,1.41254, 1.1749,  1.23027,  0.57544,  1.86209,  1.47911,  0.794328, 1.31826,  0.645654,  0.691831, 1.41254,  0.794328, 0.549541, 0.954993, 1.04713,  1,         0.537032,  0.676083, 0.707946],
  [ 1,       1.09648, 1.23027, 1.86209,  0.501187, 1.47911,  2.29087,  0.831764, 1.09648,  0.537032,  0.524807, 1.31826,  0.630957, 0.40738,  0.891251, 1.04713,  0.977237,  0.371535,  0.537032, 0.645654],
  [ 1.12202, 0.794328,1.09648, 1.02329,  0.630957, 0.794328, 0.831764, 4.57088,  0.724436, 0.354813,  0.363078, 0.776247, 0.446684, 0.301995, 0.691831, 1.09648,  0.776247,  0.398107,  0.398107, 0.467735],
  [ 0.831764,1.14815, 1.31826, 1.09648,  0.74131,  1.31826,  1.09648,  0.724436, 3.98107,  0.60256,   0.645654, 1.14815,  0.74131,  0.977237, 0.776247, 0.954993, 0.933254,  0.831764,  1.65959,  0.630957],
  [ 0.831764,0.57544, 0.524807,0.416869, 0.776247, 0.645654, 0.537032, 0.354813, 0.60256,  2.51189,   1.90546,  0.616595, 1.77828,  1.25893,  0.549541, 0.660693, 0.870964,  0.660693,  0.851138, 2.04174],
  [ 0.758578,0.60256, 0.501187,0.398107, 0.707946, 0.691831, 0.524807, 0.363078, 0.645654, 1.90546,   2.51189,  0.616595, 1.90546,  1.58489,  0.588844, 0.616595, 0.74131,   0.851138,  1,        1.51356],
  [ 0.912011,1.86209, 1.20226, 1.12202,  0.524807, 1.41254,  1.31826,  0.776247, 1.14815,  0.616595,  0.616595, 2.0893,   0.724436, 0.467735, 0.870964, 1.02329,  1.02329,   0.446684,  0.616595, 0.676083],
  [ 0.851138,0.676083,0.60256, 0.501187, 0.812831, 0.794328, 0.630957, 0.446684, 0.74131,  1.77828,   1.90546,  0.724436, 2.69153,  1.44544,  0.57544,  0.724436, 0.870964,  0.794328,  0.954993, 1.44544],
  [ 0.588844,0.47863, 0.489779,0.354813, 0.831764, 0.549541, 0.40738,  0.301995, 0.977237, 1.25893,   1.58489,  0.467735, 1.44544,  5.01187,  0.416869, 0.524807, 0.60256,   2.29087,   3.23594,  1.02329],
  [ 1.07152, 0.812831,0.812831,0.851138, 0.489779, 0.954993, 0.891251, 0.691831, 0.776247, 0.549541,  0.588844, 0.870964, 0.57544,  0.416869, 5.7544,   1.09648,  1.02329,   0.316228,  0.489779, 0.660693],
  [ 1.28825, 0.954993,1.23027, 1.12202,  1.02329,  1.04713,  1.04713,  1.09648,  0.954993, 0.660693,  0.616595, 1.02329,  0.724436, 0.524807, 1.09648,  1.65959,  1.41254,   0.467735,  0.645654, 0.794328],
  [ 1.14815, 0.954993,1.12202, 1,        0.891251, 1,        0.977237, 0.776247, 0.933254, 0.870964,  0.74131,  1.02329,  0.870964, 0.60256,  1.02329,  1.41254,  1.77828,   0.446684,  0.645654, 1],
  [ 0.436516,0.691831,0.436516,0.301995, 0.794328, 0.537032, 0.371535, 0.398107, 0.831764, 0.660693,  0.851138, 0.446684, 0.794328, 2.29087,  0.316228, 0.467735, 0.446684, 26.3027,    2.5704,   0.549541],
  [ 0.60256, 0.660693,0.724436,0.524807, 0.891251, 0.676083, 0.537032, 0.398107, 1.65959,  0.851138,  1,        0.616595, 0.954993, 3.23594,  0.489779, 0.645654, 0.645654,  2.5704,    6.0256,   0.776247],
  [ 1.02329, 0.630957,0.60256, 0.512861, 1,        0.707946, 0.645654, 0.467735, 0.630957, 2.04174,   1.51356,  0.676083, 1.44544,  1.02329,  0.660693, 0.794328, 1,         0.549541,  0.776247, 2.18776] ] )

def geneHMM(lines, angIDs):
    HMMs = []
    HMMNull=[3706,5728,4211,4064,4839,3729,4763,4308,4069,3323,5509,4640,4464,4937,4285,4423,3815,3783,6325,4665,0]
    for r in range(len(lines)):
	line = lines[r].strip()
	if line.startswith('//////////// Original HHM file'):

	    #for i in range(1):
	    for i in range(len(angIDs)):
		ID = 3*(int(angIDs[i])-1) + r + 6
		segs = lines[ID].strip().split()
		HMM = []
		if len(segs) != 23:
		    print 'HMM file fault!'
		    exit()
		else:
		    Emi = []
		    EmiProb = []
		    for k in range(20):
			if segs[k+2] == '*':
			    Emi.append(-99999.0)
			else:
   			    Emi.append(-float(segs[k+2]))
			EmiProb.append(2**(Emi[k]/1000))
		#print EmiProb
		segs = lines[ID+1].strip().split()
		NEFF = float(segs[7])
		EmiScore = np.zeros(20)
		#for j in range(1):
		for j in range(20):
		    g = 0
		    for k in range(20):
		        g += EmiProb[k]*gonnet[AA1Coding[k],AA1Coding[j]]*(2**(-HMMNull[j]/1000.0))
		    EmiScore[j] = ((NEFF/1000-1)*EmiProb[j]+g*10) / (NEFF/1000+9)
		HMMs.append(EmiScore)
	    break
    return HMMs

def genePSM(lines, angIDs):
    PSMs = []
    for r in range(len(lines)):
	line = lines[r].strip()
	if line.startswith('//////////// Original PSM file'):
	    #for i in range(1):
	    for i in range(len(angIDs)):
		ID = int(angIDs[i]) + r + 1
		segs = lines[ID].strip().split()
		#print segs
		PSM = []
		if len(segs) != 20:
		    print 'PSM file fault!'
		    exit()
		else:
	    	    for k in range(20):
	    	        score = 1.0 / ( 1 + np.exp(-float(segs[k])*0.01) )
	    	        PSM.append(score)
		    #print PSM
		PSMs.append(PSM)
	    break
    return PSMs
    

def aa2ind(c):
    inds = [1, -1, 5, 4, 7, 14, 8, 9, 10, -1, 12, 11, 13,
	    3, -1, 15, 6, 2, 16, 17, -1, 20, 18, -1, 19, -1]
    return inds[ord(c)-65]

def geneAMI(seq, angIDs):
    aainds = []
    for i in range(len(angIDs)):
        aaind = np.zeros(20)
	ID = int(angIDs[i]) - 1
	pos = aa2ind(seq[ID]) - 1
	aaind[pos] = 1
	aainds.append(aaind)
    return aainds

def geneACCSSE(lines, angIDs):
    ACCSSE = []
    for r in range(len(lines)):
	line = lines[r].strip()
	if line.startswith('//////////// Original SS3+SS8+AC'):
	    for i in range(len(angIDs)):
		ID = int(angIDs[i]) + r + 2
		segs = lines[ID].strip().split()
		SSE = [ float(segs[0]), float(segs[1]), float(segs[2]) ]
		ACC = [ float(segs[11]), float(segs[12]), float(segs[13]) ]
		ACC.extend(SSE)
		ACCSSE.append(ACC)
	    break
    return ACCSSE

#for target in targets:
f = open(fasta_file, 'r')
lines = f.readlines()
f.close()
seq = lines[1].strip()
angIDs = range(1, len(seq)+1)

fout = open(feat_file, 'w')

ftgt = open(tgt_file, 'r')
tgts = ftgt.readlines()
ftgt.close()

####### generate hmm feature(hhblits)
HMMs = geneHMM(tgts, angIDs)

####### generate PSM feature(PSSM)
PSMs = genePSM(tgts, angIDs)

####### generate seq feature
segs = tgts[3].split()
seq = segs[2]
aainds = geneAMI(seq, angIDs)

####### generate ACC and SSE
ACCSSEs = geneACCSSE(tgts, angIDs)

######## output to 66feat file
for i in range(len(angIDs)):
    HMM = HMMs[i]
    for k in range(20):
        fout.write('%f ' % HMM[k])
    PSM = PSMs[i]
    for k in range(20):
        fout.write('%f ' % PSM[k])
    aaind = aainds[i]
    for k in range(20):
        fout.write('%d ' % aaind[k])
    ACCSSE = ACCSSEs[i]
    for k in range(len(ACCSSE)):
        fout.write('%f ' % ACCSSE[k])
    fout.write('\n')
fout.close()
