#!/usr/bin/env python

import os,sys
import numpy as np

if len(sys.argv) != 3:
    print 'usage: <1>list_of_MAP <2>out_MAP'
    sys.exit()



################## Simple Mean Method
def TakeMean(lines):
    MAP = 0.
    for i in range(len(lines)):
        line = lines[i].strip()
        temp = np.loadtxt(line)
        MAP += temp
    MAP /= len(lines)
#    print MAP.shape
#    print len(lines)
    return MAP


def main(argv):
    f = open(argv[1], 'r')
    lines = f.readlines()
    f.close()
    MAP = TakeMean(lines)
    fout = open(argv[2], 'w')
    np.savetxt(fout, MAP)
    fout.close()

if __name__ == '__main__':
    main(sys.argv)
