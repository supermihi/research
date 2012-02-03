'''
Created on 14.09.2011

@author: helmling
'''
import collections
from fractions import Fraction
    
def evaluateOutput(file):
    counter = collections.Counter()
    iteration = 0
    for line in file:
        l = line.decode('utf-8').strip()
        if len(l) > 0 and l[0] == '1':
            iteration += 1
            val = 0
            for part in l.split()[1:]:
                val += float(Fraction(part))**2
            val = 1./val
            counter[round(val,8)] += 1
            if iteration % 10000 == 0:
                print('[iteration {0}]'.format(iteration))
    print('total number of pseudocodewords: {0}'.format(iteration))
    return counter

if __name__ == '__main__':
    pass