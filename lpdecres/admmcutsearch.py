# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:25:30 2014

@author: karunia
"""

from __future__ import division, print_function
from docutils.nodes import thead
import numpy as np
from numpy import maximum, minimum
from numpy import linalg as LA
from lpdec.decoders import Decoder


class ADMMDecoder(Decoder):
    
    def __init__(self, code, name="ADMM"):
        Decoder.__init__(self, code, name)
        
    def solve(self, lb=-np.inf, ub=np.inf):
        H       = self.code.parityCheckMatrix
        M, N    = H.shape
        gamma   = self.llrs
        Nj      = [np.flatnonzero(row) for row in H] # ask
        #print "Nj = ", Nj
        dj      = [nj.size for nj in Nj]
        Nv      = [np.flatnonzero(col) for col in np.transpose(H)]
        #print "Nv = ", Nv
        cj      = [nv.size for nv in Nv]
        #print "cj = ", cj
        #print "dj = ", dj
        #print "H = ", H
        Pj      = [] # list of Pj's
        zj      = []
        yj      = []
        zj_new  = []
        yj_new  = []
        d       = np.mean((dj))
        for j in range(M):
            P            = np.zeros((dj[j], N))
            P[:,Nj[j]] = np.eye(dj[j])
            Pj.append(P)
            zj.append(np.zeros(dj[j]))
            yj.append(np.zeros(dj[j]))
            zj_new.append(np.zeros(dj[j]))
            yj_new.append(np.zeros(dj[j]))            
        rho     = 3
        eps     = 1e-10
        eps2    = 1e-10
        x       = np.zeros(N)
        for i in range(N):
            if gamma[i] < 0:
                x[i] = 1        
        k       = 0 
        #normx   = np.zeros(202)
        #normx[0]= LA.norm(x)
        while True:
            # print "Step = ", k+1
            for j in range(M):
                w          = np.dot(Pj[j],x) + yj[j]
                zj_new[j]  = self.projtestCSA(w)
                yj_new[j]  = w - zj_new[j]
            zjj      = self.conv(Pj,zj_new,M) # call function conv
            yjj      = self.conv(Pj,yj_new,M) # call function conv
            #print "zjj = ", zjj
            x_new    = np.zeros(N)
            for i in range(N):
                x_new[i] = (1/cj[i])*(sum((zjj[j][i] - yjj[j][i]) for j in Nv[i]) - (1/rho)*gamma[i])
            #print "x = ", x_new
            #normx[k+1] = LA.norm(x_new)
            check1   = sum((LA.norm(np.dot(Pj[j],x_new) - zj_new[j])**2 for j in range(M)))
            check2   = sum((LA.norm(zj_new[j] - zj[j])**2 for j in range(M)))
            
            if (check1 < eps) and (check2 < eps2):
                break
            zj                 = zj_new
            yj                 = yj_new
            x                  = x_new
            if k == 500:
                break
            k                  = k+1
        #self.check    = normx
        self.solution = x_new
        self.objectiveValue = np.dot(gamma,x_new)
        
    def conv(self, Pj, zj, M):
        z  = []
        for j in range(M):
            zk = np.dot(np.transpose(Pj[j]),np.transpose(zj[j]))
            z.append(zk)
        return z
              
    def projtestCSA(self, u):
        z = maximum(0, minimum(1, u))
        theta, logic = self.csa(z)
        if logic == 1:
            nu = self.opt(u,theta)
            res = maximum(0,minimum(1,u-nu*theta))
            return res
        else:
            res = z
            return res

    @staticmethod
    def csa(u):
        theta = np.zeros(np.size(u))
        for i in range(np.size(u)):
            if u[i] > 0.5:
                theta[i] = 1
            else:
                theta[i] = -1
        if np.size(np.flatnonzero(theta > 0)) % 2 == 0:
            j = np.argmin(np.abs(.5-u), axis=None)
            theta[j] = -theta[j]
        if np.dot(theta,u) > np.size(np.flatnonzero(theta > 0))-1:
            logic = 1
            return theta, logic
        else:
            logic = 0
            theta = 0
            return theta, logic

    @staticmethod
    def opt(u,theta):
        j  = 0
        Tj = np.zeros(np.size(u))
        for i in range(np.size(u)):
            if u[i] > 1:
                Tj[j] = u[i] - 1
                j     = j + 1
            elif u[i] < 0:
                Tj[j] = -u[i]
                j     = j + 1
        T = Tj[0:j]
        delta = np.dot(theta,u) - np.size(np.flatnonzero(theta > 0)) + 1
        zeta  = np.size(theta)
        if T.size != 0:
            T = np.sort(T)[::-1]
            for ti in T:
                if delta/zeta > ti:
                    nu = delta/zeta
                    return nu
                else:
                    delta = delta - ti
                    zeta  = zeta - 1
        nu = delta/zeta
        return nu
        
        
if __name__ == "__main__":
    from lpdec.codes.classic import HammingCode
    from lpdec.codes.ldpc import ArrayLDPCCode
    from lpdec.channels import AWGNC
    code = HammingCode(3)
    #code = ArrayLDPCCode(q=7, m=4)
    decoder = ADMMDecoder(code, name="Test Decoder")
    channel = AWGNC(1, code.rate, seed=12565)
    sig = channel.signalGenerator(code, wordSeed=1)
    #u = np.array([-1,2,2,1,-1,0])
    #print "z = ", decoder.proj(u)
    signal = next(sig)
    decoder.decode(signal)
    print(decoder.objectiveValue)
    print(decoder.solution)
    H      = code.parityCheckMatrix
    print("solution x = ", decoder.solution)
    print("objective value f = ", decoder.objectiveValue)
    print("Hx = ", np.dot(H,decoder.solution))
    print("signal = ", signal)

    u = np.array([-1.029000755483433, 0.206512231401114, 1.341111305902483, 1.332716574126045])
    u2 = u.copy()
    u2[0] = 0
    u2[2] = u2[3] = 1
    theta = ADMMDecoder.csa(u2)[0]
    print(theta)
    y = ADMMDecoder.opt(u, theta)
    print(y)
    xstar = maximum(0, minimum(1, u-y*theta))
    print(np.dot(xstar, theta))
    wrongtheta = np.array([-1, -1, -1, 1])
    print(np.dot(theta, u))
    print(np.dot(wrongtheta, u))
    #normx = decoder.check
    #plt.plot(range(np.size(normx)),normx,'r')
        