
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:20:37 2013
Unterschied zu dualsimplex.py:
Setze beim Pivotieren die Komponenten des entstehenden Einheitsvektors 
manuell auf 0 bzw 1 
@author: Florian
"""
from __future__ import division
import numpy as np
from numpy import *
EPS = 1e-16

def dualboundedsimplex(A,b,c, fixed=None):
    """Apply a dual simplex for the bounded problem min cx s.t.Ax <= b, 0<=x<=u
    
    """
    # construct initial simplex tableau T
    
    m, n = A.shape
  #  print "m=", m, "n=", n
    if fixed:
        import fixedpoint as fp
        fp.setPrecision(*fixed)
        EPS = fp.fixed2float(2**(fixed[1]//3+1)-1)
        maxFp = 2**(fixed[0]-fixed[1]-1)-2**-fixed[1]
        #print(maxFp)
        if np.max(np.abs(c)) >= maxFp:
            print("scaling function by {}".format(2/3*maxFp/np.max(np.abs(c))))
            c *= 2/3*maxFp/np.max(np.abs(c))
    else:
        EPS = 1e-16
    T = np.hstack((A,np.eye(m)))
    #print "Zeile 22 T=", T
    c = np.hstack((c,np.zeros(m)))
   # print "Zeile 24 c=", c    
    T = np.vstack((c,T))
   # print "Zeile 26 T=", T    
    hilf = np.hstack((np.zeros(1),b))
    T = np.hstack((T,hilf[:,newaxis]))
   # print "Zeile 30 T=", T
   # print "shape(T)=", shape(T)
    if fixed:
        T = fp.np_float2fixed(T)   
        c = fp.np_float2fixed(c)
        #EPS = fp.fixed2float(5)
        #EPS = 0
    # initializations of indicator functions    
    
    xlabel = np.zeros(n+m) #=0 if x_i is used in the current simplex tableau
  #  print "Zeile 36 xlabel=",xlabel        #=1 if x_i^' is used in the current simplex tableau
    xbasis = np.hstack((np.zeros(n),np.ones(m))) #=1 if x_i (or x_i^') is in
   # print "Zeile 38 xbasis=",xbasis                        # the current basis, 0 else 
    basis = np.arange(n,n+m) #describes the indices of the columns of the basis
  #  print "Zeile 40 basis=",basis      # in the right order    
    
    # modify simplex tableau
   # print "Zeile 43 xlabel==",xlabel 
    for i in range(n):
        if c[i] < -EPS:
            T[:,m+n] = T[:,m+n] - T[:,i]
            T[:,i] = -T[:,i]           # change the sign of c_i and A_i
            xlabel[i] = 1 - xlabel[i]      # use x_i^' instead of x_i
    #       print "c=",c            
            c[i] =-c[i]
    
   # print "Zeile 52 T=",T
  #  print "xlabel==",xlabel    
  #  print "c=",c
    
    #Tmax = T.max() 
    #Tmin = T.min()
    #Tmaxohnez = T[0,0:n+m].max()   # maximaler Wert im Tableau ohne Wert oben
    #Tmaxohnez = max(Tmaxohnez,T[1:m+1,:].max())  # rechts (im normalen 
                                               # SimplexZielfunktionswert")
    #Tminohnez = T[0,0:n+m].min()
    #Tminohnez = min(Tminohnez,T[1:m+1,:].min()) 
    Tnonzeromax = sum(T!=0)
    Tnonzeromaxratio = floor(Tnonzeromax)/floor(size(T))
    Tminabsohnenull = min(abs(T[nonzero(T)]))
    Tabsmax = abs(T[1:m+1,:]).max()
    
    for iteration in range(1000):
        #Tmax = max(Tmax,T.max())
        #Tmin = min(Tmin,T.min())
        Tnonzeromax = max(Tnonzeromax,sum(T!=0))
        Tnonzeromaxratio = floor(Tnonzeromax)/floor(size(T))
        #Tmaxohnez = max(Tmaxohnez,T[0,0:n+m].max())
        #Tmaxohnez = max(Tmaxohnez,T[1:m+1,:].max()) 
        #Tminohnez = min(Tminohnez,T[0,0:n+m].min())
        #Tminohnez = min(Tminohnez,T[1:m+1,:].min())
        Tminabsohnenull = min(Tminabsohnenull,min(abs(T[nonzero(T)])))
        Tabsmax = max(Tabsmax,abs(T[1:m+1,:]).max())
#        print "iteration =",iteration
        #check optimality
        optimal = 1
        for i in range(1,m+1):
            if T[i,n+m] < -EPS:
                k = i
           #     print "k=",k
                optimal = 0
                break
        if optimal == 1:  #case: optimality
#            print "optimal"
            # Konstruiere Optimalloesung x            
            if fixed:
                x = fp.np_float2fixed(np.zeros(n+m))
                for index, value in enumerate(basis):
                    x[value] = T[1+index,m+n]
            else:
                x = np.zeros(n+m)
                x[basis] = T[1:m+1,m+n]
          #  print "x=",x
            for i in range(0,n):   # laeuft nur bis n-1, da Schlupfvariabeln
                if xlabel[i] == 1: #  x_n,...,x_n+m-1 nie zu x_i' werden
                    if xbasis[i] == 1:
                        x[i] = (fp.FixedPointNumber(1) if fixed else 1) - x[i]
                    else:          #xbasis[i]=0
                        x[i] = fp.FixedPointNumber(1) if fixed else 1
            z = -np.dot(x,c)
        #    print "x= (Fall optimal=1)",x
#            print "Tnonzeromax=",Tnonzeromax            
        #    print "size(T)=",size(T)
        #    print "Tnonzeromaxratio=",Tnonzeromaxratio
            solution = x[:n].astype(np.double)
            stats = {"iterations": iteration + 1, "maxabs" : Tabsmax, "minabs" : Tminabsohnenull, "maxnonzeros" : Tnonzeromaxratio}
            #result[n+m+5] = Tmaxohnez
            #result[n+m+6] = Tnonzeromax
            #result[n+m+7] = Tminabsohnenull
            #result[n+m+8] = Tnonzeromaxratio            
            #result[n+m+7] = Tnonzeromaxratio
            #result = np.hstack((x,z))
            return (z, solution, stats)
        else:             #case: not optimal
          #  print "optimal=",optimal
            #infeasibility test, 
            # test if there is no other negative entry in the i-th row in T
            
            if len([x for x in T[k,0:n+m] if x < -EPS]) == 0: #no negative entries
                return (0, -np.ones(n), {"iterations": iteration + 1, "maxabs" : Tabsmax, "minabs" : Tminabsohnenull, "maxnonzeros" : Tnonzeromaxratio} )
            else:
             #   print "problem is feasible"                
                # use dual quotient rule
                # determine minimal quotient of all negative row entries
                minarg = argmax(T[k,0:n+m]<-EPS)      #initialization
              #  print "minarg=",minarg
                minquot = -T[0,minarg]/T[k,minarg] #initialization
            #    print "minquot=", minquot
                for j in range(minarg+1,n+m):
                    if T[k,j] < 0:
                        if -T[0,j]/T[k,j] < minquot:
                            minarg = j
            #                print "minarg=",minarg
                            minquot = -T[0,j]/T[k,j]
          #                  print "minquot=", minquot
          #      print "minarg=",minarg
           #     print "minquot=", minquot
                
                # Basis exchange
                xbasis[basis[k-1]] = 0
                xbasis[minarg] = 1
           #     print "newxbasis (Zeile110)",xbasis                
                basis[k-1] = minarg
          #      print "new basis (Zeile112):",basis
                
                # pivot with respect to t_{k,minarg}                            
                for i in range(0,m+1):
                    if i != k:
                        T[i,:] = T[i,:] - T[k,:]*(T[i,minarg]/T[k,minarg])
                
                T[k,:] = T[k,:]/T[k,minarg]
                # for numerical reasons: set the minarg-th column to 
                # einheitsvector 
                T[k,minarg] = fp.FixedPointNumber(1) if fixed else 1
                for i in range(m+1):
                    if i != minarg:
                        T[i,minarg] = fp.FixedPointNumber(0) if fixed else 0
                
                # consider upper bounds u_i and modify the simplex tableau T
                for i in range(1,m+1):
                    if basis[i-1] < n:
                        if T[i,n+m] > 1:
                            for j in range(0,m+n):   #change the sign of ith row 
                                if j != basis[i-1]:  # except of t_ij=1
                                    T[i,j] = -T[i,j]
                            T[i,m+n] = (fp.FixedPointNumber(1) if fixed else 1) - T[i,m+n]
                            xlabel[basis[i-1]] = 1 - xlabel[basis[i-1]]
                            c[basis[i-1]] = -c[basis[i-1]]
                        
            #    print "new Tableau T",T
             #   print "xlabel=",xlabel
             #   print "c=",c