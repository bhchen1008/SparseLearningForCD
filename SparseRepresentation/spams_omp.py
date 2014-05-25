'''
Created on 2014/3/12

@author: bhchen
'''
import spams
import numpy as np
import time
np.random.seed(0)
print 'test omp'
X = np.asfortranarray(np.random.normal(size=(64,100000)),dtype= float)
D1 = np.asfortranarray(np.random.normal(size=(64,200)))
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)),dtype= float)
L = 10
eps = 1.0
numThreads = -1
tic = time.time()
alpha = spams.omp(X,D1,L=L,eps= eps,return_reg_path = False,numThreads = numThreads)
print alpha
tac = time.time()
t = tac - tic
print "%f signals processed per second\n" %(float(X.shape[1]) / t)
########################################
# Regularization path of a single signal 
########################################
X = np.asfortranarray(np.random.normal(size=(64,1)),dtype= float)
D1 = np.asfortranarray(np.random.normal(size=(64,10)))
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)),dtype= float)
L = 5
(alpha,path) = spams.omp(X,D1,L=L,eps= eps,return_reg_path = True,numThreads = numThreads)