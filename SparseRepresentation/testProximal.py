#encoding:utf-8
'''
Created on 2014/6/28

@author: bhchen
'''
import spams
import numpy as np
param = {'numThreads' : -1,'verbose' : True,'lambda1' : 0.1 }

m = 100;n = 1000
U = np.asfortranarray(np.random.normal(size = (m,n)))
# test L0
print 'U:'
print 
print "\nprox l0"
param['regul'] = 'l0'
param['pos'] = False # false by default
param['intercept'] = False # false by default
alpha = spams.proximalFlat(U,False,**param)

print alpha

print "\nprox mixed norm l1/l2 + l1"
param['regul'] = 'l1l2+l1'
param['lambda2'] = 0.1
alpha = spams.proximalFlat(U,**param)
print alpha