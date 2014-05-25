'''
Created on 2014/3/18

@author: bhchen
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np
input_f1 = open(sys.argv[1])
dict1 = input_f1.read()

dictLoader1 = aL.arffLoader()
#testContent = dictLoader1.load(dict1)
dictLoader1.load(dict1)

print dictLoader1.classIndex
print dictLoader1.attrName
print dictLoader1.transactionList[0]
print dictLoader1.numInstance
D1 = dictLoader1.fortranArray(dictLoader1.transactionList)
print 'D1:'
print D1
print D1.shape[0]
tmp = (D1*D1).sum(axis=0)
print 'tmp:'
print tmp
print tmp.shape[0]
notmp = np.sqrt((D1*D1).sum(axis=0))
print 'notmp:'
print notmp
print notmp.shape[0]

                      
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)))
#print D1[0][115]
#print D1[1][115]
#print D1[2][115]
print 'normalize of D1:'
print D1
X=np.array([1.211512,6.988861,0.157972])#,2.0,5.0,8.0])

X = X.reshape(3,1, order='F')
X = D1
print 'norX:'
print (X*X).sum(axis=0)
print np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1))
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'X:'
print X

tic = time.time()
alpha_lasso_m1 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 1,pos=True,mode=0)

tac = time.time()
t = tac - tic
print 'alpha_lasoo_m1:'
print 'time:'+str(t)
print alpha_lasso_m1

tic = time.time()
alpha_lasso_m2 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 0,mode=1)
tac = time.time()
t = tac - tic
print 'alpha_lasoo_m2:'
print 'time:'+str(t)
print alpha_lasso_m2
#print 'dict1'
#print alpha_lasso_m2[0]
#a = str(alpha_lasso_m2)
#print a