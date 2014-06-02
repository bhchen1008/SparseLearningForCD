'''
Created on 2014/3/18

@author: bhchen
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np
import scipy as sp

#print np.__version__
#print sp.__version__
#dictionary 1
input_f1 = open(sys.argv[1])
dict1 = input_f1.read()
#dictionary 2
input_f2 = open(sys.argv[2])
dict2 = input_f2.read()
#test signal
input_X = open(sys.argv[3])
testSignal = input_X.read()
output_f = open(sys.argv[4],'w')

dictLoader1 = aL.arffLoader()
#testContent = dictLoader1.load(dict1)
dictLoader1.load(dict1)

dictLoader2 = aL.arffLoader()
dictLoader2.load(dict2)

testLoader = aL.arffLoader()
testLoader.load(testSignal)

#
#print dictLoader1.classIndex
#print dictLoader1.attrName
#print dictLoader1.transactionContentList[0]
#print dictLoader1.numInstance
D1 = dictLoader1.fortranArray(dictLoader1.transactionContentList)
print 'D1:'
print D1
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)))
print 'normalize of D1:'
print D1

D2 = dictLoader2.fortranArray(dictLoader2.transactionContentList)
print 'D2:'
print D2
D2 = np.asfortranarray(D2 / np.tile(np.sqrt((D2*D2).sum(axis=0)),(D2.shape[0],1)))
print 'normalize of D2:'
print D2

X = testLoader.fortranArray(testLoader.transactionContentList)
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'X:'
print X

tic = time.time()
alpha_lasso_m1_D1 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 1,pos=True,mode=0)
#alpha_lasso_m1_D1 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 0,pos=True,mode=1)
#alpha_lasso_m1_D1 = spams.omp(X,D1,L=3,eps=None,lambda1 = None,return_reg_path = False,numThreads = -1)
#alpha_lasso_m1_D1 = spams.omp(X,D1,L=None,eps=0,lambda1 = None,return_reg_path = False,numThreads = -1)
tac = time.time()
t = tac - tic
print 'alpha_lasoo_m1_D1:'
print 'time:'+str(t)
tic = time.time()
alpha_lasso_m1_D2 = spams.lasso(X,D2,return_reg_path = False,lambda1 = 1,pos=True,mode=0)
#alpha_lasso_m1_D2 = spams.lasso(X,D2,return_reg_path = False,lambda1 = 0,pos=True,mode=1)
#alpha_lasso_m1_D2 = spams.omp(X,D2,L=3,eps=None,lambda1 = None,return_reg_path = False,numThreads = -1)
#alpha_lasso_m1_D2 = spams.omp(X,D2,L=None,eps=0,lambda1 = None,return_reg_path = False,numThreads = -1)
tac = time.time()
t = tac - tic
print 'alpha_lasoo_m1_D2:'
print 'time:'+str(t)

#print alpha_lasso_m1_D1.getcol(0)
allInfo1 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}
allInfo2 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}

print 'rows:'+str(X.shape[1])
for j in range(X.shape[1]):
    tmp1 = str(alpha_lasso_m1_D1.getcol(j)).strip()
    splitByEnterD1 = tmp1.split('\n')
    if j==1516:#j<20:# or j==1516:
        print 'D1-line'+str(j)
        print 'tmp1:'+tmp1
        print 'split:'+str(splitByEnterD1)
        print 'number of pages:'+str(len(splitByEnterD1))
    instanceNo = j
    for line in splitByEnterD1:
        line = line.strip()
#        instanceNo = line.split(',')[1].split(')')[0].strip()
#        if j<20:
#            print 'line'+line
#            test = line.split(',')[0].split('(')
#            print 'test:'+str(test)
#            print 'instanceNo:'+line.split(',')[0].split('(')[1].strip()
        pageNo = line.split(',')[0].split('(')[1].strip()
        weight = float(line.split(',')[1].split(')')[1].strip())
        if instanceNo in allInfo1:
            allInfo1[instanceNo][pageNo] = weight
        else:
            allInfo1[instanceNo] = {}
            allInfo1[instanceNo][pageNo] = weight
#    print '\n'
    
    tmp2 = str(alpha_lasso_m1_D2.getcol(j))
    splitByEnterD2 = tmp2.split('\n')
#    if j<20:# or j>1516:
    if j==1516:#j<20:# or j==1516:
        print 'D2-line'+str(j)
        print 'tmp1:'+tmp2
        print 'split:'+str(splitByEnterD2)
        print 'number of pages:'+str(len(splitByEnterD2))
    for line in splitByEnterD2:
        line = line.strip()
#        instanceNo = line.split(',')[1].split(')')[0].strip()
        pageNo = line.split(',')[0].split('(')[1].strip()
        weight = float(line.split(',')[1].split(')')[1].strip())
        if instanceNo in allInfo2:
            allInfo2[instanceNo][pageNo] = weight
        else:
            allInfo2[instanceNo] = {}
            allInfo2[instanceNo][pageNo] = weight

#customize
for instNo in allInfo1:
#    if instNo != 0:
#        output_f.write(',')
    max_weight_D1 = sorted(allInfo1[instNo].values(),reverse=True)[0]
    max_weight_D2 = sorted(allInfo2[instNo].values(),reverse=True)[0]
    if max_weight_D1 > max_weight_D2:
        output_f.write('5\n')
    else:
        output_f.write('6\n')    
'''    
for line in splitByEnterD1:
    line = line.strip()
    instanceNo = line.split(',')[1].split(')')[0].strip()
    pageNo = line.split(',')[0].split('(')[1].strip()
    weight = float(line.split(',')[1].split(')')[1].strip())
    if instanceNo in allInfo1:
        allInfo1[instanceNo][pageNo] = weight
    else:
        allInfo1[instanceNo] = {}
        allInfo1[instanceNo][pageNo] = weight
    
alphaLassoM1D2 = str(alpha_lasso_m1_D2)
splitByEnterD2 = alphaLassoM1D2.split('\n')
allInfo2 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}
for line in splitByEnterD2:
    line = line.strip()
    instanceNo = line.split(',')[1].split(')')[0].strip()
    pageNo = line.split(',')[0].split('(')[1].strip()
    weight = line.split(',')[1].split(')')[1].strip()
    if instanceNo in allInfo2:
        allInfo2[instanceNo][pageNo] = weight
    else:
        allInfo2[instanceNo] = {}
        allInfo2[instanceNo][pageNo] = weight
'''
        
    
    
#print alpha_lasso_m1

#tic = time.time()
#alpha_lasso_m2 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 0,mode=1)
#tac = time.time()
#t = tac - tic
#print 'alpha_lasoo_m2:'
#print 'time:'+str(t)
#print alpha_lasso_m2

#print 'dict1'
#print alpha_lasso_m2[0]
#a = str(alpha_lasso_m2)
#print a