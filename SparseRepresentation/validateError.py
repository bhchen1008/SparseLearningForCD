'''
Created on 2014/7/3

@author: bhchen
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np

Ds = []
numDicts = len(sys.argv)-2
for dictNo in range(numDicts):
    input_file = open(sys.argv[dictNo+1])
    dictContent = input_file.read()
    dictLoaders = aL.arffLoader()
    dictLoaders.load(dictContent)
    Ds.append(dictLoaders.fortranArray(dictLoaders.transactionContentList))
    Ds[dictNo] = np.asfortranarray(Ds[dictNo] / np.tile(np.sqrt((Ds[dictNo]*Ds[dictNo]).sum(axis=0)),(Ds[dictNo].shape[0],1)))
    

input_X = open(sys.argv[len(sys.argv)-1])
testSignal = input_X.read()
testLoader = aL.arffLoader()
testLoader.load(testSignal)

X = testLoader.fortranArray(testLoader.transactionContentList)
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))

alpha1Lambda = 1
alpha_lasso_m1_Ds = []
for dictNo in range(numDicts):
    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0))

dictInfo = {}

for dictNo in range(numDicts):
    tmpValue = []

for i in range(X.shape[1]):
    print 'InstNo:'+str(i)
    tmpValue = []
    for j in range(X.shape[0]):
        tmpValue.append(X[j][i])
    
    for dictNo in range(numDicts):
        Error = []
        for j in range(X.shape[0]):
            Error.append(tmpValue[j])
        tmps = str(alpha_lasso_m1_Ds[dictNo].getcol(i))
        splitByEnterDs = tmps.split('\n')
            
        for line in splitByEnterDs:
            line = line.strip()
            #mapping page
            pageNo = int(line.split(',')[0].split('(')[1].strip())
            #weight of mapping page
            weight = float(line.split(',')[1].split(')')[1].strip())
            dictInfo[pageNo] = weight
            
            
            for j in range(X.shape[0]):
                pageValue = Ds[dictNo][j][pageNo]
                Error[j] -= (pageValue * weight)
                
        error = sum(Error)
        print 'D'+str(dictNo)+',error:' + str(error)
            
        
        