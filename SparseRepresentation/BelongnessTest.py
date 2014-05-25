'''
Created on 2014/3/18

@author: bhchen
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np


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

testSLoader = aL.arffLoader()
testSLoader.load(testSignal)

#
#print dictLoader1.classIndex
#print dictLoader1.attrName
#print dictLoader1.transactionList[0]
#print dictLoader1.numInstance
D1 = dictLoader1.fortranArray(dictLoader1.transactionList)
print 'D1:'
print D1
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)))
print 'normalize of D1:'
print D1

D2 = dictLoader2.fortranArray(dictLoader2.transactionList)
print 'D2:'
print D2
D2 = np.asfortranarray(D2 / np.tile(np.sqrt((D2*D2).sum(axis=0)),(D2.shape[0],1)))
print 'normalize of D2:'
print D2

X = testSLoader.fortranArray(testSLoader.transactionList)
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'X:'
print X

tic = time.time()
alpha_lasso_m1_D1 = spams.lasso(X,D1,return_reg_path = False,lambda1 = 1,pos=True,mode=0)
alpha_lasso_m1_D2 = spams.lasso(X,D2,return_reg_path = False,lambda1 = 1,pos=True,mode=0)
tac = time.time()
t = tac - tic
print 'alpha_lasoo_m1:'
print 'time:'+str(t)
print alpha_lasso_m1_D1.getcol(0)
allInfo1 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}
allInfo2 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}

for j in range(X.shape[1]):
    tmp1 = str(alpha_lasso_m1_D1.getcol(j))
    splitByEnterD1 = tmp1.split('\n')
    instanceNo = j
    for line in splitByEnterD1:
        line = line.strip()
#        instanceNo = line.split(',')[1].split(')')[0].strip()
        pageNo = line.split(',')[0].split('(')[1].strip()
        weight = float(line.split(',')[1].split(')')[1].strip())
        if instanceNo in allInfo1:
            allInfo1[instanceNo][pageNo] = weight
        else:
            allInfo1[instanceNo] = {}
            allInfo1[instanceNo][pageNo] = weight
    
    tmp2 = str(alpha_lasso_m1_D2.getcol(j))
    splitByEnterD2 = tmp2.split('\n')
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