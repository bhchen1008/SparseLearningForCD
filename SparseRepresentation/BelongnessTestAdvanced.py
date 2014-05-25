'''
Created on 2014/3/18

@author: bhchen
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np

#test signal
input_X = open(sys.argv[1])
testSignal = input_X.read()
testSLoader = aL.arffLoader()
testSLoader.load(testSignal)

#Load dictionaries
input_files = []
dicts = []
dictLoaders = []
Ds = []
print len(sys.argv)
for i in range(len(sys.argv)-3):
    input_files.append(open(sys.argv[i+2]))
    dicts.append(input_files[i].read())
    dictLoaders.append(aL.arffLoader)
    dictLoaders[i].load(dicts[i])
    Ds.append(dictLoaders[i].fortranArray(dictLoaders[i].transactionList))
    print 'D'+(i+1)+':'
    print Ds[i]
    Ds[i] = np.asfortranarray(Ds[i] / np.tile(np.sqrt((Ds[i]*Ds[i]).sum(axis=0)),(Ds[i].shape[0],1)))
    print 'normalize of D' + (i+1) + ':'
    print Ds[i]

output_f = open(sys.argv[len(sys.argv)-1],'w')


X = testSLoader.fortranArray(testSLoader.transactionList)
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'X:'
print X

#Caculate
alpha_lasso_m1_Ds = []
tic = time.time()
for i in range(len(sys.argv)-3):
    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[i],return_reg_path = False,lambda1 = 1,pos=True,mode=0))
tac = time.time()
t = tac - tic
print 'alpha_lasoo_m1:'
print 'time:'+str(t)
#print alpha_lasso_m1_D1.getcol(0)
print alpha_lasso_m1_Ds[0].getcol(0)

#get All Dict's Info
allInfos = []
for i in range(len(sys.argv)-3):
    allInfos.append({})
    
#allInfo1 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}
#allInfo2 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}

#choose Dict
tmps = []
splitByEnterDs = []
for j in range(X.shape[1]):
    instanceNo = j
    for i in range(len(sys.argv)-3):
        tmps.append(object)
        splitByEnterDs.append(tmps[i])
        for line in splitByEnterDs[i]:
            line = line.strip()
            pageNo = line.split(',')[0].split('(')[1].strip()
            weight = float(line.split(',')[1].split(')')[1].strip())
            if instanceNo in allInfos[i]:
                allInfos[i][instanceNo][pageNo] = weight
            else:
                allInfos[i][instanceNo] = {}
                allInfos[i][instanceNo][pageNo] = weight
        
    
#customize
max_weight_Ds = []
weight2Dict = {}

for instNo in allInfos[0]:
    weights = []
    for i in range(len(sys.argv)-3):
        max_weight_Ds[i] = sorted(allInfos[i][instNo].values(),reverse=True)[0]
        if max_weight_Ds[i] in weight2Dict:
            print 'error'
        weight2Dict[max_weight_Ds[i]] = i
        weights.append(max_weight_Ds[i])
        
    output_f.write((weight2Dict[max(weights)]+1)+'\n')
#        weight2Dict[float(str(max_weight_Ds[i])+i)]


#    max_weight_D1 = sorted(allInfo1[instNo].values(),reverse=True)[0]
#    max_weight_D2 = sorted(allInfo2[instNo].values(),reverse=True)[0]
    
#    if max_weight_D1 > max_weight_D2:
#        output_f.write('5\n')
#    else:
#        output_f.write('6\n')    
