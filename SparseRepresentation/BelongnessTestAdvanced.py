#encoding:utf-8
'''
Created on 2014/3/18

@author: bhchen
待處理Bug:
1.第17.18筆會回傳錯誤訊息, weight為0但是似乎會選中正確的instance
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np

#test signal
input_X = open(sys.argv[1])
testSignal = input_X.read()
testLoader = aL.arffLoader()
testLoader.load(testSignal)
X = testLoader.fortranArray(testLoader.transactionContentList)
print 'X:'
print X
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'normalize of X:'
print X

numOfDicts = len(sys.argv)-5

#Load dictionaries
input_files = []
dicts = []
dictLoaders = []
Ds = []
for i in range(numOfDicts):
    input_files.append(open(sys.argv[i+2]))
    dicts.append(input_files[i].read())
    dictLoaders.append(aL.arffLoader())
    dictLoaders[i].load(dicts[i])
    Ds.append(dictLoaders[i].fortranArray(dictLoaders[i].transactionContentList))
    print 'D'+str(i+1)+':'
    print Ds[i]
    Ds[i] = np.asfortranarray(Ds[i] / np.tile(np.sqrt((Ds[i]*Ds[i]).sum(axis=0)),(Ds[i].shape[0],1)))
#    print np.tile(np.sqrt((Ds[i]*Ds[i]).sum(axis=0)
    print 'normalize of D' + str(i+1) + ':'
    print Ds[i]

output_f = open(sys.argv[len(sys.argv)-3],'w')
#PredictReference output
outputPredictReference = open(sys.argv[len(sys.argv)-2],'w')
#Compare Output
outputCompare = open(sys.argv[len(sys.argv)-1],'w')



#Caculate
#args for Lasso
alpha1Lambda = 1
compareLambda = 0
alpha_lasso_m1_Ds = []

for i in range(numOfDicts):
    tic = time.time()
    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[i],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0))
#    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[i],return_reg_path = False,lambda1 = compareLambda,pos=True,mode=1))
    tac = time.time()
    t = tac - tic
    print 'alpha_lasoo_m1:'
    print 'time:'+str(t)

#print alpha_lasso_m1_D1.getcol(0)
#print alpha_lasso_m1_Ds[0].getcol(0)

#save each Dict's Info
allInfos = []
#weight2Page in Dict
weight2PageDs = []
for i in range(numOfDicts):
    allInfos.append({})
    weight2PageDs.append({})
    
#allInfo1 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}
#allInfo2 = {}       #ex:{'0':{'0':0.97,'1':0.02,'2':0.01},'1':{'0':....},...}

#choose Dict
tmps = []
splitByEnterDs = []
#accroding each instance to save mapping result
for j in range(X.shape[1]):
    instanceNo = j
    #clear tmps
    for value in range(len(tmps)):
        tmps.pop()
    for value in range(len(splitByEnterDs)):
        splitByEnterDs.pop()
    for i in range(numOfDicts):
        #save mapping result
        tmps.append(str(alpha_lasso_m1_Ds[i].getcol(j)))
        if j<=20:
            print str(j)+'-D'+str(i)+':'+tmps[i]+'\n'
        outputCompare.write(str(j)+'-D'+str(i)+':'+tmps[i]+'\n\n')
        #split
#        print tmps[i].split('\n')
        splitByEnterDs.append(tmps[i].split('\n'))
#        splitByEnterD1 = tmp1.split('\n')
        for line in splitByEnterDs[i]:
            line = line.strip()
            #mapping page
            pageNo = int(line.split(',')[0].split('(')[1].strip())
            #weight of mapping page
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
    if(instNo==4):
        print 'here'
    weights = []
    
    for i in range(numOfDicts):
#        print allInfos[i][instNo].values()
        #examine the correctness of weight
        weightTmp = []
        weight2PageDs[i].clear()
#        print allInfos[i][instNo]

#        for weight in allInfos[i][instNo].values():
#            if weight < alpha1Lambda + 0.5:
#                weightTmp.append(weight)
        for page in allInfos[i][instNo].keys():
            weight = allInfos[i][instNo][page]
            if weight < alpha1Lambda + 0.5:
                weightTmp.append(weight)
                
                weight2PageDs[i][weight] = page        
                
        max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
        if max_weight_Ds[i] in weight2Dict:
            print instNo
            print max_weight_Ds[i]
            print 'error\n'
        weight2Dict[max_weight_Ds[i]] = i
        weights.append(max_weight_Ds[i])
        
    dictChoose = weight2Dict[max(weights)]
    output_f.write(str(dictChoose)+'\n')
    outputPredictReference.write(dictLoaders[dictChoose].transactionList[weight2PageDs[dictChoose][max(weights)]]+'\n')
    
    for value in range(len(max_weight_Ds)):
        max_weight_Ds.pop()
    
    weight2Dict.clear()
#        weight2Dict[float(str(max_weight_Ds[i])+i)]


#    max_weight_D1 = sorted(allInfo1[instNo].values(),reverse=True)[0]
#    max_weight_D2 = sorted(allInfo2[instNo].values(),reverse=True)[0]
    
#    if max_weight_D1 > max_weight_D2:
#        output_f.write('5\n')
#    else:
#        output_f.write('6\n')    
