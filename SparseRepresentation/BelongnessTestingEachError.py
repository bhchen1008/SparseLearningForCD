#encoding:utf-8
'''
Created on 2014/3/18

@author: bhchen
待處理Bug:
1.第17.18筆會回傳錯誤訊息, weight為0但是似乎會選中正確的instance
最後問題為Library問題
'''
import arffLoader as aL
import sys
import spams
import time
import numpy as np
import math
from collections import deque
from collections import Counter

def decideFinalDict(dictWindow,threshold):
    most_common,num_most_common = Counter(dictWindow).most_common(1)[0]
    if(len(set(dictWindow))>1):
        num_second_common = Counter(dictWindow).most_common(2)[1][1]
        if(num_most_common==num_second_common):
            return 'currentDict'
        else:
            if(num_most_common >= len(dictWindow)*threshold):
                return most_common
            else:
                return 'currentDict'
    else:
        if(num_most_common >= len(dictWindow)*threshold):
            return most_common
        else:
            return 'currentDict'

Alltic = time.time()
#numOfDicts = len(sys.argv)-5
#algowindow
numOfDicts = (len(sys.argv)-8) / 2

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
    
algoResults = []
#Load result of other algo
for i in range(numOfDicts):
    input_file = open(sys.argv[i+numOfDicts+2])
    result = input_file.read()
    algoResults.append(aL.arffLoader())
    algoResults[i].load(result)

numAlgoWindow = int(sys.argv[len(sys.argv)-6])
threshold = float(sys.argv[len(sys.argv)-5])

#DictionaryChoose
output_f = open(sys.argv[len(sys.argv)-4],'w')
#PredictReference output(Sparse Learning)
outputPredictSparse = open(sys.argv[len(sys.argv)-3],'w')
#result of other algorithm with Sparse Learning
outputPredictOtherAlgo = open(sys.argv[len(sys.argv)-2],'w')
#Compare Output
outputCompare = open(sys.argv[len(sys.argv)-1],'w')


#args for Lasso
alpha1Lambda = 1
compareLambda = 0.1
alpha_lasso_m1_Ds = None

#test signal
input_X = open(sys.argv[1])
testSignal = input_X.read()
testLoader = aL.arffLoader()
testLoader.load(testSignal)

#choose Dict
tmps = []
splitByEnterDs = []
dictInfos = []
weight2PageDs = []

for i in range(numOfDicts):
    dictInfos.append({})
    weight2PageDs.append({})


algoWindow = {}

max_weight_Ds = []
weight2Dict = {}
dictChooseUBE = {}  #用來記錄那些只有對應到一頁的Dict以及其對應的Page
weights = []        #比較該使用哪個Dict用
weightTmp = []      #用來過濾錯誤的weight
RightInstance = 0   #For caculating accuracy with Sparse Learning
RightInstanceOtherAlgo = 0  #For caculating accuracy with other algorithm with sparse learning
dictPage = 0        #兩本都只對照到一頁時使用
algoWindow = deque()
dictChoose = 0
finalDictChoose = 0
currentDict = 0

numOfInsts = len(testLoader.transactionContentList)
for instNo in range(numOfInsts):
#    print 'instNo:'+str(instNo)
    X = testLoader.singleFortranArray(testLoader.transactionContentList[instNo])
#    print 'X:'
#    print X
    #normalize X/各值平方相加開根號
    #X[0][0]為第一筆的第一個欄位值, X[1][0]為第一筆的第二個欄位值, X[A1][A2]為第(A2+1)筆的第(A1+1)個欄位值
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
#    print 'normalize of X:'
#    print X
    instanceNo = instNo
    
    tmps[:] = []
    splitByEnterDs[:] = []
    #Caculate
    for dictNo in range(numOfDicts):
        tic = time.time()
#        alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0)
#        alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = compareLambda,pos=True,mode=1)
        alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 1,pos=True,mode=0)
#        alpha_lasso_m2_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 0,pos=True,mode=1)
#        alpha_lasso_m3_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 0.5,lambda2 = 1,pos=True,mode=2)
        
        tic = time.time()
        #spams.omp
#        alpha_omp_m1_Ds = spams.omp(X,Ds[dictNo],L=2,return_reg_path = False,numThreads = -1)
#        alpha_omp_m2_Ds = spams.omp(X,Ds[dictNo],eps= 0.9,return_reg_path = False,numThreads = -1)
#        alpha_omp_m3_Ds = spams.omp(X,Ds[dictNo],lambda1=0.4,return_reg_path = False,numThreads = -1)
#        alpha_lasso_m1_Ds.max()
    #    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = compareLambda,pos=True,mode=1))
        tac = time.time()
        t = tac - tic
#        print 'alpha_lasoo_m1:'
#        print 'time:'+str(t)
        
        tmps.append(str(alpha_lasso_m1_Ds.getcol(0)))
#        tmps.append(str(alpha_omp_m1_Ds.getcol(0)))
        
        outputCompare.write(str(instNo)+'-D'+str(dictNo)+':'+tmps[dictNo]+'\n\n')
        
        splitByEnterDs.append(tmps[dictNo].split('\n'))
        for line in splitByEnterDs[dictNo]:
            line = line.strip()
            #mapping page
            pageNo = int(line.split(',')[0].split('(')[1].strip())
            #weight of mapping page
            weight = float(line.split(',')[1].split(')')[1].strip())
            #到這邊已把pageNo與weight Parse出
            if len(dictInfos[dictNo]) >= 1:
                dictInfos[dictNo][pageNo] = weight
            else:
                dictInfos[dictNo] = {}
                dictInfos[dictNo][pageNo] = weight
    
    #Choose Diction
    for dictNo in range(numOfDicts):
#        print allInfos[dictNo][instNo].values()
        #examine the correctness of weight
        
        weight2PageDs[dictNo].clear()
        
        #no one page bug
        '''
        if len(dictChooseUBE) >= 1:
            continue             
        for page in dictInfos[dictNo].keys():
            weight = dictInfos[dictNo][page]
            if weight < alpha1Lambda + 0.5:
                weightTmp.append(weight)
                weight2PageDs[dictNo][weight] = page        

#            print "instNo:"+str(instNo)                    
        max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
        #取完weightTmp後，清空
        weightTmp[:] = []
        #check for same weight
#        if max_weight_Ds[dictNo] in weight2Dict:
#            print instNo
#            print max_weight_Ds[dictNo]
#            print 'error\n'
        weight2Dict[max_weight_Ds[dictNo]] = dictNo
        weights.append(max_weight_Ds[dictNo])
        '''
        #if there is only one page in Dict, the value is almost same
        if len(dictInfos[dictNo].keys())==1:
            # 如果已經有一本字典只有對照到一頁，今天進來另外一本的話，則去比對原資料看哪個相差較
            dictChooseUBE[dictNo] = dictInfos[dictNo].keys()[0]
            #避免第一本字典對到多頁，其他本字典對到一頁時，weight2Dict與weights裡頭會有資料，所以這邊將兩個list清空
            weight2Dict.clear()
            weights[:] = []
#            for page in allInfos[dictNo][instNo].keys():
#                weight = allInfos[dictNo][instNo][page]
#                if weight < alpha1Lambda + 0.5:
#                    weightTmp.append(weight)
#                    weight2PageDs[dictNo][weight] = page        
#                    
#            max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
#            weight2Dict[max_weight_Ds[dictNo]] = dictNo
#            weights.append(max_weight_Ds[dictNo])
        else:
            if len(dictChooseUBE) >= 1:
                continue             
            for page in dictInfos[dictNo].keys():
                weight = dictInfos[dictNo][page]
                if weight < alpha1Lambda + 0.5:
                    weightTmp.append(weight)
                    weight2PageDs[dictNo][weight] = page        

#            print "instNo:"+str(instNo)                    
            max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
            #取完weightTmp後，清空
            weightTmp[:] = []
            #check for same weight
    #        if max_weight_Ds[dictNo] in weight2Dict:
    #            print instNo
    #            print max_weight_Ds[dictNo]
    #            print 'error\n'
            weight2Dict[max_weight_Ds[dictNo]] = dictNo
            weights.append(max_weight_Ds[dictNo])

    #choose Dictionary
    #the page is almost 1 to 1, if there is only one page.
    if len(dictChooseUBE) > 1:  #如果有兩本字典都只對應到一頁，那麼與test data比較看誰差距較小
        for dictNum in range(len(dictChooseUBE)):
            for i in range(testLoader.numAttribute):
                #正確答案
                testValue = float(testLoader.transactionContentList[instNo][i])
                dictPage = dictChooseUBE[dictChooseUBE.keys()[dictNum]]
                dictValue = float(dictLoaders[dictChooseUBE.keys()[dictNum]].transactionContentList[dictPage][i])
                if i == 0:
#                    weight.append(math.fabs(testValue-dictValue)/testValue)
                    weightTmp.append(math.fabs(testValue-dictValue)/testValue)
                else:
                    weightTmp[dictNum] += math.fabs(testValue-dictValue)/testValue
            weight2Dict[weightTmp[dictNum]] = dictChooseUBE.keys()[dictNum]
            weights.append(weightTmp[dictNum])
        weightTmp[:] = []
            
        dictChoose = weight2Dict[min(weights)]
        pageInDict = dictChooseUBE[dictChoose]
#        dictChoose = weight2Dict[max(weights)]
    elif len(dictChooseUBE) == 1:
        dictChoose = dictChooseUBE.keys()[0]
        pageInDict = dictChooseUBE[dictChoose]
    else:
        dictChoose = weight2Dict[max(weights)]
        pageInDict = weight2PageDs[dictChoose][max(weights)]
    
    
    #choose FinalDict
    if(len(algoWindow)==numAlgoWindow):
        algoWindow.popleft()
    algoWindow.append(dictChoose)
    finalDict = decideFinalDict(algoWindow,threshold)
    if(finalDict!='currentDict'):
        finalDictChoose = finalDict
#    else:
#        test=1#print 'same'            
    
    output_f.write(str(dictChoose)+'FinalDictChoose:'+str(finalDictChoose)+'\n')
    
    
    #use Sparse Learning
    outputPredictSparse.write(dictLoaders[dictChoose].transactionList[pageInDict]+'\n')
    #use result of other algo
    if(instNo==1504):
        print 'done'
    outputPredictOtherAlgo.write(algoResults[finalDictChoose].transactionList[instNo]+'\n')
    
    if dictLoaders[dictChoose].className[pageInDict] == testLoader.className[instNo]:
        RightInstance += 1
    else:
        print 'WrongInstance:' + str(instNo) + 'using Dict' + str(dictChoose) + 'in page_' + str(pageInDict)
    
    if int(algoResults[finalDictChoose].className[instNo]) == int(testLoader.className[instNo]):
        RightInstanceOtherAlgo += 1
    
    #clear list
    max_weight_Ds[:] = []
    weights[:] = []
    
#    for value in range(len(max_weight_Ds)):
#        max_weight_Ds.pop()
#    for value in range(len(weights)):
#        weights.pop()
    
    for i in range(len(dictInfos)):
        dictInfos[i].clear()
    weight2Dict.clear()
    dictChooseUBE.clear()

print 'RightInstance use Sparse Learning:' + str(RightInstance)
print 'Accuracy:'+str(float(RightInstance)/numOfInsts)
print '\n'
print 'RightInstance use other algorithm with Sparse Learning:' + str(RightInstanceOtherAlgo)
print 'Accuracy:'+str(float(RightInstanceOtherAlgo)/numOfInsts)

Alltac = time.time()
print 'AllTime:' + str(Alltac - Alltic)