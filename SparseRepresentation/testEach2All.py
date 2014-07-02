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
#save each Dict's Info
dictInfos = []
#weight2Page in Dict
weight2PageDs = []
#List for save weight of window
weightWindowDerrorWindowDsi in range(numOfDicts):
    dictInfos.append({})
    weight2PageDs.append({})
    weightWindowDerrorWindowDs())


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
testDataWindow = deque()

numOfInsts = testLoader.numInstance
numOfAttrs = testLoader.numAttribute
for instNo in range(numOfInsts):
#    print 'instNo:'+str(instNo)
    currentTestData = testLoader.transactionContentList[instNo]
    X = testLoader.singleFortranArray(currentTestData)
#    print 'X:'
#    print X
    #normalize X/各值平方相加開根號
    #X[0][0]為第一筆的第一個欄位值, X[1][0]為第一筆的第二個欄位值, X[A1][A2]為第(A2+1)筆的第(A1+1)個欄位值
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
#    print 'normalize of X:'
#    print X
    testDataWindow.append(currentTestData)
    
testWindow = aL.arffLoader()
#testWindow.load(currentTestData)
testD = testWindow.fortranArrayPara(testDataWindow,numOfAttrs,numOfInsts)
testD = np.asfortranarray(testD / np.tile(np.sqrt((testD*testD).sum(axis=0)),(testD.shape[0],1)))

print 'test:'
print testD