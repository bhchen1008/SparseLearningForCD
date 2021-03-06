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
from collections import deque
from collections import Counter
import os
#import operator

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
        
def reChooseDict(instNo,currDict,currMean,weightWindowDsCurDict,currStdCompare,currMeanCompare,dicts,testDWindow,numAttrs,numInsts,outputCompare,Lambda):
    outputCompare.write('Re-Choose Dictionary!\n')
    outputPredictSparse.write('Re-Choose Dictionary!\n')
    
    weightWindowDsRe = []
#    tmpsRe = []
    tmpsRe = 0
#    splitByEnterDsRe = []
    splitByEnterDsRe = 0
    numDicts = len(dicts)
    for i in range(numDicts):
        weightWindowDsRe.append(deque())
    dictWeightMeanRe = {}
    testWindow = aL.arffLoader()
    testD = testWindow.fortranArrayPara(testDWindow,numAttrs,numInsts)
    testD = np.asfortranarray(testD / np.tile(np.sqrt((testD*testD).sum(axis=0)),(testD.shape[0],1)))
    for dictNo in range(numDicts):
#        tmpsRe[:] = []
#        splitByEnterDsRe[:] = []
        if(dictNo==currDict):
            weightWindowDsRe[dictNo] = weightWindowDsCurDict
            continue
        alpha_lasso_m1_Ds_batch = spams.lasso(testD,dicts[dictNo],return_reg_path = False,lambda1 = 1,pos=True,mode=0)
        
        for j in range(alpha_lasso_m1_Ds_batch.shape[1]):
#            tmpsRe.append(str(alpha_lasso_m1_Ds_batch.getcol(j)))
            tmpsRe = str(alpha_lasso_m1_Ds_batch.getcol(j))
            #instNo+j才是正確的instance Number
            outputCompare.write(str(instNo-len(testDWindow)+j)+'-D'+str(dictNo)+':'+tmpsRe+'\n\n')
            #split
            #print tmpsRe[i].split('\n')
#            splitByEnterDsRe.append(tmpsRe[j].split('\n'))
            splitByEnterDsRe = tmpsRe.split('\n')
            #        splitByEnterD1 = tmp1.split('\n')
            weightTmp = []
#            for line in splitByEnterDsRe[j]:
            for line in splitByEnterDsRe:
                line = line.strip()
                #mapping page
                pageNo = int(line.split(',')[0].split('(')[1].strip())
                #weight of mapping page
                weight = float(line.split(',')[1].split(')')[1].strip())
                if weight < Lambda + 0.02:
                    weightTmp.append(weight)
#                if weight >= 1:
#                    print 'InstNo:'+str(instNo+j)+', DictNo:'+str(dictNo)+', page:'+str(pageNo)+', weight:'+str(weight)
            maxWeight = max(weightTmp)
            weightTmp[:] = []
            #one page bug if weight = 0.0 transform to 1
            if maxWeight==0:
                maxWeight = 1
            if maxWeight > 1:
                maxWeight = 1
            weightWindowDsRe[dictNo].append(maxWeight)
            
    #choose dict
    for dictNo in range(numOfDicts):
        #若是目前的Dict則直接給值
        if(dictNo==currDict):
            dictWeightMeanRe[dictNo] = currMean
        #若不是則要重新計算一次
        else:
            dictWeightMeanRe[dictNo] = np.mean(weightWindowDsRe[dictNo])
    #找出值最大的Dict，以及平均值
    maxWeightDict,meanCompareRe = max(dictWeightMeanRe.iteritems(), key=lambda x:x[1])
    if(maxWeightDict==currDict):
        #更新currMean
        meanCompareRe = currMean
        #保留舊的currMeanCompare
#        meanCompareRe = currMeanCompare
        #更新stdCompare
        
        #保留舊的stdCompare
#        stdCompareRe = currStdCompare
        #更新stdCompare
        stdCompareRe = np.std(weightWindowDsRe[maxWeightDict])
#        weightWindowDsRe[maxWeightDict] = weightWindowDsCurDict
        outputCompare.write('Keep same model'+str(maxWeightDict)+'!\n')
        outputPredictSparse.write('Keep same model'+str(maxWeightDict)+'!\n')
    else:
#        changeModelRe = 1
        stdCompareRe = np.std(weightWindowDsRe[maxWeightDict])
        outputCompare.write('Change model to model-' + str(maxWeightDict) +',meanCompare:'+str(meanCompareRe)+',stdCompare:'+str(stdCompareRe)+'!\n')
        outputPredictSparse.write('Change model to model-' + str(maxWeightDict) +',meanCompare:'+str(meanCompareRe)+',stdCompare:'+str(stdCompareRe)+'!\n')
    
    return (maxWeightDict,meanCompareRe,stdCompareRe,weightWindowDsRe[maxWeightDict])
    
Alltic = time.time()
#numOfDicts = len(sys.argv)-5
#algowindow
numOfDicts = (len(sys.argv)-12) / 2

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
#    print 'D'+str(i+1)+':'
#    print Ds[i]
    Ds[i] = np.asfortranarray(Ds[i] / np.tile(np.sqrt((Ds[i]*Ds[i]).sum(axis=0)),(Ds[i].shape[0],1)))
#    print np.tile(np.sqrt((Ds[i]*Ds[i]).sum(axis=0)
#    print 'normalize of D' + str(i+1) + ':'
#    print Ds[i]
    
algoResults = []
#Load result of other algo
for i in range(numOfDicts):
    input_file = open(sys.argv[i+numOfDicts+2])
    result = input_file.read()
    algoResults.append(aL.arffLoader())
    algoResults[i].load(result)

numAlgoWindow = int(sys.argv[len(sys.argv)-10])
threshold = float(sys.argv[len(sys.argv)-9])

#DictionaryChoose
output_f = open(sys.argv[len(sys.argv)-8],'w')
#PredictReference output(Sparse Learning)
outputPredictSparse = open(sys.argv[len(sys.argv)-7],'w')
#result of other algorithm with Sparse Learning
outputPredictOtherAlgo = open(sys.argv[len(sys.argv)-6],'w')
#Compare Output
outputCompare = open(sys.argv[len(sys.argv)-5],'w')
outputExecutionTime = open(sys.argv[len(sys.argv)-4],'w')
#Change Threshold
changeRefStd = float(sys.argv[len(sys.argv)-3])
#ReTrainDict
reTrainDict = sys.argv[len(sys.argv)-2]
#ReTrainPredict
reTrainPredict = sys.argv[len(sys.argv)-1]

for i in range(len(sys.argv)):
    print 'sys.argv['+str(i)+']='+sys.argv[i]


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
weightWindowDs = []

for i in range(numOfDicts):
    dictInfos.append({})
    weight2PageDs.append({})
    weightWindowDs.append(deque())


#algoWindow = {}

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
currentTestData = 0
dictErrorMean = {}
std2Dict = {}
#目前使用的字典
currentDict = 0
meanCompare = 0
stdCompare = 0

currentMean = 0
currentStd = 0

numOfInsts = testLoader.numInstance
numOfAttrs = testLoader.numAttribute
instExecTime = 0
for instNo in range(numOfInsts):
#    print 'instNo:'+str(instNo)
    instExecTime = time.time()
    currentTestData = testLoader.transactionContentList[instNo]
    X = testLoader.singleFortranArray(currentTestData)
#    print 'X:'
#    print X
    #normalize X/各值平方相加開根號
    #X[0][0]為第一筆的第一個欄位值, X[1][0]為第一筆的第二個欄位值, X[A1][A2]為第(A2+1)筆的第(A1+1)個欄位值
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
#    print 'normalize of X:'
#    print X
    if(len(testDataWindow)==numAlgoWindow):
        testDataWindow.popleft()
    testDataWindow.append(currentTestData)
        
    instanceNo = instNo
    
    tmps[:] = []
    splitByEnterDs[:] = []
    #Initial in slidingWindow decide the initial Dictionary
    #Caculate
    if(instNo < numAlgoWindow):
        for dictNo in range(numOfDicts):
    #        alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0)
    #        alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = compareLambda,pos=True,mode=1)
            alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 1,pos=True,mode=0)
    #        alpha_lasso_m2_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 0,pos=True,mode=1)
    #        alpha_lasso_m3_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = 0.5,lambda2 = 1,pos=True,mode=2)
            
            #spams.omp
    #        alpha_omp_m1_Ds = spams.omp(X,Ds[dictNo],L=2,return_reg_path = False,numThreads = -1)
    #        alpha_omp_m2_Ds = spams.omp(X,Ds[dictNo],eps= 0.9,return_reg_path = False,numThreads = -1)
    #        alpha_omp_m3_Ds = spams.omp(X,Ds[dictNo],lambda1=0.4,return_reg_path = False,numThreads = -1)
    #        alpha_lasso_m1_Ds.max()
        #    alpha_lasso_m1_Ds.append(spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = compareLambda,pos=True,mode=1))
            
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
            
            for page in dictInfos[dictNo].keys():
                weight = dictInfos[dictNo][page]
                if weight <= alpha1Lambda + 0.05:
                    weightTmp.append(weight)
                    weight2PageDs[dictNo][weight] = page
#                if weight >= 1:
#                    print 'InstNo:'+str(instNo)+', DictNo:'+str(dictNo)+', page:'+str(page)+', weight:'+str(weight)
                    
            maxWeight = max(weightTmp)
            max_weight_Ds.append(maxWeight)
            #取完weightTmp後，清空
            weightTmp[:] = []
            
            #one page bug if weight = 0.0 transform to 1
            if maxWeight==0:
                maxWeight = 1
            elif maxWeight > 1:
                maxWeight = 1
#            if(len(weightWindowDs)==numAlgoWindow):
#                weightWindowDs.popleft()            
            weightWindowDs[dictNo].append(maxWeight)
            
#        print 'instNo:'+str(instNo)
        #當sliding window的資料已滿，即可開始選擇初始字典            
        if(instNo==(numAlgoWindow-1)):
            for dictNo in range(numOfDicts):
                dictErrorMean[dictNo] = np.mean(weightWindowDs[dictNo])
            currentDict,meanCompare = max(dictErrorMean.iteritems(), key=lambda x:x[1])
            stdCompare = np.std(weightWindowDs[currentDict])
#            currentDict = max(dictErrorMean.iteritems(), key=operator.itemgetter(1))[0]
            outputPredictSparse.write('Initial:\n'+'currentDict:'+str(currentDict)+'\n'+'meanCompare:'+str(meanCompare)+'\n'+'StdCompare:'+str(stdCompare)+'\n\n')
            for i in range(numAlgoWindow):
                output_f.write(str(currentDict)+'\n')
                outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[i]+'\n')

            
    #initial done
    else:
        #一筆資料進來後先predict那筆屬於哪個Label再去check有沒有需要更換Model
        output_f.write(str(currentDict)+'\n')
        outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[instNo]+'\n')
        if int(algoResults[currentDict].className[instNo]) == int(testLoader.className[instNo]):
            RightInstanceOtherAlgo += 1
        
        alpha_lasso_m1_Ds = spams.lasso(X,Ds[currentDict],return_reg_path = False,lambda1 = 1,pos=True,mode=0)
        tmps.append(str(alpha_lasso_m1_Ds.getcol(0)))
        outputCompare.write(str(instNo)+'-D'+str(currentDict)+':'+tmps[0]+'\n\n')
        splitByEnterDs.append(tmps[0].split('\n'))
        for line in splitByEnterDs[0]:
            line = line.strip()
            #mapping page
            pageNo = int(line.split(',')[0].split('(')[1].strip())
            #weight of mapping page
            weight = float(line.split(',')[1].split(')')[1].strip())
            #到這邊已把pageNo與weight Parse出
            if len(dictInfos[currentDict]) >= 1:
                dictInfos[currentDict][pageNo] = weight
            else:
                dictInfos[currentDict] = {}
                dictInfos[currentDict][pageNo] = weight
            
        for page in dictInfos[currentDict].keys():
            weight = dictInfos[currentDict][page]
            if weight < alpha1Lambda + 0.05:
                weightTmp.append(weight)
                weight2PageDs[currentDict][weight] = page
#            if weight >= 1:
#                    print 'InstNo:'+str(instNo)+', DictNo:'+str(currentDict)+', page:'+str(page)+', weight:'+str(weight)
                    
        maxWeight = max(weightTmp)
        max_weight_Ds.append(maxWeight)
        #取完weightTmp後，清空
        weightTmp[:] = []
            
        #one page bug if weight = 0.0 transform to 1
        if maxWeight==0:
            maxWeight = 1
        elif maxWeight > 1:
            maxWeight = 1
        if(len(weightWindowDs[currentDict])==numAlgoWindow):
            weightWindowDs[currentDict].popleft()            
        weightWindowDs[currentDict].append(maxWeight)
        
        currentMean = np.mean(weightWindowDs[currentDict])
#        print 'instNo:'+str(instNo)+',currentMean:'+str(currentMean)+'\n'        
        outputPredictSparse.write('instNo:'+str(instNo)+',currentMean:'+str(currentMean)+'\n')
        #目前的mean小於於之前選dict時的2個標準差，啟動重新選擇字典
#        if((meanCompare-currentMean) > 2*stdCompare):
#        changeThreshold = meanCompare-(changeRefStd*stdCompare)
        changeThreshold = meanCompare-(changeRefStd*meanCompare)
        if(currentMean < changeThreshold ):
            #currentDict,meanCompare,stdCompare,wegightWindowDsTmp,changeModel = reChooseDict(instNo,currentDict,currentMean,weightWindowDs[currentDict],stdCompare,meanCompare,Ds,testDataWindow,numOfAttrs,numAlgoWindow,outputCompare,alpha1Lambda)
            outputPredictSparse.write("<changeThreshold\n")
            currentDict,currentMean,currentStd,wegightWindowDsTmp = reChooseDict(instNo,currentDict,currentMean,weightWindowDs[currentDict],stdCompare,meanCompare,Ds,testDataWindow,numOfAttrs,numAlgoWindow,outputCompare,alpha1Lambda)
            outputPredictSparse.write("Change Dict to " + str(currentDict)+'\n')
            if(currentMean<changeThreshold):
                print 'Re-Train Model'
                outputPredictSparse.write("Re-Train Model\n")
                #load ReTrain Result
                input_file = open(reTrainPredict)
                result = input_file.read()
                algoResults.append(aL.arffLoader())
                algoResults[len(algoResults)-1].load(result)
                                
                currentDict=numOfDicts    #currentDict = NewDict
                for i in range(instNo,numOfInsts):
                    output_f.write(str(currentDict)+'\n')
                    outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[instNo]+'\n')
                break
            else:
                meanCompare = currentMean
                stdCompare = currentStd
                outputPredictSparse.write("After Change Dict to " + str(currentDict) + ":\n" + "mean:"+str(meanCompare)+"\nstd:"+str(stdCompare)+'\n')
            
            weightWindowDs[currentDict] = wegightWindowDsTmp
#            if(changeModel==1):
                         
#        else:
#            print 'currentMean'+str(currentMean)
            
#        output_f.write(str(currentDict)+'\n')
        #use Sparse Learning
#        outputPredictSparse.write(dictLoaders[dictChoose].transactionList[pageInDict]+'\n')
#        outputPredictSparse.write('This algo no this value\n')
        #use result of other algo
#        outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[instNo]+'\n')
        
        
            
    #clear list
    max_weight_Ds[:] = []
    weights[:] = []
    
    for i in range(len(dictInfos)):
        dictInfos[i].clear()
    
    instExecTime = time.time() - instExecTime
    outputExecutionTime.write(str(instExecTime))
    
            
print 'RightInstance use other algorithm with Sparse Learning:' + str(RightInstanceOtherAlgo)
print 'Accuracy:'+str(float(RightInstanceOtherAlgo)/(numOfInsts-numAlgoWindow))

Alltac = time.time()
output_f.write('\nAllTime:' + str(Alltac - Alltic))
print 'AllTime:' + str(Alltac - Alltic)            
        
                    
'''    
    #Choose Dictionary
    if(instNo == numAlgoWindow-1):
        for page in dictInfos[dictNo].keys():
            weight = dictInfos[dictNo][page]
            if weight < alpha1Lambda + 0.5:
                weightTmp.append(weight)
                weight2PageDs[dictNo][weight] = page
        
    elif(instNo > numAlgoWindow):
    
    else:
        continue
        
    for dictNo in range(numOfDicts):
#        print allInfos[dictNo][instNo].values()
        #examine the correctness of weight
        
        weight2PageDs[dictNo].clear()
        
        #no one page bug
        
#        if len(dictChooseUBE) >= 1:
#            continue             
#        for page in dictInfos[dictNo].keys():
#            weight = dictInfos[dictNo][page]
#            if weight < alpha1Lambda + 0.5:
#                weightTmp.append(weight)
#                weight2PageDs[dictNo][weight] = page        
#
##            print "instNo:"+str(instNo)                    
#        max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
#        #取完weightTmp後，清空
#        weightTmp[:] = []
#        #check for same weight
##        if max_weight_Ds[dictNo] in weight2Dict:
##            print instNo
##            print max_weight_Ds[dictNo]
##            print 'error\n'
#        weight2Dict[max_weight_Ds[dictNo]] = dictNo
#        weights.append(max_weight_Ds[dictNo])
        
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
                if weight < alpha1Lambda + 0.1:
                    weightTmp.append(weight)
                    weight2PageDs[dictNo][weight] = page        

#            print "instNo:"+str(instNo)                    
#            max_weight_Ds.append(sorted(weightTmp,reverse=True)[0])
            maxWeight = max(weightTmp)
            max_weight_Ds.append(maxWeight)
            
            #one page bug if weight = 0.0 transform to 1
            if maxWeight==0:
                maxWeight = 1
            if(len(weightWindowDs)==numAlgoWindow):
                weightWindowDs.popleft()            
            weightWindowDs[dictNo].append(maxWeight)
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
'''