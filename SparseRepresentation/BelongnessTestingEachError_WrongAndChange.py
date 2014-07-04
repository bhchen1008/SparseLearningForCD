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
import operator

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
        
def errorCaculation(test,dictContent,dictInfo,alpha1Lambda):
    #dictInfo裏頭即為page與weight的Dict
    tmpData = []
    for attrIndex in range(test.shape[0]):
        negative = False
        if(test[attrIndex][0]<0):
            negative = True
        tmpData.append(test[attrIndex][0])
        tmpWeight = []
        sorted_dictInfo = sorted(dictInfo.items(),reverse=True, key=lambda x: x[1])
        for page in dictInfo.keys():
            pageWeight = dictInfo[page]
#        for pageTmp in sorted_dictInfo:
#            page = pageTmp[0]
#            pageWeight = pageTmp[1]
            if(pageWeight < alpha1Lambda+0.02):
                if(pageWeight > alpha1Lambda):
                    pageWeight = 1
                elif(pageWeight==0 and len(dictInfo)==1):
                    pageWeight = 1
                tmpWeight.append(pageWeight)
                partitionWeight = dictContent[attrIndex][page]*pageWeight
                tmpData[attrIndex] -= partitionWeight
                if(tmpData[attrIndex]<0 and not(negative)):
                    tmpData[attrIndex] += partitionWeight
                if(tmpData[attrIndex]>0 and negative):
                    tmpData[attrIndex] += partitionWeight
    return sum(tmpData)*(1-max(tmpWeight))
            
        
    
        
#def reChooseDict(instNo,currMean,dicts,testWindow,numAttrs,numInsts,outputCompare,Lambda):
def reChooseDict(instNo,currDict,currMean,currStd,dicts,testDWindow,numAttrs,numInsts,outputCompare,Lambda):
    outputCompare.write('Re-Choose Dictionary!\n')
    errorWindowDsRe = []
#    tmpsRe = []
    tmpsRe = 0
    splitByEnterDsRe = 0
    numDicts = len(dicts)
    for i in range(numDicts):
        errorWindowDsRe.append(deque())
    dictErrorMeanRe = {}
    testWindow = aL.arffLoader()
    testD = testWindow.fortranArrayPara(testDWindow,numAttrs,numInsts)
    testD = np.asfortranarray(testD / np.tile(np.sqrt((testD*testD).sum(axis=0)),(testD.shape[0],1)))
    dictInfosRe = {}
    for dictNo in range(numDicts):
        if(dictNo==currDict):
            continue
        alpha_lasso_m1_Ds_batch = spams.lasso(testD,dicts[dictNo],return_reg_path = False,lambda1 = Lambda,pos=True,mode=0)
        
        for j in range(alpha_lasso_m1_Ds_batch.shape[1]):
            currentTestDataRe = testDWindow[j]
            testX = testLoader.singleFortranArray(currentTestDataRe)
            testX = np.asfortranarray(testX / np.tile(np.sqrt((testX*testX).sum(axis=0)),(testX.shape[0],1)))
            
            dictInfosRe.clear()
#            tmpsRe.append(str(alpha_lasso_m1_Ds_batch.getcol(j)))
            tmpsRe = str(alpha_lasso_m1_Ds_batch.getcol(j))
            #instNo+j才是正確的instance Number
#            outputCompare.write(str(instNo-len(numAlgoWindow)+j)+'-D'+str(dictNo)+':'+tmpsRe[j]+'\n\n')
            outputCompare.write(str(instNo-len(testDWindow)+j)+'-D'+str(dictNo)+':'+tmpsRe+'\n\n')
            #split
            #print tmps[i].split('\n')
#            splitByEnterDsRe.append(tmpsRe[j].split('\n'))
#            splitByEnterDsRe.append(tmpsRe.split('\n'))
#            splitByEnterDsRe = tmpsRe[j].split('\n')
            splitByEnterDsRe = tmpsRe.split('\n')
            
            
            #        splitByEnterD1 = tmp1.split('\n')
#            weightTmp = []
#            for line in splitByEnterDsRe[i]:
            for line in splitByEnterDsRe:
                line = line.strip()
                #mapping page
                pageNo = int(line.split(',')[0].split('(')[1].strip())
                #weight of mapping page
                weight = float(line.split(',')[1].split(')')[1].strip())
                #到這邊已把pageNo與weight Parse出
                dictInfosRe[pageNo] = weight
            
            error = errorCaculation(testX, dicts[dictNo], dictInfosRe, Lambda)
            
            errorWindowDsRe[dictNo].append(error)
            
    #choose dict
    for dictNo in range(numOfDicts):
        dictErrorMeanRe[dictNo] = np.mean(errorWindowDs[dictNo])
    maxWeightDict,meanCompareRe = max(dictErrorMeanRe.iteritems(), key=lambda x:x[1])
    stdCompareRe = np.std(errorWindowDs[currentDict])
    
    return maxWeightDict,meanCompareRe,stdCompareRe
    #tmp

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

numAlgoWindow = int(sys.argv[len(sys.argv)-7])
threshold = float(sys.argv[len(sys.argv)-6])

#DictionaryChoose
output_f = open(sys.argv[len(sys.argv)-5],'w')
#PredictReference output(Sparse Learning)
outputPredictSparse = open(sys.argv[len(sys.argv)-4],'w')
#result of other algorithm with Sparse Learning
outputPredictOtherAlgo = open(sys.argv[len(sys.argv)-3],'w')
#Compare Output
outputCompare = open(sys.argv[len(sys.argv)-2],'w')
#Change Threshold
changeRefStd = float(sys.argv[len(sys.argv)-1])

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
#List for save error of window
errorWindowDs = []

for i in range(numOfDicts):
    dictInfos.append({})
    weight2PageDs.append({})
    errorWindowDs.append(deque())


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
currentTestData = 0
dictErrorMean = {}
std2Dict = {}
#目前使用的字典
currentDict = 0
meanCompare = 0
stdCompare = 0

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
            alpha_lasso_m1_Ds = spams.lasso(X,Ds[dictNo],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0)
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
                if weight <= alpha1Lambda + 0.02:
                    weightTmp.append(weight)
                    weight2PageDs[dictNo][weight] = page
#                if weight >= 1:
#                    print 'InstNo:'+str(instNo)+', DictNo:'+str(dictNo)+', page:'+str(page)+', weight:'+str(weight)
            
            error = errorCaculation(X, Ds[dictNo], dictInfos[dictNo], alpha1Lambda)
                    
            errorWindowDs[dictNo].append(error)
        #當sliding window的資料已滿，即可開始選擇初始字典            
        if(instNo==(numAlgoWindow-1)):
            for dictNo in range(numOfDicts):
                dictErrorMean[dictNo] = np.mean(errorWindowDs[dictNo])
            currentDict,meanCompare = min(dictErrorMean.iteritems(), key=lambda x:x[1])
            stdCompare = np.std(errorWindowDs[currentDict])
            outputPredictSparse.write('Initial:\n'+'currentDict:'+str(currentDict)+'\n'+'meanCompare:'+str(meanCompare)+'\n'+'StdCompare:'+str(stdCompare)+'\n\n')
#            currentDict = max(dictErrorMean.iteritems(), key=operator.itemgetter(1))[0]
            for i in range(numAlgoWindow):
                output_f.write(str(currentDict)+'\n')
                outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[i]+'\n')

            
    #initial done
    else:
        alpha_lasso_m1_Ds = spams.lasso(X,Ds[currentDict],return_reg_path = False,lambda1 = alpha1Lambda,pos=True,mode=0)
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
            if weight < alpha1Lambda + 0.02:
                weightTmp.append(weight)
                weight2PageDs[currentDict][weight] = page
#            if weight >= 1:
#                    print 'InstNo:'+str(instNo)+', DictNo:'+str(currentDict)+', page:'+str(page)+', weight:'+str(weight)
        error = errorCaculation(X, Ds[currentDict], dictInfos[currentDict], alpha1Lambda)
                    
#        errorWindowDs[dictNo].append(error)
#                    
#        maxWeight = max(weightTmp)
#        max_weight_Ds.append(maxWeight)
#            
#        #one page bug if weight = 0.0 transform to 1
#        if maxWeight==0:
#            maxWeight = 1
#        elif maxWeight > 1:
#            maxWeight = 1
        if(len(errorWindowDs[currentDict])==numAlgoWindow):
            errorWindowDs[currentDict].popleft()            
        errorWindowDs[currentDict].append(error)
        
        
        currentMean = np.mean(errorWindowDs[currentDict])
        outputPredictSparse.write('instNo:'+str(instNo)+',currentMean:'+str(currentMean)+'\n')
        #目前的mean小於於之前選dict時的2個標準差，啟動重新選擇字典
#        if((meanCompare-currentMean) > 2*stdCompare):
        if((currentMean - meanCompare) > changeRefStd*stdCompare):
#        if(meanCompare > currentMean):#+0.002):
            currentDict,currentMean,stdCompare = reChooseDict(instNo,currentDict,currentMean,stdCompare,Ds,testDataWindow,numOfAttrs,numAlgoWindow,outputCompare,alpha1Lambda)
#        else:
#            print 'currentMean'+str(currentMean)
            
        output_f.write(str(currentDict)+'\n')
        #use Sparse Learning
#        outputPredictSparse.write(dictLoaders[dictChoose].transactionList[pageInDict]+'\n')
#        outputPredictSparse.write('This algo no this value\n')
        #use result of other algo
        outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[instNo]+'\n')
        
        if int(algoResults[currentDict].className[instNo]) == int(testLoader.className[instNo]):
            RightInstanceOtherAlgo += 1
    
    
#    outputPredictOtherAlgo.write(algoResults[currentDict].transactionList[instNo]+'\n')
    #clear list
    max_weight_Ds[:] = []
    weights[:] = []
    
    for i in range(len(dictInfos)):
        dictInfos[i].clear()
    weight2Dict.clear()
    dictChooseUBE.clear()
            
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
            if(len(errorWindowDs)==numAlgoWindow):
                errorWindowDs.popleft()            
            errorWindowDs[dictNo].append(maxWeight)
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