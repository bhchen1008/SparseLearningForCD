#encoding:utf-8
'''
Created on 2014/3/18

@author: bhchen
'''
import numpy as np
class arffLoader:
    def __init__(self):
        self.classIndex = 0     #index of class
        self.attrName = []      #index map to AttrName
        self.arffAttribute = ""
        self.transactionList = []  #transactionList
        self.transactionContentList = []   #transaction list
        self.className = []
        self.numAttribute = 0
        self.numInstance = 0
    def load(self,content):
        splitByAttr = content.split('@attribute')
        attrIndex = 0
        attrName = self.attrName
        #
        #    Load all attribute name
        #    get index of class
        #
        for j in range(1,len(splitByAttr)-1):           
            attrName.append(splitByAttr[j].split()[0])
            attrIndex += 1
        self.numAttribute = attrIndex   #classIndex = numAttribute       
        self.classIndex = attrIndex
        
        #
        #    Save the value in transaction to transaction list
        #    Save transaction list into transactionList list
        #
        self.arffAttribute = content.split('@data')[0].strip()
        transContent = content.split('@data')[1].strip()
        transaction = transContent.split('\n')
#        transactionContentList = self.transactionContentList
#        numInstance = self.numInstance
        self.numInstance = len(transaction)
        for tran in transaction:
            self.transactionList.append(tran)
            tranList = []
            tranContents = tran.split(',')  #save every value of attribute into list
#            for tranC in tranContents:
#                tranList.append(tranC)
#old method spend time-cost
#            for tranNum in range(len(tranContents)):
#                tranList.append(tranContents[tranNum])
#                if tranNum == len(tranContents)-1:
#                    self.className.append(tranContents[tranNum])
            
            self.className.append(tranContents[self.classIndex])
#            self.transactionContentList.append(tranList)
            self.transactionContentList.append(tranContents)
            i = 1
    def fortranArray(self,twoDList):
        tmpArray = []
        for trans in twoDList:
            for j in range(len(trans)-1):
                tmpArray.append(float(trans[j]))
        array = np.array(tmpArray)
        fortranArray = array.reshape(self.classIndex, self.numInstance, order='F')
        return fortranArray
    def fortranArrayPara(self,twoDList,numAttribute,numInstance):
        tmpArray = []
        for trans in twoDList:
            for j in range(len(trans)-1):
                tmpArray.append(float(trans[j]))
        array = np.array(tmpArray)
        fortranArray = array.reshape(numAttribute, numInstance, order='F')
        return fortranArray
    def singleFortranArray(self,oneDList):
        tmpArray = []
        for j in range(len(oneDList)-1):
            tmpArray.append(float(oneDList[j]))
        array = np.array(tmpArray)
        fortranArray = array.reshape(self.classIndex, 1, order='F')
        return fortranArray