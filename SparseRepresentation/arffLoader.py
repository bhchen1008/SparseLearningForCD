#encoding:utf-8
'''
Created on 2014/3/18

@author: bhchen
'''
import numpy as np
class arffLoader:
    def __init__(self):
        self.classIndex = 1     #index of class
        self.attrName = []      #index map to AttrName
        self.transactionList = []   #transaction list
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
        self.classIndex = attrIndex
        
        #
        #    Save the value in transaction to transaction list
        #    Save transaction list into transactions list
        #
        transContent = content.split('@data')[1].strip()
        transaction = transContent.split('\n')
#        transactionList = self.transactionList
#        numInstance = self.numInstance
        self.numInstance = len(transaction)
        for tran in transaction:
            tranList = []
            tranContents = tran.split(',')  #save every value of attribute into list
            for tranC in tranContents:
                tranList.append(tranC)
            self.transactionList.append(tranList)
    def fortranArray(self,twoDList):
        tmpArray = []
        for trans in twoDList:
            for j in range(len(trans)-1):
                tmpArray.append(float(trans[j]))
        array = np.array(tmpArray)
        fortranArray = array.reshape(self.classIndex, self.numInstance, order='F')
        return fortranArray