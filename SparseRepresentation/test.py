import spams
import numpy as np
import time
np.random.seed(0)

#a = np.random.normal(size=(3,3))
#D1 = np.asfortranarray(a)
#print 'a:'
#print a
#num=1
#for i in range(3):
#    for j in range(3):
#        D1[i][j]=num
#        num+=1
#    
#print 'D1:'version
#print D1
#ta = np.array([1,2,3,4,5,6].res, order='F')
#D1=np.array([1.0,4.0,7.0,2.0,5.0,8.0,3.0,6.0,9.0])
D1=np.array([1,4,7,2,5,8,3,6,9.0])
#ta=np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])

D1 = D1.reshape(3,3, order='F')
#D1 = D1.reshape(3,3)
print 'D1:'
print D1
print (D1*D1)
print (D1*D1).sum(axis=0)
print np.sqrt((D1*D1).sum(axis=0))
print D1.shape[0]
print (D1.shape[0],1)
print np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1))
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)))
print 'newD:'
print D1

X=np.array([1.0,4,7])#,1,2,3])#,2.0,5.0,8.0])
X = X.reshape(3,1, order='F')
#X = X.reshape(3,2)
print 'X:'
print X
#X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
print 'newX:'
print X
#for i in range(3):
#    X[i][0]=i*3+1
#for i in range(3):
#    X[i][1]=i*3+2

tic = time.time()
#alpha = spams.lasso(X,D1,return_reg_path = False,lambda1 = 2,pos=True,mode=0)
alpha = spams.lasso(X,D1,return_reg_path = False,lambda1 = 0.1,pos=True,mode=1,verbose=True)
#alpha = spams.omp(X,D1,L=2,lambda1 = None,return_reg_path = False,numThreads = -1)
#alpha = spams.omp(X,D1,L=None,eps= 0,lambda1 = None,return_reg_path = False,numThreads = -1)
tac = time.time()
t = tac - tic
print 'time:'+str(t)


print 'alpha:'
print alpha
