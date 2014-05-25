import spams
import numpy as np
import time
m = 100;n = 10;nD = 5
np.random.seed(0)
X = np.asfortranarray(np.random.normal(size=(m,n)))
print X
X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
D1 = np.asfortranarray(np.random.normal(size=(100,400)))
D1 = np.asfortranarray(D1 / np.tile(np.sqrt((D1*D1).sum(axis=0)),(D1.shape[0],1)))

tic = time.time()
alpha = spams.lasso(X,D1 = D1,return_reg_path = False,lambda1 = 0.15,pos= True)
tac = time.time()
t = tac - tic


print 'alpha:'
print alpha

print 'time'+str(t)
