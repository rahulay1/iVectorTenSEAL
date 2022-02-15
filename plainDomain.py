"""
This is iVector implementation
Only coveres the inferencing part
Training was done on TIMIT dataset using the first 151 speakers
Each speaker got 10 utterances - 8 were used to train the model; 2 considered here for testing
"""

from multiprocessing import Pool
import multiprocessing
import itertools
import math
from math import sqrt
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

# import channel compensation matrix V from Matlab
# dim V (200,200)
V = scipy.io.loadmat('V.mat')
V=V.get('V')
V=V*10

# import test iVectors from matlab mat files
# dim (200,302)
test_ivs=scipy.io.loadmat('test_ivs.mat')
T=test_ivs.get('test_ivs')
T=T*10

# T=np.ones([200,302])
T=np.random.rand(200,302)


# import template iVectors from matlab mat files
# dim (200,151)
model_ivs1=scipy.io.loadmat('model_ivs1.mat')
M=model_ivs1.get('model_ivs1')
M=M*10


# Compute T'x V x V' x M
# dim of X (302,151)
X=((T.transpose()@V)@V.transpose())@M

# Normalising the each element of X by sqrt(t'xVxV'xt) and sqrt(m'xVxV'xm)

shape = (302,151)
# X_approximation=np.zeros(shape)

a=000000090.00
b=000000800.0
x_initial=0.5*(1/sqrt(a) + 1/sqrt(b))

def newton_inverse(d):
    x_0=0.5*(1/sqrt(a) + 1/sqrt(b))
    for i in range(2):
        y = (3*x_0 - d*x_0*x_0*x_0)/2
        x_0=y
    return x_0

#Generate values for each parameter
c = range(151)
r = range(302)

#Generate a list of tuples where each tuple is a combination of parameters.
#The list will contain all possible combinations of parameters.
paramlist = list(itertools.product(c,r))
def func(params):
    row =[]
    c = params[0]
    r = params[1]


    testV=T[:,r]
    templateV=M[:,c]
    x1=((testV.transpose()@V)@V.transpose())@testV
    x2=((templateV.transpose()@V)@V.transpose())@templateV
    d1=math.sqrt(x1)
    d2=math.sqrt(x2)

    W=V@(V.transpose())
    X_actual=X[r,c] /(d1*d2)
    X_approximation=X[r,c] * (newton_inverse(x1 * x2))
    denominatorDistribution=1/(d1*d2)


    row.append(r)
    row.append(c)
    row.append(X_actual)
    row.append(denominatorDistribution)


    return row



pool = multiprocessing.Pool()

#Distribute the parameter sets evenly across the cores
res  = pool.map(func,paramlist)

shape = (len(r),len(c))
X=np.zeros(shape)
DenominatorD=np.zeros(shape)

for row in res:
    r=int(row[0])
    c=int(row[1])
    X[r,c] = row[2]
    DenominatorD[r,c]=row[3]

np.save('data/DenominatorD.npy', DenominatorD) # save
np.save('data/X.npy', X) # save
Xone = np.load('data/Xones.npy') # load
Xrand = np.load('data/Xrandom.npy') # load
# print(X.max())
# print(X.min())

# fig = plt.figure()
# plt.hist(X, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# fig.savefig("denominatorDistribution.png")

# for c in range(0,151):
#     for r in range(0,302):
#             testV=T[:,r]
#             templateV=M[:,c]
#             x1=((testV.transpose()@V)@V.transpose())@testV
#             x2=((templateV.transpose()@V)@V.transpose())@templateV
#             d1=math.sqrt(x1)
#             d2=math.sqrt(x2)
#
#             W=V@(V.transpose())
#
#             # X_approximation[r,c]=X[r,c] * (newton_inverse(x1 * x2))
#             t0 = 1.5*x_initial*X[r,c]
#
#
#             t11 = (0.5*x_initial*x_initial*x_initial*testV.transpose()) @W
#             t12=t11@templateV
#
#             t13 = templateV.transpose()@W
#             t14=t13@templateV
#
#             t15 = testV.transpose()@W
#             t16=t15@testV
#
#             # print(t0)
#             # print(t12)
#             # print(t14)
#             # print(t16)
#
#             # X_approximation_2= t0 - t12 * t14 * t16
#             # X_approximation[r,c]=X[r,c] * (newton_inverse(x1 * x2))
#             X[r,c]=X[r,c] / (d1*d2)
#             # arr[r,c] = x1 * x2
#             print('start')
#             # print(X_approximation_2)
#             # print(X_approximation[r,c])
#             # print(X[r,c])


# print(min(X))
# print(max(X))
#
fig = plt.figure()




X=Xrand


def returnResult(X):
    Result=[[],[],[],[]]
    for n in range(0,1000,10):
        theta=n/1000
        TP=0
        TN=0
        FP=0
        FN=0
        for c in range(0,151):
            for r in range(0,302):
                # print(X[r,c])
                if X[r,c]>theta:
                    if r==2*c or r==2*c+1:
                        TP+=1
                    else:
                        FP+=1
                else:
                     if r==2*c or r==2*c+1:
                        FN+=1
                     else:
                        TN+=1
        FAR=FP/(TN+FP)
        FRR=FN/(TP+FN)
        Accuracy=(TP+TN)/(TP+TN+FP+FN)
        new_col=[[theta],[FAR],[FRR],[Accuracy]]
        Result=np.append(Result,new_col,axis=1)

    Result*=100
    theta= Result[0,]/100
    FAR = Result[1,]
    FRR = Result[2,]

    return theta,FAR,FRR

theta,FAR_1,FRR_1 = returnResult(Xone)
theta,FAR_r,FRR_r = returnResult(Xrand)
plt.plot(theta, FAR_1,'r--')
plt.plot(theta,FRR_1,'g-.')
plt.plot(theta, FAR_r,'b:')
plt.plot(theta,FRR_r,'c.')
plt.legend(['FAR_1','FRR_1','FAR_r','FRR_r'])



plt.title("Random Attacks")
plt.xlabel("Threshold")
plt.ylabel("Percentage")
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 100)
fig.savefig("randomTwos.png")

