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

# # import channel compensation matrix V from Matlab
# # dim V (200,200)
# V = scipy.io.loadmat('V.mat')
# V=V.get('V')
#
# # import test iVectors from matlab mat files
# # dim (200,302)
# test_ivs=scipy.io.loadmat('test_ivs.mat')
# T=test_ivs.get('test_ivs')
#
# # import template iVectors from matlab mat files
# # dim (200,151)
# model_ivs1=scipy.io.loadmat('model_ivs1.mat')
# M=model_ivs1.get('model_ivs1')
#
#
# # Compute T'x V x V' x M
# # dim of X (302,151)
# X=((T.transpose()@V)@V.transpose())@M
#
#
# for c in range(0,151):
#     print(c)
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
#
#             X[r,c]=X[r,c] / (d1*d2)
#             # arr[r,c] = x1 * x2
#
#
# np.save('data81924134Scaledby10/scores_plain.npy', X) # save
X = np.load('data81924134Scaledby10/scores_plain.npy') # load

fig = plt.figure()

for i in range(1):
    X=X
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
    print(FAR[33])
    print(FRR[33])
    Accuracy=Result[3,]
    # Plotting on a figure
    plt.plot(theta, FAR,'r.')
    plt.plot(theta,FRR,'g--')
    plt.plot(theta,Accuracy,'b')
    plt.legend(['False Acceptance Rate','False Rejection Rate','Accuracy'])
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.grid()

plt.title("FAR Vs FRR Vs Accuracy")
plt.xlabel("Threshold")
plt.ylabel("Percentage")

fig.savefig("BasicFigure.png")

