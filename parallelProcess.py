#!/usr/bin/env python3
import itertools
import multiprocessing
import numpy
import numpy as np
import tenseal as ts
import base64
import scipy.io
import math
from multiprocessing import Pool

from matplotlib import pyplot as plt

import time
start_time = time.time()



V = scipy.io.loadmat('V.mat')
V=V.get('V')
V=V*10;
# Get the Channel decomposition matrix
Q = V@(V.transpose())



def write_data(file_name, data):
    if type(data) == bytes:
        #bytes to base64
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    #base64 to bytes
    return base64.b64decode(data)



# Retreive keys from storage
context = ts.context_from(read_data("data/public.txt"))
#client has the secret key
context = ts.context_from(read_data("data/secret.txt"))

# Results =[]
#
#
# X=(numpy.array(Results)).transpose()
# print(X)
#
# np.save('data/X_enc_approx.npy', X) # save
# X = np.load('data/X.npy') # load


#Generate values for each parameter
c = range(151)
r = range(302)

#Generate a list of tuples where each tuple is a combination of parameters.
#The list will contain all possible combinations of parameters.
paramlist = list(itertools.product(c,r))

#A function which will process a tuple of parameters
def func(params):
    row = []
    c = params[0]
    r = params[1]
    print(c*r)
    print()


    s=str(c)

    tempQ_proto = read_data("data/tempQ_"+s+"_.txt")
    tempQ = ts.lazy_ckks_vector_from(tempQ_proto)
    tempQ.link_context(context)

    d1_proto = read_data("data/d1_"+s+"_.txt")
    d1 = ts.lazy_ckks_vector_from(d1_proto)
    d1.link_context(context)

    s=str(r)

    test1_proto = read_data("data/test1"+s+"_.txt")
    test1 = ts.lazy_ckks_vector_from(test1_proto)
    test1.link_context(context)

    test2_proto = read_data("data/test2"+s+"_.txt")
    test2 = ts.lazy_ckks_vector_from(test2_proto)
    test2.link_context(context)

    test3_proto = read_data("data/test3"+s+"_.txt")
    test3 = ts.lazy_ckks_vector_from(test3_proto)
    test3.link_context(context)


        # Server compute test x tQ
    test3Q_proto = read_data("data/test3Q"+s+"_.txt")
    test3Q = ts.lazy_ckks_vector_from(test3Q_proto)
    test3Q.link_context(context)


        # Server compute test x tQ
    t1=test1.dot(tempQ)

    t21=test2.dot(tempQ)

    t22=d1

        # t231=test3.matmul(Q)
        # t23=t231.dot(test3)

    t23 = test3Q

    finalValue=t1 + t21 * t22 * t23

        # client decrypt and get the result
    finalValueDecrypted=finalValue.decrypt()[0]
    row.append(r)
    row.append(c)
    row.append(finalValueDecrypted)


    return row

# #Generate processes equal to the number of cores
# pool = multiprocessing.Pool()
#
# #Distribute the parameter sets evenly across the cores
# res  = pool.map(func,paramlist)
#
#
#
# shape = (len(r),len(c))
# X=np.zeros(shape)
#
# for row in res:
#     r=int(row[0])
#     c=int(row[1])
#     X[r,c] = row[2]
#
#
# np.save('data/X_appro_CKKS_Parrallel.npy', X) # save
# print("--- %s seconds ---" % (time.time() - start_time))



X = np.load('data163846035Scaledby10/X_appro_CKKS_Parrallel.npy') # load
print(X.min(),X.max())
# print(np.shape(X))




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
# Synthetic Data
theta= Result[0,]/100
FAR = Result[1,]
FRR = Result[2,]
print(FAR[25])
print(FRR[25])
Accuracy=Result[3,]
# Plotting on a figure
fig = plt.figure()
plt.plot(theta, FAR,'r')
plt.plot(theta,FRR,'g')
plt.plot(theta,Accuracy,'b')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 100)
plt.title("FAR Vs FRR Vs Accuracy")
plt.xlabel("Threshold")
plt.ylabel("Percentage")
fig.savefig("approxCKKS_Figure_CKKS.png")





