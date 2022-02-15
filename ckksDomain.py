import math

import numpy
import tenseal as ts
import base64
import scipy.io

# context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 4096, coeff_mod_bit_sizes = [37, 17,17, 39])
# context.generate_galois_keys()
# context.global_scale = 2**17
#
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40,40, 60])
context.generate_galois_keys()
context.global_scale = 2**40


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


# # secret_context = context.serialize(save_secret_key = True)
# # write_data("secret.txt", secret_context)
# #
# # context.make_context_public()
# # #drop the secret_key from the context
# # public_context = context.serialize()
# # write_data("public.txt", public_context)

# Load the mat files

# import channel compensation matrix V from Matlab
# dim V (200,200)
V = scipy.io.loadmat('V.mat')
V=V.get('V')

# import test iVectors from matlab mat files
# dim (200,302)
test_ivs=scipy.io.loadmat('test_ivs.mat')
T=test_ivs.get('test_ivs')

# import template iVectors from matlab mat files
# dim (200,151)
model_ivs1=scipy.io.loadmat('model_ivs1.mat')
M=model_ivs1.get('model_ivs1')

# Get the Channel decomposition matrix
Q = V@(V.transpose())


# Server Template Preparation

for c in range (0,1):
    # Encrypt template
    template =ts.ckks_vector(context, M[:,r])
    # Get template x Q <- Q in plain domain
    tempQ=template.matmul(Q)


print(template)

# Results =[]
#
# for c in range(0,1):
#     row = []
#     for r in range (0,1):
#
#         # Prepare Server
#         # Encrypt template
#         template =ts.ckks_vector(context, M[:,r])
#         # Get template x Q <- Q in plain domain
#         tempQ=template.matmul(Q)
#
#         # Client encrypt the test sample
#         test =ts.ckks_vector(context, T[:,c])
#
#         # Server compute test x tQ
#         topValue=test.dot(tempQ)
#
#         # Server computes the normalising factors
#         d1=tempQ.dot(template)
#         d21=test.matmul(Q)
#         d2=d21.dot(test)
#
#         # client decrypt and get the result
#         topDecpted=topValue.decrypt()[0]
#         d1Decrypted=d1.decrypt()[0]
#         d2Decrypted=d2.decrypt()[0]
#
#         theta = topDecpted/math.sqrt(d1Decrypted * d2Decrypted)
#
#         row.append(theta)
#     Results.append(row)
#
# print(type(template))



#
# print((v1@Q)@v2)
#
# enc_v1 = ts.ckks_vector(context, v1)
# enc_v2 = ts.ckks_vector(context, v2)
#
# # enc_v1_proto = enc_v1.serialize()
# # enc_v2_proto = enc_v2.serialize()
# #
# # write_data("enc_v1.txt", enc_v1_proto)
# # write_data("enc_v2.txt", enc_v2_proto)
#
#
#
# x=enc_v1.matmul(Q)
# y=x.dot(enc_v2)
#
# print(y.decrypt())
#
# #
# # v1=[0,0,0,1]
# # print(v1)
# # x1 = ts.ckks_vector(context, v1)
# # x2=x1.mul(x1)
# # x3=x2.mul(x2)
# # # x4=x3.mul(x3)
# #
# # print(x2.decrypt()[0])
# import math
#
# import numpy
# import numpy as np
# import scipy.io
# from matplotlib import pyplot as plt
#
# V = scipy.io.loadmat('V.mat')
# V=V.get('V')
#
# test_ivs=scipy.io.loadmat('test_ivs.mat')
# t=test_ivs.get('test_ivs')
#
#
# model_ivs1=scipy.io.loadmat('model_ivs1.mat')
# m=model_ivs1.get('model_ivs1')
#
#
#
# X=((t.transpose()@V)@V.transpose())@m
#
# for c in range(0,151):
#         for r in range(0,302):
#             testV=t[:,r]
#             templateV=m[:,c]
#             d1=math.sqrt(((testV.transpose()@V)@V.transpose())@testV)
#             d2=math.sqrt(((templateV.transpose()@V)@V.transpose())@templateV)
#
#             X[r,c]=X[r,c]/(d1*d2)
#
#
# Result=[[],[],[],[]]
# for n in range(0,1000,10):
#     theta=n/1000
#     TP=0
#     TN=0
#     FP=0
#     FN=0
#     for c in range(0,151):
#         for r in range(0,302):
#             # print(X[r,c])
#             if X[r,c]>theta:
#                 if r==2*c or r==2*c+1:
#                     TP+=1
#                 else:
#                     FP+=1
#             else:
#                  if r==2*c or r==2*c+1:
#                     FN+=1
#                  else:
#                     TN+=1
#     FAR=FP/(TN+FP)
#     FRR=FN/(TP+FN)
#     Accuracy=(TP+TN)/(TP+TN+FP+FN)
#     new_col=[[theta],[FAR],[FRR],[Accuracy]]
#     Result=np.append(Result,new_col,axis=1)
#
# # Synthetic Data
# theta= Result[0,]
# FAR = Result[1,]
# FRR = Result[2,]
# Accuracy=Result[3,]
# # Plotting on a figure
# fig = plt.figure()
# plt.plot(theta, FAR,'r')
# plt.plot(theta,FRR,'g')
# plt.plot(theta,Accuracy,'b')
# fig.savefig("basicFigure.png")
#
