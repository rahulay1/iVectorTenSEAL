import numpy as np
from matplotlib import pyplot as plt


def findFAR(X):
    Result=[[],[],[],[]]
    for n in range(-500,750,10):
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

    return theta, FAR, FRR






# Plotting on a figure
fig = plt.figure()
theta, FAR, FRR = findFAR(np.load('data/XOriginal.npy'))
plt.plot(theta, FAR,'--',color='blue')
# plt.plot(theta, FRR,'--',color='blue')

theta, FAR, FRR = findFAR(np.load('data/Xones.npy'))
plt.plot(theta, FAR,'.-',color='red')
# plt.plot(theta, FRR,'--',color='red')

theta, FAR, FRR = findFAR(np.load('data/Xrandom.npy'))
plt.plot(theta, FAR,'.',color='orange')
# plt.plot(theta, FRR,'--',color='orange')

# theta, FAR , FRR= findFAR(np.load('data81924134Scaledby10/X_appro_CKKS_Parrallel.npy'))
# plt.plot(theta, FAR,':',color='green')
# plt.plot(theta, FRR,'--',color='green')

plt.grid()
plt.xlim(-0.5, 0.75)
plt.ylim(0, 100)


plt.legend(['Baseline','Feature Vectors of Ones ','Random Feature Vectors'])
# plt.legend(['Original FAR','Original FRR','Ones FAR','Ones FRR','Random FAR','Random FRR','CKKS FAR','CKKS FRR'])
plt.title("FAR")
plt.xlabel("Threshold")
plt.ylabel("Percentage")
fig.savefig("FAR.png")





