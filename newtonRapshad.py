import math
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def newton_inverse(d,x_0):
    row=[]
    for i in range(2):
        y = (3*x_0 - d*x_0*x_0*x_0)/2
        x_0=y
        row.append(x_0)
    return row

a=0.000001
b=0.000007


x_0=0.45*(1/sqrt(a) + 1/sqrt(b))
x_1=0.5*(1/sqrt(a) + 1/sqrt(b))


def inverseSquareRoot(x):
    Result=[]
    for i in range(101):
        d=i*(b-a)/100 + a
        col=newton_inverse(d,x)
        col.append(1/sqrt(d))
        col.append(d)
        Result.append(col)
    return Result

X = np.load('data/DenominatorD.npy') # load
fig = plt.figure()
# ax = sns.distplot(X)
plt.hist(X*10000, bins=100)  # arguments are passed to np.histogram

plt.title("Distribution Vs Error % of Inverse Square Root")
plt.xlabel("Values")
plt.ylabel("Error Percentage")


Result = inverseSquareRoot(x_0)
Result = np.array(Result)
iter_1= 100* abs(Result[:,0] - Result[:,2])/Result[:,2]
iter_2 =100* abs(Result[:,0] - Result[:,1])/Result[:,1]
# actual = Result[:,2]
d=Result[:,2]
# plt.plot(d, actual,'r')
plt.plot(d, iter_1,'g+')
plt.plot(d,iter_2,'bv')

Result = inverseSquareRoot(x_1)
Result = np.array(Result)
iter_1= 100* abs(Result[:,0] - Result[:,2])/Result[:,2]
iter_2 =100* abs(Result[:,0] - Result[:,1])/Result[:,1]
# actual = Result[:,2]
d=Result[:,2]
# plt.plot(d, actual,'r')
plt.plot(d, iter_1,'g+')
plt.plot(d,iter_2,'bv')


plt.legend(['One Iteration','Two Iterations','One Iteration','Two Iterations'])
# plt.xlim(400, 900)
# plt.ylim(0, 20)
plt.grid()




fig.savefig("Newton.png")
print('Done')



