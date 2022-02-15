import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
# EER_difference=[0.2, 0.6, 2.2, 2.5, 2.8]
#
#
# labels = ['2 Iter \n Plain', '1 Iter \n Plain', '2 Iter\n N=16384', '1 Iter \nN=16384','1 Iter \nN=8192',]
# bar_width = 0.35

# plt.xticks(range(len(EER_difference)), labels)
# # plt.xlabel('Types')
# plt.ylabel('Difference in EER  (%)',fontsize=14)
# plt.title('EER Difference Compared to Baseline Model',fontsize=16)
# plt.grid()
# plt.bar(range(len(EER_difference)), EER_difference,bar_width,color='blue',edgecolor='black')
# plt.savefig("EERComparision.png")


first=[0.2, 0.6]
second=[2.2, 2.5]
third =[0, 2.8]

labels = ['      2 Iterations to compute \n  inverse square-root', '     1 Iteration to compute \n  inverse square-root']
legends=['Plain','Encrypted (N=16384)','Encrypted (N=8192)']


pos = np.arange(len(labels))
bar_width = 0.3

plt.bar(pos,first,bar_width,fill=False, hatch='///',edgecolor='black')
plt.bar(pos+bar_width,second,bar_width,fill=False, hatch='xx',edgecolor='black')
plt.bar(pos+2*bar_width,third,bar_width,fill=False, hatch='..',edgecolor='black')
plt.xticks(pos+bar_width, labels)
plt.grid()
plt.ylabel('Difference in EER  (%)', fontsize=14)
plt.title('EER Difference Compared to Baseline Model',fontsize=16)
plt.legend(legends,loc=2)
plt.savefig('EERComparision1.png')
