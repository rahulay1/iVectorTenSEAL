import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


labels=['Secret \n Key','Enrolled \n Template \n d=100','Enrolled \n Template \n d=200','Enrolled \n Template \n d=1000']
legends = ['N=8192', 'N=16384']

pos = np.arange(len(labels))
bar_width = 0.35


N8192=[1.1,0.6,0.6,0.6]
N16384=[4.2,2.4,2.4,2.4]




plt.bar(pos,N8192,bar_width,fill=False, hatch='..',edgecolor='black')
plt.bar(pos+bar_width,N16384,bar_width,fill=False, hatch='xx',edgecolor='black')
plt.xticks(pos+bar_width/2, labels)
plt.grid()
plt.ylabel('Size in MB', fontsize=14)
plt.title('Size of keys and templates',fontsize=16)
plt.legend(legends,loc=1)
plt.savefig('sizeComparison.png')
