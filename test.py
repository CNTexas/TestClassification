import matplotlib.pyplot as plt
import numpy as np
from Tools import *

x=[1,3,5,7,9]
y=[0.8,0.8,0.8,0.8,0.8]

plt.figure()
plt.ylim(0,1,0.1)
plt.plot(x,y)
plt.show()
#plt.show()
'''

plt.figure(figsize=(8,4))
plt.plot([0,10],[0.8,0.8],'r')
plt.show()

x=[0,10]
y=[0.8,0.8]
plt.figure()
plt.plot(x,y)
plt.show()
#plt.savefig("easyplot2.jpg")
'''