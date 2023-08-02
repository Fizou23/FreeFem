import numpy as np
import matplotlib.pyplot as plt
import math

samples = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/freefemfuncs/normal_dist.txt")
Z = samples
#Z2 = samples[:, 1]

#mean, stddev and plotting interval
mu=0
sigma=0.01
alpha=5*sigma
#bins
bns=150

x=(np.linspace(mu-alpha,mu+alpha,200*math.ceil(sigma)))
y=1/(((2*np.pi)**0.5)*sigma)*np.exp((((x-mu)/sigma)**2)*(-0.5))

#plt.subplot(1,2,1)
plt.hist(mu+(sigma*Z),bins=bns,density=True)

plt.plot(x,y)

'''plt.subplot(1,2,2)
plt.hist(mu+(sigma*Z2),bins=bns,density=True)
plt.plot(x,y)
'''
plt.show()