import numpy as np
import matplotlib.pyplot as plt
import math

samples = np.loadtxt("C:/Users/khfz2/Desktop/freefemfuncs/normal_dist.txt")
Z = samples
#Z2 = samples[:, 1]

#mean, stddev and plotting interval
mu=0
sigma=5
alpha=5*math.ceil(sigma)
#bins
bns=150

x=(np.linspace(mu-alpha,mu+alpha,200*alpha))
y=1/(((2*np.pi)**0.5)*sigma)*np.exp((((x-mu)/sigma)**2)*(-0.5))

#plt.subplot(1,2,1)
#plt.hist(mu+(sigma*Z),bins=bns,density=True)

plt.plot(x,y)

'''plt.subplot(1,2,2)
plt.hist(mu+(sigma*Z2),bins=bns,density=True)
plt.plot(x,y)
'''
plt.show()