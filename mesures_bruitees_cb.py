#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data for the P1 fe
data1 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/bruit_P1.txt")
h_values_P1 = data1[:, 0]
h1_norms_P1 = data1[:, 1]

#data for the P2 fe
data2 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/bruit_P2.txt")
h_values_P2 = data2[:, 0]
h1_norms_P2 = data2[:, 1]

#data for the P3 fe
data3 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/bruit_P3.txt")
h_values_P3 = data3[:, 0]
h1_norms_P3 = data3[:, 1]

#compute the slope for each plot using a linear regression
model = LinearRegression()

#P1
model.fit(np.log10(h_values_P1).reshape((-1, 1)),np.log10(h1_norms_P1))
print(f"slope for P1: {model.coef_[0]}")
slope1 = model.coef_[0]
intercept1 = model.intercept_

#P2
model.fit(np.log10(h_values_P2).reshape((-1, 1)),np.log10(h1_norms_P2))
print(f"slope for P2: {model.coef_[0]}")
slope2 = model.coef_[0]
intercept2 = model.intercept_

#P3
model.fit(np.log10(h_values_P3).reshape((-1, 1)),np.log10(h1_norms_P3))
print(f"slope for P3: {model.coef_[0]}")
slope3 = model.coef_[0]
intercept3 = model.intercept_

# Create a log-log plot
plt.subplot(1,3,1)
plt.loglog(h_values_P1, h1_norms_P1,'b',label="P1")
plt.plot(h_values_P1, 10**(intercept1) * h_values_P1**(slope1), 'r--', label='Regression Line P1')
plt.xlabel("P1 error")
plt.subplot(1,3,2)
plt.loglog(h_values_P2, h1_norms_P2,'r', label="P2")
plt.plot(h_values_P2, 10**(intercept2) * h_values_P2**(slope2), 'r--', label='Regression Line P1')
plt.title('H1 Norm of Error vs. h')
plt.xlabel("P2 error")
plt.subplot(1,3,3)
plt.loglog(h_values_P3, h1_norms_P3,'g', label="P3")
plt.plot(h_values_P3, 10**(intercept3) * h_values_P3**(slope3), 'r--', label='Regression Line P1')
plt.xlabel("P3 error")

# Display the plot
plt.show()