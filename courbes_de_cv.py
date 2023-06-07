#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data for the P1 fe
data1 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFEM/courbe_de_convergence_P1.txt")
n_values_P1 = data1[:, 0]
h1_norms_P1 = data1[:, 1]

#data for the P2 fe
data2 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFEM/courbe_de_convergence_P2.txt")
n_values_P2 = data2[:, 0]
h1_norms_P2 = data2[:, 1]

#data for the P3 fe
data3 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFEM/courbe_de_convergence_P3.txt")
n_values_P3 = data3[:, 0]
h1_norms_P3 = data3[:, 1]

# Create a log-log plot
plt.subplot(1,3,1)
plt.loglog(n_values_P1, h1_norms_P1,'b',label="P1")
plt.xlabel("P1 error")
plt.subplot(1,3,2)
plt.loglog(n_values_P2, h1_norms_P2,'r', label="P2")
plt.title('H1 Norm of Error vs. Number of Subdivisions')
plt.xlabel("P2 error")
plt.subplot(1,3,3)
plt.loglog(n_values_P3, h1_norms_P3,'g', label="P3")
plt.xlabel("P3 error")

#compute the slope for each plot using a linear regression
model = LinearRegression()
model.fit(np.log(n_values_P1).reshape((-1, 1)),np.log(h1_norms_P1))
print(print(f"slope for P1: {model.coef_}"))
model.fit(np.log(n_values_P2).reshape((-1, 1)),np.log(h1_norms_P2))
print(print(f"slope for P2: {model.coef_}"))
model.fit(np.log(n_values_P3).reshape((-1, 1)),np.log(h1_norms_P3))
print(print(f"slope for P3: {model.coef_}"))

# Display the plot
plt.show()