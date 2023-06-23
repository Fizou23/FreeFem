#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import the data
data1 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFem2/FreeFem/courbe_de_convergence_da_P1.txt")
h_values1 = data1[:, 0]
h1_norms1 = data1[:, 1]

data2 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFem2/FreeFem/courbe_de_convergence_da_P2.txt")
h_values2 = data2[:, 0]
h1_norms2 = data2[:, 1]

data3 = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFem2/FreeFem/courbe_de_convergence_da_P3.txt")
h_values3 = data3[:, 0]
h1_norms3 = data3[:, 1]

# Create a log-log plot
plt.loglog(h_values, h1_norms,label='loglog plot')
plt.xlabel("h")
plt.ylabel("H1 norm of the error in B")




#compute the slope using a linear regression
model = LinearRegression()
model.fit(np.log10(h_values).reshape((-1, 1)),np.log10(h1_norms))
print(f"slope = {model.coef_[0]}")
slope = model.coef_[0]
intercept = model.intercept_
plt.plot(h_values, 10**(intercept) * h_values**(slope), 'r--', label='Regression Line')
plt.plot(h_values, h_values**(0.5), 'k--', label='slope = 0.5')
plt.plot(h_values, h_values, 'g--', label='slope = 1')
plt.legend()
plt.grid(True)
plt.show()