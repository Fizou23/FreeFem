#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import the data
data = np.loadtxt("C:/Users/khfz2/OneDrive/Bureau/Pro_docs/2A/PRe/FreeFem2/FreeFem/courbe_de_convergence_da.txt")
h_values = data[:, 0]
h1_norms = data[:, 1]

# Create a log-log plot
plt.loglog(h_values, h1_norms,label='loglog plot')
plt.xlabel("h")
plt.ylabel("H1 norm of the error in B")


#compute the slope using a linear regression
model = LinearRegression()
model.fit(np.log10(h_values).reshape((-1, 1)),np.log10(h1_norms))
print(f"slope = {model.coef_}")
slope = model.coef_[0]
intercept = model.intercept_
plt.plot(h_values, 10**(intercept) * h_values**(slope), 'r', label='Regression Line')
plt.legend()
plt.grid(True)
plt.show()