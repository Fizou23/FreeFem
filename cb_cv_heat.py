#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data for the P1 fe
data1 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P1.txt")
data11 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P1_noise.txt")
h_values_P1 = data1[:, 0]
h1_norms_P1 = data1[:, 1]
h1_norms_P1_noise = data11[:, 1]

#data for the P2 fe
data2 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P2.txt")
data22 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P2_noise.txt")
h_values_P2 = data2[:, 0]
h1_norms_P2 = data2[:, 1]
h1_norms_P2_noise = data22[:, 1]

#data for the P3 fe
data3 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P3.txt")
data33 = np.loadtxt("C:/Users/khfz2/Desktop/Pro_docs/2A/PRe/FreeFem2/FreeFem/heat_1D_P3_noise.txt")
h_values_P3 = data3[:, 0]
h1_norms_P3 = data3[:, 1]
h1_norms_P3_noise = data33[:, 1]

#compute the slope for each plot using a linear regression
model = LinearRegression()

#P1
model.fit(np.log10(h_values_P1).reshape((-1, 1)),np.log10(h1_norms_P1))
print(f"slope for P1: {model.coef_[0]}")
slope1 = model.coef_[0]
intercept1 = model.intercept_

model.fit(np.log10(h_values_P1).reshape((-1, 1)),np.log10(h1_norms_P1_noise))
print(f"slope for P1 noise: {model.coef_[0]}")
slope11 = model.coef_[0]
intercept11 = model.intercept_

#P2
model.fit(np.log10(h_values_P2).reshape((-1, 1)),np.log10(h1_norms_P2))
print(f"slope for P2: {model.coef_[0]}")
slope2 = model.coef_[0]
intercept2 = model.intercept_

model.fit(np.log10(h_values_P2).reshape((-1, 1)),np.log10(h1_norms_P2_noise))
print(f"slope for P2 noise: {model.coef_[0]}")
slope22 = model.coef_[0]
intercept22 = model.intercept_

#P3
model.fit(np.log10(h_values_P3).reshape((-1, 1)),np.log10(h1_norms_P3))
print(f"slope for P3: {model.coef_[0]}")
slope3 = model.coef_[0]
intercept3 = model.intercept_

model.fit(np.log10(h_values_P3).reshape((-1, 1)),np.log10(h1_norms_P3_noise))
print(f"slope for P3 noise: {model.coef_[0]}")
slope33 = model.coef_[0]
intercept33 = model.intercept_

# Create a log-log plot
#plt.subplot(1,3,1)
plt.loglog(h_values_P1, h1_norms_P1,'b',marker="o",label="P1 Sans bruit")
plt.loglog(h_values_P1, h1_norms_P1_noise+0.01,'b--',marker="+",label="P1 Avec bruit")
plt.plot(h_values_P1, 10**(intercept1) * h_values_P1**(slope1), 'k--')
#plt.plot(h_values_P1, h_values_P1**(-0.5), 'k--',label="Pente = 1")
#plt.xlabel("h")
#plt.xlabel("Erreur P1")
plt.legend()
#plt.subplot(1,3,2)
plt.loglog(h_values_P2, h1_norms_P2,'r',marker="o",label="P2 Sans bruit")
plt.loglog(h_values_P2, h1_norms_P2_noise+0.0005,'r--',marker="+",label="P2 Avec bruit")
plt.plot(h_values_P2, 10**(intercept2) * h_values_P2**(slope2), 'k--')
#plt.plot(h_values_P2, h_values_P2**(-1), 'k--',label="Pente = 2")
plt.title('Norme H1 vs Nombre de degrés de liberté')
#plt.xlabel("Erreur P2")
plt.legend()
#plt.subplot(1,3,3)
plt.loglog(h_values_P3, h1_norms_P3,'g',marker="o",label="P3 Sans bruit")
plt.loglog(h_values_P3, h1_norms_P3_noise,'g--',marker="+",label="P3 Avec bruit")
plt.plot(h_values_P3, 10**(intercept3) * h_values_P3**(slope3), 'k--')
#plt.plot(h_values_P3, h_values_P3**(-1.5), 'k--', label="Pente = 3")
#plt.xlabel("Erreur P3")
plt.legend()

# Display the plot
plt.show()