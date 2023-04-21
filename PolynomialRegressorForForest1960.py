# POLINOMIAL REGRESSOR DEL GRUPPO RICE GIRLS FT. NIK
#ENERGY AND CLIMATE CHANGE MODELING AND SCENARIOS;  PROFESSOR MASSIMO TAVONI
#POLITECNICO DI MILANO ANNO ACCADEMICO 2020/2021

#QUESTO SCRIPT SERVE A INTERPOLARE I DATI SULLA FORESTA IN MODO DA OTTENERE GLI ETTARI DI FORESTA DAL 1960 AL 1990 (DATI
#CHE LA FAO NON FORNISCE

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#IMPORTO IL CSV
nomi_colonne = ["Years","Forestland"]
dati = pd.read_csv('C:/Users/annam/Desktop/climate project/DatiForestGump.csv', names=nomi_colonne)

X = []
y = []
for i in dati[nomi_colonne[0]].values:
    X.append([i])

for i in dati[nomi_colonne[1]].values:
    y.append(i)

#DEFINISCO IL REGRESSORE
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

#FACCIO IL FITTING
X_poly1 =[]
X1 = []
for i in range(1960, 1990, 1):
    X1.append([i])
for i in X1:
    poly_reg1 = PolynomialFeatures(degree=2)
    X_poly1.append(poly_reg.fit_transform([i])[0])
pol_reg1 = LinearRegression()
pol_reg1.fit(X_poly, y)


#UNISCO I DATI CON LE PREDIZIONI IN UN SOLO ARRAY E CREO IL NUOVO CSV
time = np.concatenate((X1, X))
forest = np.concatenate((pol_reg.predict(poly_reg.fit_transform(X1)), y))
#print(f'questi sono gli anni {time} e questa la predizione {forest}')
#plt.plot(time, forest, color='red')
#plt.show()

time1=[]
for i in range(len(time)):
    time1.append(time[i][0])
print(time1)


d = {'years': time1, 'forest [he]': forest}
#print(d)
df = pd.DataFrame(data=d)
#print(df)
df.to_csv('Forest_prediction.csv', index=False)

#FACCIO UN GRAFICO IN CUI IN VERDE VISUALIZZO I DATI CHE HO E IN BLU VISUALIZZO I DATI CHE MI FORNISCE LA REGRESSIONE
plt.scatter(X, y, color='green', alpha=0.5, label='real values')
plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue', alpha=0.5, label='predicted values')
plt.plot(X1, pol_reg.predict(poly_reg.fit_transform(X1)), color='blue', alpha=0.5)
plt.title('Forest prevision 1960/1990 with polinomial regression')
plt.xlabel('Years')
plt.ylabel('Hectars of forest')
leg = plt.legend()
plt.show()

