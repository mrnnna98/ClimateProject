import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

nomi_colonne = ["Years","Forestland"]
dati = pd.read_csv('C:/Users/annam/Desktop/climate project/DatiForestGump.csv', names=nomi_colonne)

X = []
y = []
for i in dati[nomi_colonne[0]].values:
    X.append([i])

for i in dati[nomi_colonne[1]].values:
    y.append(i)

print(X)
print(y)

reg = LinearRegression().fit(X, y)
reg.score(X, y)

time = []
predict = []
for i in range (1990, 2019, 1):
    predict.append(reg.predict([[i]])[0])
    time.append(i)


#print(reg.coef_)
#print(reg.intercept_)

plt.plot(time, predict ,'-', label= 'predizione')
plt.plot(time, y ,'b-', label='real')
plt.legend(loc='best')
plt.show()