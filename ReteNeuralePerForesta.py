#RETE NEURALE DI REGRESSIONE SULLA FORESTA

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Don't cheat - fit only on training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # apply same transformation to test data
# X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def traina_Regressori_multipli(X1_train, y1_train):
    regressori = []
    max_iterList = [500, 1000, 2000]
    for i in max_iterList:
        regressori.append(MLPRegressor(random_state=1, max_iter=i).fit(X1_train, y1_train))

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(len(regressori)):
        fig, ax = plt.subplots()
        ax.plot(range(0, regressori[i].n_iter_, 1), regressori[i].loss_curve_)

        ax.set(xlabel='Iteration', ylabel='Loss', title='Max Iter {}'.format(max_iterList[i]))
        ax.grid()

        # fig.savefig("test.png")
        plt.show()

#CARICA I DATI, TU DOVRAI CAMBIARE LA RIGA SUCCESSIVA PER LEGGERE I TUOI X E Y.
#X, y = make_regression(n_samples=200, random_state=1)

#print('printo x e y {} ,{}'.format(X, y))


nomi_colonne1 = ["Years","Forestland"]
dati = pd.read_csv('C:/Users/annam/Desktop/climate project/DatiForestGump.csv', names=nomi_colonne1)
nuovoPorcellino1 = pd.DataFrame(columns=nomi_colonne1)


X1 = []
y1 = []
for i in dati[nomi_colonne1[0]].values:
    X1.append(i)

for i in dati[nomi_colonne1[1]].values:
    y1.append(i)


print("Dataset Completo X1:{}".format(X1))
print("Dataset Completo Y1:{}".format(y1))

#print('lunghezza di x1 e y1')
#print(len(X1[0]), len(y1[0]))

#SPLITTI IN TRAIN TEST.
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=1)
print(X1_train, X1_test, y1_train, y1_test)

import numpy as np
X1_train = np.asarray(X1_train).reshape(-1,1)
y1_train = np.asarray(y1_train).reshape(-1,1)
print(X1_train, X1_test, y1_train, y1_test)

#deifinisco il numero di allenamenti e il regressore

traina_Regressori_multipli(X1_train, y1_train)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X1_train, y1_train)

X1_test = np.asarray(X1_test).reshape(-1, 1)
print(X1_test, y1_test, regr.predict(X1_test[:]))


#cambia da list a array di numpy per fare la previsione e fa la previsione
predizioni = []
for i in range(1990, 2019, 1):
    #print("La Predizione è {}".format(regr.predict(np.asarray([i]).reshape(-1, 1))))
    predizioni.append(regr.predict(np.asarray([i]).reshape(-1, 1)))
print('queste sono le predizioni{}'.format(predizioni))

#cambia da array di numpy a list per fare il grafico
predizione = []
for i in predizioni:
    predizione.append(i.tolist()[0])

print(f'List: {predizione}')

plt.plot(X1, predizione, 'b-', label='predicted values')
plt.plot(X1, y1, '-', label='real values')
plt.legend(loc='best')
plt.show()


#print("Lo score del regressore è: {}".format(regr.score(X1_test, y1_test)))

#print("Coefficienti della rete neurale: {}".format(regr.coefs_))


#import matplotlib.pyplot as plt
#import numpy as np

#fig, ax = plt.subplots()
#ax.plot(range(0,regr.n_iter_,1), regr.loss_curve_)

#ax.set(xlabel='Iteration', ylabel='Loss',title='regr.loss_curve_')
#ax.grid()

#fig.savefig("test.png")
#plt.show()