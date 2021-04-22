from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

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
test = [10, 50, 100, 200, 500, 1000]
for k in test:
    clf = RandomForestRegressor(n_estimators=k)
    clf = clf.fit(X, y)
    print(clf)

    time = []
    predict = []
    for i in range (1960, 1990, 1):
        predict.append(clf.predict([[i]])[0])
        time.append(i)
    #for i in range(1990,2019,1):
    #    predict.append(clf.predict([[i]])[0])
    #    time.append(i)
    for i in range(2020,2050,1):
        predict.append(clf.predict([[i]])[0])
        time.append(i)


    plt.plot(time,predict,'-', label= 'predizione')
    #plt.plot(time,y,'b-', label='real')
    plt.legend(loc='best')
    plt.show()