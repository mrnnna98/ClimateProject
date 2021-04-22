#Autore: Il mitico Frenci
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
nomi_colonne = ["Years","Forestland"]
trees = pd.read_csv('C:/Users/annam/Desktop/climate project/DatiForestGump.csv', names=nomi_colonne)

#print(trees['Years'].values[1])

X = []
y = []
for i in trees[nomi_colonne[0]].values:
    X.append([i])

for i in trees[nomi_colonne[1]].values:
    y.append([i])

print("x:{}".format(X))
print("y:{}".format(y))
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
k = 0
forest = []
time = []
for i in range(1990, 2019, 1):
    time.append(i)
    forest.append(int(clf.predict([[i]])[0]))
    k += 1

forest1 = []
time1 = []
for i in range(1960,1990,1):
    time1.append(i)
    forest1.append(int(clf.predict([[i]])[0]))
print('forestwithoutprediction {}'.format(forest1))

plt.close('all')

plt.plot(time, forest, 'b-', label='predicted values with training')
plt.plot(time, y, '-', label='real values')
plt.plot(time1, forest1, 'r-', label='forest predicted without training')
plt.legend(loc='best')
plt.show()

import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig, ax = plt.subplots()
# ax.plot(trees[nomi_colonne[0]].values, trees[nomi_colonne[1]].values)
#
# ax.set(xlabel='time (years)', ylabel='Forest (??)',
#        title='Piccoli Porcellini crescono')
# ax.grid()
#
# #fig.savefig("test.png")
# plt.show()

