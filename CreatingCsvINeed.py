#IMPORTO E COPIO I DUE DATAFRAME CHE DEVO UNIRE

import pandas as pd

nomi_colonne1 = ["Years","Population","GDP","Agriculture","Forestland0"]
big_dataset = pd.read_csv('C:/Users/annam/Desktop/climate project/FRENCITEST.csv', header=0)

nomi_colonne2 = ['Years','Forestland']
small_dataset = pd.read_csv('C:/Users/annam/PycharmProjects/RegressionCoLab/Forest_prediction.csv', header=0)


big_dataset1 = big_dataset.copy()
small_dataset1 = small_dataset.copy()

#MODOFICO IL DATAFRAME DOVE LE COLONNE ARRIVANO SOLO FINO AL 2018 PERCHè VOGLIO AGGIUNGERLO IN QUELLO GRANDE


lastsyears = [2019, 2020]
for i in lastsyears:
    small_dataset1 = small_dataset1.append({'years' : i,
                    'forest [he]' : '0'},
                    ignore_index=True)


#PULIVO I MIEI ERRORI (TENGO PER AVERE RIFERIMENTI)
#mod_small_dataset1 = mod_small_dataset1.drop(mod_small_dataset1.index[[60]])
#mod_small_dataset1 = mod_small_dataset1.drop(['Year'], axis=1)
#print(small_dataset1.tail())

#TIRO FUORI LA COLONNA DELLA FORESTA DAL 1960 AL 2020 E LA AGGIUNGO AL MIO DATASET BIG


final_dataset = pd.concat([big_dataset1, small_dataset1], axis=1)
final_dataset = final_dataset.drop(columns=['years', 'Forestland'])
print(final_dataset)

#CREO NUOVO CSV

final_dataset.to_csv('FinalCsvForNeuralNetwork.csv', index=False)