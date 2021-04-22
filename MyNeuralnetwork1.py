#RETE NEURALE PER IL PROGETTO CLIMATE E MODELLING;
# ANNA MARAN 10799994 anna.maran@mail.polimi.it

# Make numpy printouts easier to read.

import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
print(tf.__version__)


def parse_datasetANNA(dataset):
    dataset.isna().sum()
    #Controllare di cancellare i NAN giusti e non tutto il dataset
    dataset = dataset.dropna()
    dataset = dataset[(dataset[:] != 0).all(axis=1)]
    #print(dataset.tail())

    #Splittalo in train and test, 80% e 20%
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    #print(train_dataset.tail())
    test_dataset = dataset.drop(train_dataset.index)
    return dataset, train_dataset, test_dataset


def plot_datasetANNA(train_dataset):
    print(train_dataset.tail())
    #sns.pairplot(train_dataset[['Years','Population','GDP','Agriculture','Forestland']], x_vars=['Years'],
    #y_vars=['Population','GDP','Agriculture','Forestland'], kind = 'scatter', diag_kind='auto')
    #plt.show()
    #train_dataset.describe().transpose()
    #return train_dataset


def splitFeaturesFromLabelsANNA(train_dataset, test_dataset):
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Forestland')
    test_labels = test_features.pop('Forestland')
    return train_labels, test_labels, train_features, test_features


def plot_lossANNA(history):
    x_axis = np.linspace(0, 500, num=500, endpoint=True)
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x_axis, history.history['loss'], label='loss')
    axs[1].plot(x_axis, history.history['val_loss'], label='val_loss')
    plt.show()

    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # #plt.ylim([0, 10])
    # plt.xlabel('Epoch')
    # plt.ylabel('Error Forestland')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()





if __name__ == '__main__':
    raw_dataset = pd.read_csv("C:/Users/annam/PycharmProjects/RegressionCoLab/FinalCsvForNeuralNetwork.csv", header=0)

    dataset = raw_dataset.copy()

    #DROPPO LA PRIMA RIGA DEL DATASET PERCHè DENTRO C'ERA UNA STRINGA CON POP,.. STRINGA CHE NON VENIVA LETTA E DAVA ERRORI
    dataset.drop(index=dataset.index[0],
        axis=0,
        inplace=True)

    #RINOMINO LA COLONNA FORESTA (MI DAVA FASTIDIO COME ERA SCRITTA PRIMA)

    dataset.rename(columns={'forest [he]': 'Forestland'},
              inplace=True, errors='raise')

    #TOLGO AGRICOLTURA(VOLRENDO SI PUò RIMETTERE PER FARE LE POLITICHE)

    dataset = dataset.drop(columns='Agriculture')

    #PULISCO IL DATASET DA ZERI E DA NAN E LO DIVIDO IN TEST E TRAIN

    dataset, train_dataset, test_dataset = parse_datasetANNA(dataset)
    #print(dataset.eq(0).any().any())

    train_labels, test_labels, train_features, test_features = splitFeaturesFromLabelsANNA (train_dataset, test_dataset)

    #CREO GLI ARRAY CHE MI SERVONO TIRANDOLI FUORI DAL DATASET E CONVERTENDOLI
    GDP = train_features['GDP']
    GDP = np.array(GDP)
    Population = train_features['Population']
    Population = np.array(Population).astype(np.float)
    #print(GDP, Population)

    #CREO GLI INPUT DA INSERIRE MANIPOLANDO GLI ARRAY
    Input = []
    if len(Population) == len(GDP):
        for i in range(len(Population)):
            Input.append([Population[i], GDP[i]])
    #print(Input)

    #NORMALIZZO GLI INPUT DA INSERIRE
    Input_normalizer = preprocessing.Normalization()  #(input_shape=[1, ])
    Input_normalizer.adapt(Input)
    normalized_input = Input_normalizer(Input)
    #print(normalized_input)

    #DEFINISCO IL MODELLO
    GDP_and_Pop_model = tf.keras.Sequential([
        Input_normalizer,
        layers.Dense(units=1)
    ])

    #PRIMA PREVISIONE CON 10 INPUT
    #print(GDP_and_Pop_model.predict(Input[:10]))

    #MATRICI DEI PESI (chiedi a frency cosa fa questo che non capisco su internet)
    GDP_and_Pop_model.layers[1].kernel

    # #SETTO
    GDP_and_Pop_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    #TRASFORMO L'INPUT IN UN ARRAY
    Input_array = np.asarray(Input)
    #print(Input_array)

    #FITTO
    # %%time
    history = GDP_and_Pop_model.fit(
        Input_array, train_labels,
        epochs=500,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    #print("key della history".format(history.history.keys()))

    plot_lossANNA(history)

    # CREO GLI ARRAY CHE MI SERVONO PER IL TEST TIRANDOLI FUORI DAL DATASET E CONVERTENDOLI
    GDP_test = test_features['GDP']
    GDP_test = np.array(GDP_test)
    Population_test = test_features['Population']
    Population_test = np.array(Population_test).astype(np.float)
    # print(GDP_test, Population_test)

    # CREO GLI INPUT DA INSERIRE NEL TEST MANIPOLANDO GLI ARRAY
    Input_test = []
    if len(Population_test) == len(GDP_test):
        for i in range(len(Population_test)):
            Input_test.append([Population_test[i], GDP_test[i]])
    # print(Input_test)

    #TRASFORMO L'INPUT TEST IN UN ARRAY
    Input_test_array = np.asarray(Input_test)
    #print(Input_test_array)

    #SONO MOLTO PERPLESSA DA QUESTO
    test_results = {}
    test_results['GDP_and_Pop_model'] = GDP_and_Pop_model.evaluate(
        Input_test_array, test_labels, verbose=0)

    #print(f'questo è il risultato del test{test_results}')
    #print(f'questo è il test labels {test_labels}')

    #PROVO A PREDIRRE
    x = Input_test_array
    y = []
    for i in Input_test_array:
        y.append(GDP_and_Pop_model.predict(i))
    print(test_labels)
    print(f'questa è la predizione {y}')

    # plt.scatter(Input_test_array, test_labels, label='Data')
    # plt.plot(x, y, color='k', label='Predictions')
    # plt.show()
    # plt.xlabel('Horsepower')
    # plt.ylabel('MPG')
    # plt.legend()