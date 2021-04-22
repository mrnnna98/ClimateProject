#Gentilmente offerto dal moroso di Anna 8)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


def parse_datasetMPG(dataset):
    dataset.isna().sum()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    dataset.tail()

    #Splittalo in train and test, 80% e 20%
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return dataset, train_dataset, test_dataset


def parse_datasetANNA(dataset):
    dataset.isna().sum()
    #Controllare di cancellare i NAN giusti e non tutto il dataset
    #dataset = dataset.dropna()
    print(dataset.tail())

    #Splittalo in train and test, 80% e 20%
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    print(train_dataset.tail())
    test_dataset = dataset.drop(train_dataset.index)
    return dataset, train_dataset, test_dataset


def plot_datasetMPG(train_dataset):
    sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    plt.show()
    train_dataset.describe().transpose()
    return train_dataset

def plot_datasetANNA(train_dataset):
    print(train_dataset.tail())
    #sns.pairplot(train_dataset[['Years','Population','GDP','Agriculture','Forestland']], x_vars=['Years'],
    #y_vars=['Population','GDP','Agriculture','Forestland'], kind = 'scatter', diag_kind='auto')
    #plt.show()
    #train_dataset.describe().transpose()
    #return train_dataset

def splitFeaturesFromLabels(train_dataset, test_dataset):
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    return train_labels, test_labels, train_features, test_features

def splitFeaturesFromLabelsANNA(train_dataset, test_dataset):
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Forestland')
    test_labels = test_features.pop('Forestland')
    return train_labels, test_labels, train_features, test_features

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()






if __name__ == '__main__':
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    nomi_colone = ["Years","Population","GDP","Agriculture","Forestland"]
    raw_dataset = pd.read_csv("C:/Users/annam/PycharmProjects/RegressionCoLab/FinalCsvForNeuralNetwork.csv", header=0)

    print(raw_dataset.tail())

    #pd.read_csv(url, names=column_names,
                  #            na_values='?', comment='\t',
                  #            sep=' ', skipinitialspace=True)
    dataset = raw_dataset.copy()


    dataset, train_dataset, test_dataset = parse_datasetANNA(dataset)

    
    #train_dataset = plot_datasetANNA(train_dataset)
    
    train_labels, test_labels, train_features, test_features = splitFeaturesFromLabelsANNA(train_dataset, test_dataset)
    print('printo train')
    print(train_labels, test_labels.tail(), train_features.tail(), test_features.tail())

    horsepower = np.array(train_features['Horsepower'])

    horsepower_normalizer = preprocessing.Normalization(input_shape=[1, ])
    horsepower_normalizer.adapt(horsepower)

    horsepower_model = tf.keras.Sequential([
        horsepower_normalizer,
        layers.Dense(units=1)
    ])

    #print(horsepower_model.summary())
    print(horsepower_model.predict(horsepower[:10]))

    horsepower_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    print(train_features['Horsepower'])

    #%%time
    history = horsepower_model.fit(
        train_features['Horsepower'], train_labels,
        epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()



    plot_loss(history)

    test_results = {}

    test_results['horsepower_model'] = horsepower_model.evaluate(
        test_features['Horsepower'],
        test_labels, verbose=0)

    x = tf.linspace(0.0, 250, 251)
    y = horsepower_model.predict(x)

    plot_horsepower(x, y)
    plt.show()