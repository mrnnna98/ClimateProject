# RETE NEURALE DEL GRUPPO RICE GIRLS FT. NIK
#ENERGY AND CLIMATE CHANGE MODELING AND SCENARIOS;  PROFESSOR MASSIMO TAVONI
#POLITECNICO DI MILANO ANNO ACCADEMICO 2020/2021

#QUESTO SCRIPT SERVE AD ALLENARE LA RETE CON DIVERSI OTTIMIZZATORI INOLTRE SALVA UN GRAFICO SUL TRAINING DI OGNI
#OTTIMIZZATORE E SALVA ANCHE IL MODELLO TRAINATO

import numpy as np
from keras.layers import Dense
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

#print(tf.__version__)



def plot_loss(history):
    x_axis = np.linspace(0, epochs, num=epochs, endpoint=True)
    fig, axs = plt.subplots(2)
    fig.suptitle(f'loss and val_loss with {epochs} epochs and {optimizer} optimizer')
    axs[0].plot(x_axis, history.history['loss'],'b',alpha=0.5)
    #plt.yticks(np.arange(0, 120, step=20), [0, 20, 40, 60, 80, 100])
    axs[0].set(ylabel='loss')
    axs[1].plot(x_axis, history.history['val_loss'],'g', alpha=0.5)
    axs[1].set(xlabel='epochs', ylabel='val_loss')
    name = "{}_{}.png".format(optimizer, epochs)
    fig.savefig("./training/" + name)


def normalization(X_train):
    X_train_normalizer = preprocessing.Normalization()  # (input_shape=[1, ])
    X_train_normalizer.adapt(X_train)
    normalized_X_train = X_train_normalizer(X_train)
    return X_train_normalizer, normalized_X_train


def create_model(X_train_normalizer, opt):
    model = Sequential(X_train_normalizer)
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # COMPILE
    model.compile(loss='mean_absolute_percentage_error', optimizer=opt)
    return model

def method():

    # IMPORTO DATASET
    raw_dataset = pd.read_csv("./input_data_network.csv", header=0)
    dataset = raw_dataset.copy()

    # SETTO LA PRIMA COLONNA COME INDICE
    dataset = dataset.set_index('Years')
    # print(dataset.head())

    # DIVIDO GLI INPUT DAGLI OUTPUT
    X = dataset[["Population [mill]", "GDP[trill$]"]]
    y = dataset[["Forestland [Millha]"]]

    # DIVIDO IN TRAIN E TEST

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # print(X_test.head(), X_train.head(), y_test.head(), y_train.head())

    # NORMALIZZO GLI INPUT DA INSERIRE
    X_train_normalizer, normalized_X_train = normalization(X_train)
    # print(X_train_normalizer, normalized_X_train)

    # CALLBACK CHECKPOINT
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/{}_{}".format(optimizer, epochs),
                                                     save_weights_only=True, verbose=1)
    # CREO IL MODELLO
    GDP_and_Pop_model = create_model(X_train_normalizer, opt)
    print(GDP_and_Pop_model.summary())

    # FITTO
    # %%time
    history = GDP_and_Pop_model.fit(
        X_train, y_train,
        epochs=epochs,
        # suppress logging
        # verbose=0,
        # Calculate validation results on 20% of the training data
        callbacks=[cp_callback],
        validation_split=0.2)


    # DATAFRAME DELLA HISTORY
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    # print("key della history".format(history.history.keys()))

    # # CREO UN NUOVO MODELLO SENZA TRAINING
    # untrained_model = create_model(X_train_normalizer, opt)
    #
    # # VALUTO IL NUOVO MODELLO
    # loss = untrained_model.evaluate(X_test, y_test, verbose=1)
    #
    # # VALUTO IL MODELLO TRAINATO DOPO AVER CARICATO I WEIGHTS
    # GDP_and_Pop_model.load_weights("./checkpoints/{}_{}".format(optimizer, epochs))
    # loss = GDP_and_Pop_model.evaluate(X_test, y_test, verbose=1)

    # PLOTTO L'ERRORE
    plot_loss(history)
    #plt.show()

    # MI FACCIO UN'IDEA DI COME PREDICE
    print(y_test)
    print(GDP_and_Pop_model.predict(X_test))


if __name__ == '__main__':
    set_ottimizzatori = [[tf.keras.optimizers.Adadelta(learning_rate=0.01), 'adadelta'],
                        [tf.keras.optimizers.Adagrad(learning_rate=0.01),'adagrad'],
                        [tf.keras.optimizers.Adamax(learning_rate=0.01), 'adamax'],
                        [tf.keras.optimizers.Ftrl(learning_rate=0.01), 'ftrl'],
                        [tf.keras.optimizers.Nadam(learning_rate=0.01), 'nadam'],
                        [tf.keras.optimizers.RMSprop(learning_rate=0.01), 'rmsprop'],
                        [tf.keras.optimizers.SGD(learning_rate=0.01), 'sgd'],
                        [tf.keras.optimizers.Adam(learning_rate=0.01), 'adam']]

    set_iterazioni = [5000]

    for j in set_iterazioni:
        epochs = j
        for i in set_ottimizzatori:
            opt = i[0]
            optimizer = i[1]
            method()



