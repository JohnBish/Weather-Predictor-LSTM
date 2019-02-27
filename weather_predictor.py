from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import tensorflowjs as tfjs
import numpy as np
import pandas as pd
from matplotlib import pyplot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def one_hot_cols(df, cols):
    for each in cols:
        df[each] = pd.Categorical(df[each])
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df.drop(each, axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1)
    return df

def fit_lstm(X, y, batch_size, nb_epoch, neurons):
    history = []
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(BatchNormalization(batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
    model.add(LSTM(neurons, stateful=True, dropout=0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    plot_model(model, to_file='model.png')
    for _ in range(nb_epoch):
        history.append(model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False))
        model.reset_states()
    model.save('model.h5')
    tfjs.converters.save_keras_model(model, './')
    return history

# import dataset
series = pd.read_csv('./weatherHistory.csv', header=0, index_col=0, squeeze=True)
df = pd.DataFrame(series)
ds = series.values

# one hot encode feature strings
df = one_hot_cols(df, ['Daily Summary', 'Precip Type'])

# match up each prediction with the next
columns = [df.shift(i) for i in range(1, 2)]
labeled = pd.concat(columns, axis=1)
labeled.fillna(0, inplace=True)
#print(df.head())

# remove Summary for labels and one hot encode
labels = labeled['Summary']
df.drop('Summary', axis=1, inplace=True)
labels = one_hot_cols(pd.DataFrame(labels), ['Summary'])
labels.drop('Summary_0', axis=1, inplace=True)

X = df.values
Y = labels.values
# print(X.head())

histories = fit_lstm(X, Y, 7, 5, 22)
history = np.array([])
for h in histories:
    history = np.append(history, h.history)
print(history)