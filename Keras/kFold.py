from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop
import numpy as np
import pandas as pd

# Read data
white = pd.read_csv("winequality-white.csv", sep=';')
red = pd.read_csv("winequality-red.csv", sep=';')

white['type'] = 0
red['type'] = 1

wines = red.append(white, ignore_index=True).sample(frac=1)
Y = np.ravel(wines.quality)
X = wines.drop(['quality'], axis=1)

# Create model
model = Sequential()

model.add(Dense(64, activation='relu', input_dim=12))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

k = 5
l = int(len(X) / k)
mse_total, mae_total = 0, 0
for i in range(k):
    train_x = X[i*l:(i+1)*l]
    train_y = Y[i*l:(i+1)*l]

    test_x = np.concatenate([X[:i*l], X[(i+1)*l:]]);
    test_y = np.concatenate([Y[:i*l], Y[(i+1)*l:]]);

    model.fit(train_x, train_y, epochs=15)

    predictions = model.predict(test_x)
    mse, mae = model.evaluate(test_x, test_y)
    mse_total += mse
    mae_total += mae

mse_avg = mse_total / k
mae_avg = mae_total / k
print(mse_avg, mae_avg)
