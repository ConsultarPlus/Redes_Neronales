import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Carga de datos
# company = 'XBT' # tiker symbol para bitcoin
company = 'ETH' # tiker symbol para etherium

inicio = dt.datetime(2016, 7, 1)
fin = dt.datetime(2021, 3, 31)

data = web.DataReader(company, 'yahoo', inicio, fin)

# Preparar los datos
# Escalar los valores de los datos para que queden todos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
dias_historicos = 90

X = []
Y = []
for x in range(dias_historicos, len(scaled_data)):
    X.append(scaled_data[x-dias_historicos:x, 0])
    Y.append(scaled_data[x, 0])
X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=20, batch_size=32)

"""Prueba de perform and accuracy"""
test_inicio = dt.datetime(2020, 1, 1)
test_fin = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_inicio, test_fin)
precio_real = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - dias_historicos:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Predicciones en Test Data
x_test = []
y_test = []

for x in range(dias_historicos, len(model_inputs)):
    x_test.append(model_inputs[x-dias_historicos:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediccion = model.predict(x_test)
prediccion = scaler.inverse_transform(prediccion)

plt.plot(precio_real, color='Black', label='Precio Real')
plt.plot(prediccion, color='Green', label='Predicción')
plt.title(f"{company} Precio de Acciones")
plt.xlabel('Tiempo (x)')
plt.ylabel(f"{company} Precio de Acciones")
plt.legend()
plt.show()

""" Predicción a futuro """
real_data = [model_inputs[len(model_inputs) + 1 - dias_historicos:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediccion = model.predict(real_data)
prediccion = scaler.inverse_transform(prediccion)
print(f"Predicción: {prediccion}")