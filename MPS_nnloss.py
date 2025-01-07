import numpy as np
import pandas as pd
import datetime

from MPS_functions import *
from itertools import product
import tensorflow as tf

T = 3000
np.random.seed(2024)
file_path = 'Loss_nn.csv'
y = DGP_NN(T)
x_train, y_train = create_sequences(y.reshape((-1, 1)), 3)

column_names = ['nn'+str(L0)+str(L1)+str(L2) for (L0, L1, L2) in product([3,4], [3,4], [3,4])] + \
                ['nn'+str(L0)+str(L1) for (L0, L1) in product([3,4], [3,4])]
Loss_matrix = pd.DataFrame(columns=column_names)
Loss_matrix.to_csv(file_path, index=False)
for t in range(500, y_train.shape[0]):
    errors = []
    for L1 in range(5, 7):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(L1, activation='relu', input_dim=x_train.shape[1]),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train[:t, :], y_train[:t, ], epochs=3)
        errors += [([model.predict(x_train[[t-1], :])[0][0]] - y_train[t])**2]

    new_row = pd.DataFrame({ k:v for (k,v) in zip(column_names, errors)})
    new_row.to_csv(file_path, mode='a', index=False, header=False)
