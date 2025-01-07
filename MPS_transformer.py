import numpy as np
import pandas as pd
import datetime

from itertools import product
import tensorflow as tf

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from MPS_functions import *

T = 3000
file_path = 'Loss_transformer.csv'
np.random.seed(2024)
y = DGP_TAR(0.6,-0.8, 1.0, T)
lookback = 10
x_train, y_train = create_sequences(y.reshape((-1, 1)), lookback)

model = build_decoder_only_transformer(input_shape=(lookback, 1), num_heads=8, ff_dim=64, num_layers=2)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

column_names = ['Transformer']
Loss_matrix = pd.DataFrame(columns=column_names)
Loss_matrix.to_csv(file_path, index=False)
for t in range(500, y_train.shape[0]):
    history = model.fit(x_train[:t, :], y_train[:t, :], epochs=5, batch_size=32)
    input_seq = x_train[t-1, :]  # Take a sample from the test set
    forecasted_values = forecast(model, input_seq, n_steps=1)

    new_row = pd.DataFrame({'Transformer': [(forecasted_values[0][0][0]-y_train[t])**2]})
    new_row.to_csv(file_path, mode='a', index=False, header=False)