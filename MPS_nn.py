import numpy as np
import pandas as pd
import datetime

from itertools import product
import tensorflow as tf

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from MPS_functions import *

Loss_nn = pd.read_csv('Loss_nn.csv')
T = 3000
Tr = robjects.FloatVector([T])
robjects.r.assign('Tr', Tr)
np.random.seed(2024)
y = DGP_NN(T)
x_train, y_train = create_sequences(y.reshape((-1, 1)), 3)
yr = robjects.FloatVector(y)
robjects.r.assign('yr', yr)
robjects.r('''
models <- c('AR1', 'AR2', 'AR3', 'AR4', 'AR5', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5')
loss_matrix <- data.frame(matrix(ncol = length(models), nrow = 0))
colnames(loss_matrix) <- models

for (t in 500:(Tr-1)){
  AR1 <- arima(yr[1:t], order = c(1,0,0), include.mean=FALSE)
  AR1_p <- as.numeric(predict(AR1, se.fit = FALSE))

  AR2 <- arima(yr[1:t], order = c(2,0,0), include.mean=FALSE)
  AR2_p <- as.numeric(predict(AR2, se.fit = FALSE))

  AR3 <- arima(yr[1:t], order = c(3,0,0), include.mean=FALSE)
  AR3_p <- as.numeric(predict(AR3, se.fit = FALSE))

  AR4 <- arima(yr[1:t], order = c(4,0,0), include.mean=FALSE)
  AR4_p <- as.numeric(predict(AR4, se.fit = FALSE))

  AR5 <- arima(yr[1:t], order = c(5,0,0), include.mean=FALSE)
  AR5_p <- as.numeric(predict(AR5, se.fit = FALSE))

  MA1 <- arima(yr[1:t], order = c(0,0,1), include.mean=FALSE)
  MA1_p <- as.numeric(predict(MA1, se.fit = FALSE))

  MA2 <- arima(yr[1:t], order = c(0,0,2), include.mean=FALSE)
  MA2_p <- as.numeric(predict(MA2, se.fit = FALSE))

  MA3 <- arima(yr[1:t], order = c(0,0,3), include.mean=FALSE)
  MA3_p <- as.numeric(predict(MA3, se.fit = FALSE))

  MA4 <- arima(yr[1:t], order = c(0,0,4), include.mean=FALSE)
  MA4_p <- as.numeric(predict(MA4, se.fit = FALSE))

  MA5 <- arima(yr[1:t], order = c(0,0,5), include.mean=FALSE)
  MA5_p <- as.numeric(predict(MA5, se.fit = FALSE))

  newrow <- data.frame(AR1 = (AR1_p-yr[t+1])^2,
  AR2 = (AR2_p-yr[t+1])^2,
  AR3 = (AR3_p-yr[t+1])^2,
  AR4 = (AR4_p-yr[t+1])^2,
  AR5 = (AR5_p-yr[t+1])^2,
  MA1 = (MA1_p-yr[t+1])^2,
  MA2 = (MA2_p-yr[t+1])^2,
  MA3 = (MA3_p-yr[t+1])^2,
  MA4 = (MA4_p-yr[t+1])^2,
  MA5 = (MA5_p-yr[t+1])^2
  )
  loss_matrix = rbind(loss_matrix, newrow)

}

''')
Loss_linear = pandas2ri.rpy2py(robjects.r['loss_matrix'])
Loss_linear.reset_index(drop=True, inplace=True)
Loss_matrix = pd.concat([Loss_linear.iloc[:Loss_nn.shape[0], :], Loss_nn], axis=1)

MPS_matrix = MPS(Loss_matrix, 500, 0.2, 0.1,
                 1000.0, beta_search=20, alpha_search=20, bootstrap=100, lable='NN')

