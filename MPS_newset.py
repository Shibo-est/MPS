import numpy as np
import pandas as pd
import datetime

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from MPS_functions import *

T1 = 2000
T2 = 1000
np.random.seed(2024)
Tr = robjects.FloatVector([T1+T2])
robjects.r.assign('Tr', Tr)
#yr = DGP_ARMA(alpha=0.3, beta=0.3, T=T1, scale=0.1)
#yr = DGP_AR(alpha=0.3, T=T2, scale=0.1, pre_y=yr)
yr = DGP_TAR(0.6,-0.8, 1.0, T1+T2)
#yr = DGP_STAR(0.4,-0.1, 1.0, T1+T2)
#yr = DGP_MSA(T1+T2, np.ones((2,2)))
yr = robjects.FloatVector(yr)
robjects.r.assign('yr', yr)
robjects.r('''
library(TSA)
models <- c('AR1', 'AR2', 'AR3', 'MA1', 'MA2', 'MA3', 'TARd211', 'TARd212', 'TARd221', 'TARd222')
loss_matrix <- data.frame(matrix(ncol = length(models), nrow = 0))
colnames(loss_matrix) <- models

for (t in 500:(Tr-1)){
  AR1 <- arima(yr[1:t], order = c(1,0,0), include.mean=FALSE)
  AR1_p <- as.numeric(predict(AR1, se.fit = FALSE))

  AR2 <- arima(yr[1:t], order = c(2,0,0), include.mean=FALSE)
  AR2_p <- as.numeric(predict(AR2, se.fit = FALSE))

  AR3 <- arima(yr[1:t], order = c(3,0,0), include.mean=FALSE)
  AR3_p <- as.numeric(predict(AR3, se.fit = FALSE))
  
  MA1 <- arima(yr[1:t], order = c(0,0,1), include.mean=FALSE)
  MA1_p <- as.numeric(predict(MA1, se.fit = FALSE))
  
  MA2 <- arima(yr[1:t], order = c(0,0,2), include.mean=FALSE)
  MA2_p <- as.numeric(predict(MA2, se.fit = FALSE))
  
  MA3 <- arima(yr[1:t], order = c(0,0,3), include.mean=FALSE)
  MA3_p <- as.numeric(predict(MA3, se.fit = FALSE))
  
  TARd211 <- tar(y=yr[1:t],p1=1,p2=1,d=2)
  TARd211_p <- predict(TARd211)$fit
  
  TARd221 <- tar(y=yr[1:t],p1=2,p2=1,d=2)
  TARd221_p <- predict(TARd221)$fit
  
  TARd212 <- tar(y=yr[1:t],p1=1,p2=2,d=2)
  TARd212_p <- predict(TARd212)$fit
  
  TARd222 <- tar(y=yr[1:t],p1=2,p2=2,d=2)
  TARd222_p <- predict(TARd222)$fit

  newrow <- data.frame(AR1 = (AR1_p-yr[t+1])^2,
  AR2 = (AR2_p-yr[t+1])^2,
  AR3 = (AR3_p-yr[t+1])^2,
  
  MA1 = (MA1_p-yr[t+1])^2,
  MA2 = (MA2_p-yr[t+1])^2,
  MA3 = (MA3_p-yr[t+1])^2,
  
  TARd211 = (TARd211_p-yr[t+1])^2,
  TARd212 = (TARd212_p-yr[t+1])^2,
  TARd221 = (TARd221_p-yr[t+1])^2,
  TARd222 = (TARd222_p-yr[t+1])^2
  )
  loss_matrix = rbind(loss_matrix, newrow)

}

''')
Loss_matrix = pandas2ri.rpy2py(robjects.r['loss_matrix'])
# print(Loss_matrix)

MPS_matrix = MPS(Loss_matrix, 500, 0.2, 0.1,
                 1000.0, beta_search=20, alpha_search=20, bootstrap=100,lable='TAR_TARcandi')

#MPS_matrix.to_csv('MPS_2024_10_16_py', index=False)
