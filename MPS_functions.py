import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import ortho_group
import copy
from itertools import product
import datetime
from ast import literal_eval
from scipy import linalg
from scipy.stats import ortho_group
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import plotly.express as px
import plotly
import datetime
import random
from tensorflow.keras import layers, Model
import tensorflow as tf
np.set_printoptions(precision=8, suppress=True)


#import os 
#os.environ['R_HOME'] = '/gpfs/sharedfs1/admin/hpc2.0/apps/r/4.2.2/lib64/R'
#os.environ['LD_LIBRARY_PATH'] = '/gpfs/sharedfs1/admin/hpc2.0/apps/r/4.2.2/lib64/R/library' 

# 
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def DGP_STAR(gamma4, gamma5, gamma6, T, burn=200):
    y = np.zeros(T + burn)
    for t in range(T+burn-1):
        noise = np.random.normal(loc=0.0, scale=1)
        y[t+1] = (gamma4-gamma5*np.exp(-gamma6*y[t-1]))/(1+np.exp(-gamma6*y[t-1]))*y[t]+noise
        
    return y[burn:]

def DGP_TAR(gamma1, gamma2, gamma3, T, burn=200):
    y = np.zeros(T + burn)
    for t in range(T+burn-1):
        noise = np.random.normal(loc=0.0, scale=1)
        y[t+1] = (gamma1+gamma2*(y[t-1]<gamma3))*y[t]+noise
        
    return y[burn:]

def DGP_AR(alpha, T, burn = 200, scale=1, pre_y=np.arange(0)):
    match len(pre_y):
        case 0:
            y = np.zeros(T + burn)
            for t in range(T+burn-1):
                noise = np.random.normal(loc=0.0, scale=scale)
                y[t+1] = alpha*y[t]+noise
            return y[burn:]
        case _:
            y = np.append(pre_y, np.zeros(T))
            for t in range(len(pre_y), len(pre_y)+T):
                noise = np.random.normal(loc=0.0, scale=scale)
                y[t] = alpha*y[t-1]+noise
                noise_pt = noise
            return y 
        
    return y[burn:]

def DGP_ARMA(alpha, beta, T, burn = 200, scale=1, pre_y=np.arange(0)):
    match len(pre_y):
        case 0:
            y = np.zeros(T + burn)
            noise_pt = 0
            for t in range(T+burn-1):
                noise = np.random.normal(loc=0.0, scale=scale)
                y[t+1] = alpha*y[t]+noise+beta*noise_pt
                noise_pt = noise
            return y[burn:]
        case _:
            y = np.append(pre_y, np.zeros(T))
            noise_pt = 0
            for t in range(len(pre_y), len(pre_y)+T):
                noise = np.random.normal(loc=0.0, scale=scale)
                y[t] = alpha*y[t-1]+noise+beta*noise_pt
                noise_pt = noise
            return y

def DGP_MSA(T, trans_matrix, burn=200, scale=1):
    states = [0]
    y = np.zeros(T+burn)
    for t in range(T+burn):
        states = random.choices([0,1], weights = trans_matrix[:, states[0]])
        match states[0]:
            case 0:
                y[t] = 0.3*y[t-1]+np.random.normal(loc=0.0, scale=scale)
            case 1:
                y[t] = -0.3*y[t-1]+np.random.normal(loc=0.0, scale=scale)
                
    return y[burn:]

def DGP_NN(T, L0_len=3, L1_len=10, burn = 200, scale = 1):
    y = np.zeros(T+burn)
    L0_matrix = np.random.rand(L0_len,L1_len)*2-1
    L1_matrix = np.random.rand(L1_len)*2-1
    
    for t in range(T+burn-L0_len):
        L0 = y[t:t+L0_len]@L0_matrix
        L0 = np.multiply(L0, L0>0) + np.random.normal(loc=0.0, scale=scale, size=L1_len)
        y[t+L0_len] = 1/(1+np.exp(-L1_matrix@L0))+np.random.normal(loc=0.0, scale=scale)
    
    return y[burn:]

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def build_decoder_only_transformer(input_shape, num_heads=8, ff_dim=16, num_layers=2):
    inputs = layers.Input(shape=input_shape)

    # Positional Encoding
    positional_encoding = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[0])(inputs)

    # Decoder-only transformer block
    x = positional_encoding
    for _ in range(num_layers):
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
        x = layers.Add()([x, attention_output])  # Add & Normalize (Skip connection)
        x = layers.LayerNormalization()(x)

        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(input_shape[-1])(ff_output)
        x = layers.Add()([x, ff_output])  # Add & Normalize (Skip connection)
        x = layers.LayerNormalization()(x)

    # Output layer
    outputs = layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def forecast(model, input_seq, n_steps):
    """
    Generate forecasted values for the next n_steps given the input sequence.

    Args:
    - model: The trained model.
    - input_seq: The input sequence to predict from.
    - n_steps: The number of future steps to forecast.

    Returns:
    - forecast: The forecasted values for the next n_steps.
    """
    forecast = []
    current_input = input_seq

    for _ in range(n_steps):
        pred = model.predict(current_input[np.newaxis, :])  # Add batch dimension
        forecast.append(pred[0, 0])  # Extract forecasted value
        current_input = np.roll(current_input, -1, axis=0)  # Shift the input sequence
        current_input[-1] = pred[0, 0]  # Append the predicted value

    return np.array(forecast)


def MPS(Loss, start, alpha_bar, gamma, lambda_max, 
        beta_search=10, alpha_search = 10, T=1, B=100, bootstrap=100, lable=''):
    MPS_all = pd.DataFrame(columns = ['t', 'MPS', 'uncal_MPS', 'mis_cover_rate', 'uncal_cover_rate',
                                      'alpha_t', 'lambda_t', 'beta_t'])
    file_path = 'MPS_logs_'+datetime.datetime.today().strftime('%Y_%m_%d_') + lable+'.csv'
    
    MPS_all.to_csv(file_path, index=False)
    
    K = len(Loss)
    alpha_t = alpha_bar
    lambda_t = lambda_max/2
    
    alpha_barr = robjects.FloatVector([alpha_bar])
    robjects.r.assign('alpha_barr', alpha_barr)
    
    bootstrapr = robjects.IntVector([bootstrap])
    robjects.r.assign('bootstrapr', bootstrapr)
    
    pandas2ri.activate()
    MCS = importr('MCS')
    #yr = pandas2ri.py2rpy(y)
    #robjects.r.assign('yr', yr)
    Loss = pandas2ri.py2rpy(Loss)
    robjects.r.assign('Loss', Loss)
    
    # find F_t
    F_t = np.zeros(B)
    for t in range(B):
        tr = robjects.IntVector([start-t-1])
        robjects.r.assign('tr',tr)
        
        for beta_t in np.arange(0, 1, 1/beta_search):
            beta_tr = robjects.FloatVector([beta_t])
            robjects.r.assign('beta_tr',beta_tr)
            robjects.r('''
            library(MCS)
            sink("/dev/null")
            MCS_t = MCSprocedure(Loss=Loss[1:(tr-1),], alpha=beta_tr, B=bootstrapr, statistic='Tmax', cl=NULL)
            sink()
            outr <- colnames(Loss)[which.min(Loss[tr, ])] %in% MCS_t@Info$model.names
            ''')
            if not int(robjects.r('outr')[0]):
                break
                
        F_t[t] = beta_t - 1/beta_search
        
    F_t = np.flip(F_t)
    
    print(F_t)
    
    mis_cover_rate = []
    for t in range(K-start):
        print(t)
        tr = robjects.IntVector([start+t])
        robjects.r.assign('tr',tr)
        
        # update F_t
        for beta_t in np.arange(0, 1, 1/beta_search):
            beta_tr = robjects.FloatVector([beta_t])
            robjects.r.assign('beta_tr',beta_tr)
            robjects.r('''
            library(MCS)
            sink("/dev/null")
            MCS_t = MCSprocedure(Loss=Loss[1:(tr-1),], alpha=beta_tr, B=bootstrapr, statistic='Tmax', cl=NULL)
            outr <- colnames(Loss)[which.min(Loss[tr, ])] %in% MCS_t@Info$model.names
            sink()
            ''')
            if not int(robjects.r('outr')[0]):
                break
                
        F_t = np.append(F_t[1:], beta_t - 1/beta_search)
        
        # update alpha_t
        alpha_t_loss = np.zeros(alpha_search)
        for alpha_index in range(alpha_search):
            alpha_try = (alpha_index+1)/alpha_search
            alpha_tryr = robjects.FloatVector([alpha_try])
            robjects.r.assign('alpha_tryr',alpha_tryr)
            robjects.r('''
            library(MCS)
            sink("/dev/null")
            MCS_t = MCSprocedure(Loss=Loss[1:tr,], alpha=alpha_tryr, B=bootstrapr, statistic='Tmax', cl=NULL)
            sink()
            Lossr <- length(MCS_t@Info$model.names)
            ''')
            
            alpha_t_loss[alpha_index] = (int(robjects.r('Lossr')[0]) + 
                                         lambda_t*(1-alpha_bar)*(F_t < alpha_try).mean())
            
        alpha_t = ((np.argmin(alpha_t_loss)+1)/alpha_search)*(lambda_t<lambda_max)
        
        # get MPS
        alpha_tr = robjects.FloatVector([alpha_t])
        robjects.r.assign('alpha_tr',alpha_tr)
        robjects.r('''
        library(MCS)
        print(c(alpha_tr, alpha_barr))
        sink("/dev/null")
        MCS_t = MCSprocedure(Loss=Loss[1:tr,], alpha=alpha_tr, B=bootstrapr, statistic='Tmax', cl=NULL)
        uncal_MPSt = MCSprocedure(Loss=Loss[1:tr,], alpha=alpha_barr, B=bootstrapr, statistic='Tmax', cl=NULL)
        sink()
        MCSr <- MCS_t@Info$model.names
        uncal_MPSr <- uncal_MPSt@Info$model.names
        lambda_outr <- !colnames(Loss)[which.min(Loss[tr+1, ])] %in% MCSr
        uncal_outr <- !colnames(Loss)[which.min(Loss[tr+1, ])] %in% uncal_MPSr
        ''')
        MPS = robjects.r('MCSr')
        uncal_MPS = robjects.r('uncal_MPSr')
        
        # update lambda_t
        lambda_t = lambda_t - gamma*lambda_max*(alpha_bar - int(robjects.r('lambda_outr')[0]))
        
        print(MPS)
        MPS_new_row = pd.DataFrame({'t':t,
                                    'MPS':[MPS],
                                    'uncal_MPS':[uncal_MPS],
                                    'mis_cover_rate': int(robjects.r('lambda_outr')[0]), 
                                    'uncal_cover_rate': int(robjects.r('uncal_outr')[0]), 
                                    'alpha_t':alpha_t,
                                    'lambda_t':lambda_t,
                                    'beta_t':beta_t})
        MPS_new_row.to_csv(file_path, mode='a', index=False, header=False)
        # MPS_all = pd.concat([MPS_all, MPS_new_row])
        
    return MPS_all
    
        
    