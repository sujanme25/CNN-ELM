#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def picp(actual, upper_b , lower_b):
    picp = 0
    for (a, u, l) in zip(actual, upper_b , lower_b): 
        if l < a < u:
            picp += 1 
    return 100*picp/len(actual)

def mpiw( upper_b , lower_b ):
    return np.sum(np.absolute(upper_b.T - lower_b.T))/len(upper_b)

def WS(y_pred_upper_LSTM, y_pred_lower_LSTM, y_actual_LSTM, confidence):
  # Find out of bound indices for WS
    idx_oobl = np.where((y_actual_LSTM < y_pred_lower_LSTM) > 0)
    idx_oobu = np.where((y_actual_LSTM > y_pred_upper_LSTM) > 0)

    PICP = np.sum((y_actual_LSTM > y_pred_lower_LSTM) & (y_actual_LSTM <= y_pred_upper_LSTM)) / len(y_actual_LSTM) * 100
    WS = np.sum(np.sum(y_pred_upper_LSTM - y_pred_lower_LSTM) + 
              np.sum(2 * (y_pred_lower_LSTM[idx_oobl[0]] - y_actual_LSTM[idx_oobl[0]]) / confidence) +
              np.sum(2 * (y_actual_LSTM[idx_oobu[0]] - y_pred_upper_LSTM[idx_oobu[0]]) / confidence)) / len(y_actual_LSTM)
    return WS
def ARIL(actual, upper_b , lower_b):
    A1=np.sum(np.absolute(upper_b.T - lower_b.T)/actual)
    return A1/len(upper_b)
def PINAW(act_range, upper_b , lower_b):
    P1=np.sum(np.absolute(upper_b.T - lower_b.T)/act_range)
    return P1/len(upper_b)
def F(P,M):
    P1=2*(P/M)
    P2=P+(1/M)
    P3=P1/P2
    return P3

# In[ ]:




