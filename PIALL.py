#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
import TSMETRICS
import pandas as pd
def PI(Y_test,Y_pred):
    R2= r2_score(Y_test, Y_pred)
    WI=TSMETRICS.WI(Y_test, Y_pred)
    LM=TSMETRICS.LM(Y_test, Y_pred)
    NS=TSMETRICS.NS(Y_test, Y_pred)
    KGE=TSMETRICS.KGE(Y_test, Y_pred)
    RMSE=TSMETRICS.rmse(Y_test, Y_pred)
    NRMSE=TSMETRICS.nrmse(Y_test, Y_pred)
    RRSE=TSMETRICS.rrse(Y_test, Y_pred)
    RAE=TSMETRICS.rae(Y_test, Y_pred)
    MAE=TSMETRICS.mae(Y_test, Y_pred)
    INRSE=TSMETRICS.inrse(Y_test, Y_pred)
    APB=TSMETRICS.APB(Y_test, Y_pred)
    MAPE=TSMETRICS.mape(Y_test, Y_pred)
    RRMSE=TSMETRICS.RRMSE(Y_test, Y_pred)
    RMAE=TSMETRICS.RMAE(Y_test, Y_pred)
    PIALL=[R2,WI,LM,NS,KGE,RMSE,RRMSE,MAE,RMAE,APB,MAPE]
    PIALL1= pd.DataFrame(PIALL).T
    PIALL1.columns=['R2','WI','LM','NS','KGE','RMSE','RRMSE', 'MAE','RMAE','APB','MAPE']
    dfPIALL=pd.DataFrame(PIALL1) 
    return PIALL1
    return dfPIALL

