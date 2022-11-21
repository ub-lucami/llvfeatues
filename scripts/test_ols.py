# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:10:15 2022

@author: urbanb
"""
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as smf
#import statsmodels.api as smf

import matplotlib.pyplot as plt
import pydove as dv
import numpy as np
import pandas as pd

#%%
# Load data from low level video features file:

# Video feature
#file_path = '../samples/'
file_path = '../../pydove/CW30x5N/'
data_fn = 'UES_LLFeatures.csv'


data_df = pd.read_csv(file_path + data_fn)
data_df_fnorm = pd.DataFrame()

#%%
# provide UES data column
data_df['UES']=data_df[['FA', 'PU', 'AE', 'RW']].mean(axis=1)
#normalize data_df
fnames_norm=[
    'avgLightKeyAd',    
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    ]
data_df_fnorm = data_df.copy()
#data_df_fnorm[fnames_norm]=(data_df_fnorm[fnames_norm]-data_df_fnorm[fnames_norm].min())/(data_df_fnorm[fnames_norm].max()-data_df_fnorm[fnames_norm].min())
data_df_fnorm[fnames_norm]=(data_df_fnorm[fnames_norm]/data_df_fnorm[fnames_norm].max())


#%%
# Perform OLSs

results = smf.ols('FA ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df).fit()
#data_df['pFA']=results.predict(data_df)
results.summary()

results = smf.ols('PU ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df).fit()
#data_df['pPU']=results.predict(data_df)
results.summary()

results = smf.ols('AE ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df).fit()
#data_df['pAE']=results.predict(data_df)
results.summary()

results = smf.ols('RW ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df).fit()
#data_df['pRW']=results.predict(data_df)
results.summary()

results = smf.ols('UES ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df).fit()
#data_df['pUES']=results.predict(data_df)
results.summary()

results = smf.ols('FA ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df_fnorm).fit()
#data_df['pFA']=results.predict(data_df)
results.summary()

results = smf.ols('PU ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df_fnorm).fit()
#data_df['pPU']=results.predict(data_df)
results.summary()

results = smf.ols('AE ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df_fnorm).fit()
#data_df['pAE']=results.predict(data_df)
results.summary()

results = smf.ols('RW ~ avgLightKeyAd + I(avgLightKeyAd**2) + avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df_fnorm).fit()
#data_df['pRW']=results.predict(data_df)
results.summary()

results = smf.ols('UES ~ avgLightKeyAd  + I(avgLightKeyAd**2)+ avgColVarAd + I(avgColVarAd**2) + avgMotionMeanAd + I(avgMotionMeanAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) + avgShotLenAd + I(avgShotLenAd**2) ', data=data_df_fnorm).fit()
#data_df['pRW']=results.predict(data_df)
results.summary()

results = smf.ols('UES ~   avgColVarAd + I(avgColVarAd**2) + avgMotionStdAd + I(avgMotionStdAd**2) ', data=data_df_fnorm).fit()
#data_df['pUES']=results.predict(data_df)
results.summary()
