# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:43:25 2022

# -*- coding: utf-8 -*-

@author: Urban Burnik
data acquisition reassembled: April 29th 2022
get Exposure values from File 1
get 
"""
#%% start  cell

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
#import scipy.stats as scs
#import pwlf  # install by: conda install -c conda-forge pwlf

#import explVar_tools as evt

# conda install -c conda-forge neurokit2

# https://towardsdatascience.com/piecewise-linear-regression-model-what-is-it-and-when-can-we-use-it-93286cfee452
# http://web.pdx.edu/~newsomj/pa551/lectur18.htm

# Load data from low level video features:

# load video features and create dict for all videos

#%% import video features

file_path = '../samples/'
data_fn = 'LLFeatures.csv'
data_df = pd.read_csv(file_path + data_fn)
data_df.rename(columns = {'videoName':'videoID'}, inplace = True)
vFeatures_df=data_df.groupby('videoID')[
    'avgShotLenAd',
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgLightKeyAd'
    ].mean()

vFeatures_di=vFeatures_df.to_dict();
# from here data_df can be released and reused.
# features are safely stored in a dictionary.

#%% import OBrian scores and get averages over each video

data_fn = 'OBrienUES.csv'
data_df = pd.read_csv(file_path + data_fn)

# Rename column

data_df.rename(columns = {'adID':'videoID'}, inplace = True)

# addLLFeature values to each video related UES score 

data_df['avgShotLenAd']=data_df['videoID'].map(vFeatures_di['avgShotLenAd'])
data_df['avgColVarAd']=data_df['videoID'].map(vFeatures_di['avgColVarAd'])
data_df['avgMotionMeanAd']=data_df['videoID'].map(vFeatures_di['avgMotionMeanAd'])
data_df['avgMotionStdAd']=data_df['videoID'].map(vFeatures_di['avgMotionStdAd'])
data_df['avgLightKeyAd']=data_df['videoID'].map(vFeatures_di['avgLightKeyAd'])

# calculate average scores per each video
oBrienUEScores_df=data_df.groupby('videoID')[
    'FA',    
    'PU', 
    'AE', 
    'RW'
    ].mean()

oBrienUEScores_di=oBrienUEScores_df.to_dict();

data_df['avgFA']=data_df['videoID'].map(oBrienUEScores_di['FA'])
data_df['avgPU']=data_df['videoID'].map(oBrienUEScores_di['PU'])
data_df['avgAE']=data_df['videoID'].map(oBrienUEScores_di['AE'])
data_df['avgRW']=data_df['videoID'].map(oBrienUEScores_di['RW'])

#%% here mapping of video id's is provided from 2 different data sources
# file_path = '../samples/'
# dict_csv = 'mapVideo2Group.csv'

# videoID_di = pd.read_csv(file_path + dict_csv, header=None, index_col=0, squeeze=True).to_dict()
# videoID_di = {'C1': 'V1',
#               'C4': 'V2', 
#               'C5': 'V3', 
#               'C6': 'V4',
#               'C7': 'V5',
#               'C8': 'V6'}

# data_df['videoIDc']=data_df['videoID']
# data_df['videoID']=data_df['videoID'].map(videoID_di)

#%% reassemble teble and write the results to table

#data_df['Gender']=data_df['userID'].map(uGender_di)
data_df= data_df[['userID', 
                  'FA', 'PU', 'AE', 'RW',
                  'avgFA', 'avgPU', 'avgAE', 'avgRW',
                  'videoID',
                  'avgShotLenAd', 'avgColVarAd', 'avgMotionMeanAd', 'avgMotionStdAd', 'avgLightKeyAd'
        ]]

UES_LLF_fn = 'UES_LLFeatures.csv'
data_df.to_csv(file_path + UES_LLF_fn, index=False)

