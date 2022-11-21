# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:58:45 2022

@author: Urban Burnik, Andrej Košir
This is towards code cleaning and produces the paper version of statistical plots:
    - video features: last row = logarithmic plots
      permorm nonlinear feature transformation based on shift and log
    - multimedia exposure

"""
#%% start  cell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import seaborn as sns
#from sklearn.metrics import r2_score
#import scipy.stats as scs
#import pwlf  # install by: conda install -c conda-forge pwlf

#import explVar_tools as evt

# conda install -c conda-forge neurokit2

# https://towardsdatascience.com/piecewise-linear-regression-model-what-is-it-and-when-can-we-use-it-93286cfee452
# http://web.pdx.edu/~newsomj/pa551/lectur18.htm

#%%
# get data
file_path = '../../pydove/CW30x5N/'
data_fn = 'UES_LLFeatures.csv'

data_df = pd.read_csv(file_path + data_fn)
data_rp = pd.DataFrame()

#%% tentative outlier drop eg. C1: single shot.
#data_df=data_df[data_df['videoID']!='C1']
data_df['UES']=data_df[['FA', 'PU', 'AE', 'RW']].mean(axis=1)

#%%
# add rows with log-lin adjustments to dataframe
# no logarithm for 'avgLightKeyAd'

for varNamesAdj in [
    'avgLightKeyAd',    
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    ]:
    data_df.insert(data_df.columns.get_loc(varNamesAdj)+1, varNamesAdj+'Adj', data_df[varNamesAdj])
    data_df[varNamesAdj+'Adj']=data_df[varNamesAdj+'Adj'].apply(lambda x: (x-data_df[varNamesAdj+'Adj'].min())/(data_df[varNamesAdj+'Adj'].max()-data_df[varNamesAdj+'Adj'].min()))
    
#log_offset=[0.001, 0.1, 0.3, 0.00003]
log_offset=[0, 0, 0, 0]

i=0
for varNamesAdj in [  
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    ]:
    data_df[varNamesAdj+'Adj']=data_df[varNamesAdj+'Adj'].apply(lambda x: (x+log_offset[i]))
    data_df[varNamesAdj+'Adj']=data_df[varNamesAdj+'Adj'].apply(np.log10)
    i=i+1


#%%
# plot feature statistics

# tentative outlier removal
data_LandTableVid=data_df[data_df['videoID']!='Cx']

# take video features from data collection only once per video
data_LandTableVid=data_LandTableVid.drop_duplicates(subset=['videoID'])
data_LandTable=data_LandTableVid[[
    'avgLightKeyAd',
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    #'videoID'
    ]]


data_LandTableLog=data_LandTableVid[[
    'avgLightKeyAdAdj',
    'avgColVarAdAdj', 
    'avgMotionMeanAdAdj', 
    'avgMotionStdAdAdj', 
    'avgShotLenAdAdj'
    #'videoID'
    ]]

varNames=data_LandTable.columns
numRows=len(varNames)
varNamesPrint=varNames
varNamesPrint=[
    'Lighting Key',
    'Color Variance', 
    'Mean Motion', 
    'StDev Motion', 
    'Shot Lenght ',
    'videoID'
    ]

# begin: prepare table of plots

fig, axes = plt.subplots(ncols=8, nrows=numRows+1, figsize=(8,3.6),
                         gridspec_kw={"width_ratios":[1.5,0.5,0.5,0.5,0.5,1,1,0], "height_ratios":[0.5,1,1,1,1,1]})
for ax in axes.flatten():
    ax.tick_params(labelbottom=0, labelleft=0, bottom=0, top=0, left=0, right=0)
    ax.ticklabel_format(useOffset=False, style="plain")
    for _,s in ax.spines.items():
        s.set_visible(False)
# border = fig.add_subplot(111)
# border.tick_params(labelbottom=0, labelleft=0, bottom=0, top=0, left=0, right=0)
# border.set_facecolor("None")

text_kw = dict(ha="center", va="center", size=8)
ax=axes[0,1]
ax.text(0.5, 0.5, 'min', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,2]
ax.text(0.5, 0.5, 'max', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,3]
ax.text(0.5, 0.5, 'mean', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,4]
ax.text(0.5, 0.5, 'std', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,5]
ax.text(0.5, 0.5, 'raw', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,6]
ax.set_facecolor([1,1,1])
ax.text(0.5, 0.5, 'hist', **text_kw)
# ax=axes[0,7]
# ax.set_facecolor([1,1,1])
# ax.text(0.5, 0.5, 'log hist', **text_kw)


for i,ax in enumerate(axes[1:,0]):
    ax.text(0.5, 0.5, varNamesPrint[i], **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,1]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].min():.3g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,2]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].max():.4g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,3]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].mean():.3g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,4]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].std():.3g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,5]):
    #ax.boxplot(x=varNames[i], data=data_LandTable, vert=False)
    #sns.boxplot(x=varNames[i], data=data_LandTable, ax=ax)
    sns.barplot(x='videoID', y=varNames[i], data=data_LandTableVid.reset_index(), color='b', ax=ax)
    ax.set(xlabel=None,  ylabel=None)
    #ax.set_yscale("log")
    delta=0.0*(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())
    ax.set(ylim=(data_LandTable[varNames[i]].min()-delta,data_LandTable[varNames[i]].max()+delta))
    ax.axes.yaxis.set_visible(False)
    #sns.histplot(data_LandTable[varNames[i]], stat="density", bins=8, kde=True, ax=ax)
    ax.set_facecolor([1,1,1])
#ax.set_xlabel(" C1 C2 C3 C4 C5 C6 C7 C8", fontsize=7.7)
for i,ax in enumerate(axes[1:,6]):
    #ax.boxplot(x=varNames[i], data=data_LandTable, vert=False)
    #sns.boxplot(x=varNames[i], data=data_LandTable, ax=ax)
    #sns.barplot(x='index', y=varNames[i], data=data_df.reset_index(), color='b', ax=ax)
    #ax.set(xlabel=None,  ylabel=None)
    #delta=0.0*(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())
    #ax.set(ylim=(data_LandTable[varNames[i]].min()-delta,data_LandTable[varNames[i]].max()+delta))
    sns.histplot(data_LandTable[varNames[i]], stat="density", bins=10, log_scale=False, kde=True, ax=ax)
    ax.set_facecolor([1,1,1])    
    ax.set(xlabel=None,  ylabel=None, xticklabels=[])
    ax.axes.xaxis.set_visible(False)
log_offset=[1, 0.015, 0.1, 0.1, 0.00001]
# for i,ax in enumerate(axes[1:,7]):
#     #ax.boxplot(x=varNames[i], data=data_LandTable, vert=False)
#     #sns.boxplot(x=varNames[i], data=data_LandTable, ax=ax)
#     #sns.barplot(x='index', y=varNames[i], data=data_df.reset_index(), color='b', ax=ax)
#     #ax.set(xlabel=None,  ylabel=None)
#     #delta=0.0*(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())
#     #ax.set(ylim=(data_LandTable[varNames[i]].min()-delta,data_LandTable[varNames[i]].max()+delta))
#     sns.histplot(data_LandTableLog[varNames[i]+'Adj'], stat="density", bins=10, log_scale=False, kde=True, ax=ax)
#     #sns.histplot(data_LandTable[varNames[i]].apply(lambda x: (x-data_LandTable[varNames[i]].min())/(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())+log_offset[i]), stat="density", bins=10, log_scale=True, kde=True, ax=ax)
#     ax.set_facecolor([1,1,1])    
#     ax.set(xlabel=None,  ylabel=None, xticklabels=[])
#     ax.axes.xaxis.set_visible(True)
fig.subplots_adjust(0.05,0.05,0.95,0.95, wspace=0.02, hspace=0.05)
plt.savefig(file_path+'VideoFeaturesStatsBarLinLogHist.pdf')
plt.show()

#%% UES-SF statistics plots

data_LandTable=data_df[[
    'FA', 
    'PU', 
    'AE', 
    'RW',
    'UES'
    #'videoID'
    ]]
varNames=data_LandTable.columns
varNamesPrint=varNames
varNamesPrint=[
    'Focused attention (FA)',
    'Perceived usability (PU)', 
    'Aesthetic appeal (AE)', 
    'Reward (RW)',
    'Overall engagement (UES)'
    #'videoID'
    ]

numRows=len(varNames)
fig, axes = plt.subplots(ncols=7, nrows=numRows+1, figsize=(8,3),
                         gridspec_kw={"width_ratios":[1.5,0.5,0.5,0.5,0.5,1,1], "height_ratios":[0.5,1,1,1,1,1]})
for ax in axes.flatten():
    ax.tick_params(labelbottom=0, labelleft=0, bottom=0, top=0, left=0, right=0)
    ax.ticklabel_format(useOffset=False, style="plain")
    for _,s in ax.spines.items():
        s.set_visible(False)
# border = fig.add_subplot(111)
# border.tick_params(labelbottom=0, labelleft=0, bottom=0, top=0, left=0, right=0)
# border.set_facecolor("None")

text_kw = dict(ha="center", va="center", size=8)
ax=axes[0,1]
ax.text(0.5, 0.5, 'min', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,2]
ax.text(0.5, 0.5, 'max', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,3]
ax.text(0.5, 0.5, 'mean', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,4]
ax.text(0.5, 0.5, 'std', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,5]
ax.text(0.5, 0.5, 'raw', **text_kw)
ax.set_facecolor([1,1,1])
ax=axes[0,6]
ax.set_facecolor([1,1,1])
ax.text(0.5, 0.5, 'hist', **text_kw)


for i,ax in enumerate(axes[1:,0]):
    ax.text(0.5, 0.5, varNamesPrint[i], **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,1]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].min():.2g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,2]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].max():.2g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,3]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].mean():.2g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,4]):
    ax.text(0.5, 0.5, f"{data_LandTable[varNames[i]].std():.2g}", **text_kw)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,5]):
    #ax.boxplot(x=varNames[i], data=data_LandTable, vert=False)
    #sns.boxplot(x=varNames[i], data=data_LandTable, ax=ax)
    data_df_rand=data_df.sample(frac=1).reset_index(drop=True)
    sns.barplot(x='index', y=varNames[i], data=data_df_rand.reset_index(), color='b', ax=ax)
    ax.set(xlabel=None,  ylabel=None)
    delta=0.0*(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())
    ax.set(ylim=(data_LandTable[varNames[i]].min()-delta,data_LandTable[varNames[i]].max()+delta))
    #sns.histplot(data_LandTable[varNames[i]], stat="density", bins=8, kde=True, ax=ax)
    ax.set_facecolor([1,1,1])
for i,ax in enumerate(axes[1:,6]):
    #ax.boxplot(x=varNames[i], data=data_LandTable, vert=False)
    #sns.boxplot(x=varNames[i], data=data_LandTable, ax=ax)
    #sns.barplot(x='index', y=varNames[i], data=data_df.reset_index(), color='b', ax=ax)
    #ax.set(xlabel=None,  ylabel=None)
    #delta=0.0*(data_LandTable[varNames[i]].max()-data_LandTable[varNames[i]].min())
    #ax.set(ylim=(data_LandTable[varNames[i]].min()-delta,data_LandTable[varNames[i]].max()+delta))
    sns.histplot(data_LandTable[varNames[i]], stat="density", bins=10, kde=True, ax=ax)
    ax.set_facecolor([1,1,1])    
fig.subplots_adjust(0.05,0.05,0.95,0.95, wspace=0.02, hspace=0.05)
plt.savefig(file_path+'UES_SF_StatsBar.pdf')
plt.show()

data_df.to_csv('../../pydove/CW30x5N/UES_LLFeaturesAll.csv')