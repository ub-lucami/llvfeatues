# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:58:45 2022

@author: Andrej Košir, Urban Burnik
data acquisition reassembled: April 29th 2022 for UES-SF
This code creates
- R2/p tables for UES-SF vs VideoFeatures
- pairwise lin and pw-lin regression curves with/without CI
get Exposure values from File 1
get 
"""
#%% start  cell
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
#import scipy.stats as scs
#import pwlf  # install by: conda install -c conda-forge pwlf

import explVar_tools as evt

# conda install -c conda-forge neurokit2

# https://towardsdatascience.com/piecewise-linear-regression-model-what-is-it-and-when-can-we-use-it-93286cfee452
# http://web.pdx.edu/~newsomj/pa551/lectur18.htm

#%%
# Load data from low level video features file:

# Video feature
file_path = '../samples/'
data_fn = 'UES_LLFeatures.csv'


data_df = pd.read_csv(file_path + data_fn)
data_rp = pd.DataFrame()

#%%
# add rows with log-lin adjustments to dataframe
# no logarithm for 'avgLightKeyAd'

for varNamesAdj in [
    'avgLightKeyAd',    
    'avgColVarAd', 
    #'avgColVarAdLog', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    #'avgShotLenAdLog'
    ]:
    data_df.insert(data_df.columns.get_loc(varNamesAdj)+1, varNamesAdj+'Adj', data_df[varNamesAdj])
    data_df[varNamesAdj+'Adj']=data_df[varNamesAdj+'Adj'].apply(lambda x: (x-data_df[varNamesAdj+'Adj'].min())/(data_df[varNamesAdj+'Adj'].max()-data_df[varNamesAdj+'Adj'].min()))
    
log_offset=[0.015, 0.1, 0.1, 0.00001]
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
#%% tentative outlier drop eg. C1: single shot.
#data_df=data_df[data_df['videoID']!='C1']


#%%
# Get model estiamtes
#selected_subscale = "AA"  # "AA"  # "RE"


# Select setup
#vID_nm, uID_nm = 'AdID', 'uID' # PsiSignals
vID_nm, uID_nm = 'videoID', 'userID' # VideoFeatures
tex_listname = file_path+'ExplVarTables.tex'
tex_graphtable= file_path+'TableOfGraphs.tex'
with open(tex_listname,"w") as file:
    file.write("\% list of tables\n")
with open(tex_graphtable,"w") as file:
    file.write("\\documentclass{article}\n")
    file.write("\\usepackage[a4paper,landscape, margin=0.5in]{geometry}\n")
    file.write("\\usepackage{graphicx}\n")
    file.write("\\usepackage{tabularx}\n")
    file.write("\\usepackage{booktabs}\n")
    file.write("\\setlength\parindent{0pt}\n")
    file.write("\\setlength{\columnsep}{1em}\n")
    file.write("\\pagestyle{empty}\n")
    file.write("\\begin{document}\n")
       
# for idx_var in range(5,11):
#     ind_var_nms = [data_df.columns[idx_var]] #['avgLightKeyAd']
#for idx_var in ['avgColVarAdLog', 'avgLightKeyAdExp']:
#for idx_var in ['avgShotLenAdLog', 'avgShotLenAd', 'avgColVarAd', 'avgMotionMeanAd', 'avgMotionStdAd', 'avgLightKeyAd']:

varNamesPrint=[
    'Lighting Key',
    'Color Variance', 
    'Mean Motion', 
    'StDev Motion', 
    'Shot Lenght ',
    'Lighting Key (log)',
    'Color Variance (log)', 
    'Mean Motion (log)', 
    'StDev Motion (log)', 
    'Shot Lenght (log)'
    ]    

for idx_var in [
    'avgLightKeyAd',    
    'avgColVarAd', 
    'avgMotionMeanAd', 
    'avgMotionStdAd', 
    'avgShotLenAd'
    # 'avgLightKeyAdAdj',
    # 'avgColVarAdAdj', 
    # 'avgMotionMeanAdAdj', 
    # 'avgMotionStdAdAdj', 
    # 'avgShotLenAdAdj'
    ]:
    ind_var_nms = [idx_var] #['avgLightKeyAd']
    if not os.path.exists(file_path+'Tables/'):
        os.makedirs(file_path+'Tables/')
    tex_tablename = file_path+'Tables/'+ind_var_nms[0] + '_tbl.tex'
    with open(tex_listname,"a") as file:
        file.write("\input{"+tex_tablename+"}\n")
    with open(tex_tablename,"w") as file:
        file.write("\\begin{table}\n")
        file.write("\\centering \n")
        file.write("\\begin{tabular}{lllllll}\n")
        file.write("\\toprule\n")
        file.write("UES-SF & \\multicolumn{3}{c}{linear} & \\multicolumn{3}{c}{pw-linear}  \\\\ \n")
        file.write("     & $R^2$     & $p$     & $CI$   & $R^2$     & $p$     & $CI$     \\\\ \n")
        file.write("\\midrule \n")
    with open(tex_graphtable, "a") as file:
        file.write("\input{"+tex_tablename+"}\n")
        file.write("\\begin{table}\n")
        file.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        file.write("\\toprule\n")
        file.write("UES-SF & Linear & Linear + CI & pw-Linear & pw-Linear \\\\ \n")
    print(ind_var_nms[0])    
    #for selected_feat in ['AE', 'RE', 'AA', 'PI','avgAE', 'avgRE', 'avgAA', 'avgPI']:
    #for selected_feat in ['AE', 'RE', 'AA', 'PI']:
    for selected_feat in ['FA', 'PU', 'AE', 'RW']:
    # for idx_dep in range(11,15):
    #     selected_feat = data_df.columns[idx_dep]
        print(selected_feat+"==================================")
        # ind_var_nms = ["f1"]
        #ind_var_nms = [ AA_features[1]] # ['f1'] # [ AE_features[1]]  # ['f1', 'f2'] # ['avgMotionStdAd', 'avgLightKeyAd'] # ['f3'] #, 'f2', 'f4']
        #ind_var_nms = [AA_features[7]] 
        #ind_var_nms = [AA_features[8]] 
        dep_var_nm = selected_feat  # 'PI'
        with open(tex_tablename,"a") as file:
            file.write(selected_feat)
        with open(tex_graphtable, "a") as file:
            file.write("\\midrule\\\\ \n")
            file.write(selected_feat)
        line_rp={('','vFeature'): ind_var_nms,
                 ('','UES-SF'): selected_feat}
        for codeIn in ["linear", "picewlin"]:        #codeIn = 'quadratic' # 'linear'#,   # 'picewlin'
            parsIn = [2]  # Number of line segments
            codeAxIn = 'lin-lin'
            
            # Set unique  to vIDuID:
            #conC = 'PsiSigs' 
            conC = 'VideoFeats'
            vID_uID_df = evt.from_vID_uID_to_vIDuID(data_df[vID_nm], data_df[uID_nm], code=conC)
            data_df.index = vID_uID_df.values.flatten()
            
            # remove rows with nan values, since score values are not nan at all, removing only ind_var_nms is enough
            data_df = data_df.dropna(subset=ind_var_nms)
            
            folds_num, folds_q = 10, 0.7
            
            # GRAF 1
            model_est_df, data_outl_df, reg_model  = evt.get_model_estimates(
                data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num, folds_q, code=codeIn, codeAx=codeAxIn, pars=parsIn, plotQ=True, xtickNames='videoID', exportPDF=file_path
            )
            
            # Get explaiend vairance
            true_var_nm = selected_feat
            est_var_nm = selected_feat
            k = len(ind_var_nms)
            R2, R2_mf, SS_tot, SS_R, p_val, p_mf_val = evt.get_expl_var(
                data_df, true_var_nm, model_est_df, est_var_nm, k
            )
            print(codeIn + ": R2, p_val: ", round(R2, 3), round(p_val, 4))
            print(codeIn + ": R2_mf, p_fm_val: ", round(R2_mf, 3), round(p_mf_val, 4))
            line_rp[(codeIn, 'R2')] = R2
            line_rp[(codeIn, 'p_val')] = p_val
            line_rp[(codeIn, 'R2_mf')] = R2_mf
            line_rp[(codeIn, 'p_mf_val')] = p_mf_val
            # Get confidence curve
            folds_num, folds_q = 100, 0.7
            mean_CI = evt.get_model_conf_curve(data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num, folds_q, code=codeIn, codeAx=codeAxIn, pars=parsIn, plotQ=True, xtickNames='videoID', exportPDF=file_path)
            print ('Mean confidence interval size: ', mean_CI)
            if codeIn=='non_linear':
                line_rp[(codeIn, 'parsIn')] = parsIn[0]
            with open(tex_graphtable,"a") as file:
                file.write(' & \includegraphics[scale=0.35]{Figs/'+ind_var_nms[0]+'_'+dep_var_nm+'_'+codeIn+'.pdf} ')
                file.write(' & \includegraphics[scale=0.35]{Figs/'+ind_var_nms[0]+'_'+dep_var_nm+'_'+codeIn+'_CI.pdf} ')
            line_rp[(codeIn, 'mean_CI')] = mean_CI                  
            with open(tex_tablename,"a") as file:
                file.write(' & '+str(round(R2_mf, 3)))
                if p_mf_val > 0.01:
                    file.write(' & '+str(round(p_mf_val, 3)))
                else:
                    file.write(' & \\textless 0.01')                 
                file.write(' & '+str(round(mean_CI, 3)))
                
        with open(tex_tablename,"a") as file:
            file.write(' \\\\ \n')
        with open(tex_graphtable,"a") as file:
            file.write('\\\\ \n')
        data_rp=data_rp.append(line_rp, ignore_index=True)
    with open(tex_tablename,"a") as file:
        file.write("\\bottomrule \n")
        file.write("\\end{tabular} \n")
        file.write("\\caption{"+ind_var_nms[0]+"} \n")
        file.write("\\label{Table:"+ind_var_nms[0]+"} \n")
        file.write("\\end{table}\n")
    data_rp.columns = pd.MultiIndex.from_tuples(line_rp.keys())
    data_rp.to_csv('test.csv')
    # def fmtp(x):
    #     if x<0.01:
    #         return '\\textless 0.01'
    #     else:
    #         return '%.2f' % x
    # def fmt(x):
    #     return '%.2f' % x
    # print(data_rp.to_latex(index=False, multicolumn=True,
    #                                     # header=[
    #                                     # (          '',     'UES-SF'),
    #                                     # # (    'linear',       'R2'),
    #                                     # # (    'linear',    'p_val'),
    #                                     # (    'linear',    '$R^2$'),
    #                                     # (    'linear', '$p$'),
    #                                     # (    'linear',  'mean_CI'),
    #                                     # # ('non_linear',       'R2'),
    #                                     # # ('non_linear',    'p_val'),
    #                                     # ('non_linear',    '$R^2$'),
    #                                     # ('non_linear', '$p$'),
    #                                     # ('non_linear',  '\\overline{||CI||}')],    
    #                                     columns=[
    #                                     (          '',     'UES-SF'),
    #                                     # (    'linear',       'R2'),
    #                                     # (    'linear',    'p_val'),
    #                                     (    'linear',    'R2_mf'),
    #                                     (    'linear', 'p_mf_val'),
    #                                     (    'linear',  'mean_CI'),
    #                                     # ('non_linear',       'R2'),
    #                                     # ('non_linear',    'p_val'),
    #                                     ('non_linear',    'R2_mf'),
    #                                     ('non_linear', 'p_mf_val'),
    #                                     ('non_linear',  'mean_CI')], 
    #                                 escape = False,
    #                                 formatters=[None,fmt,fmtp,fmt,fmt,fmtp,fmt]))                           
    #                                 # float_format='%.2f'))
    with open(tex_graphtable,"a") as file:
        file.write("\\bottomrule \n")
        file.write("\\end{tabular} \n")
        file.write("\\caption{Graph: "+ind_var_nms[0]+"} \n")
        file.write("\\label{Table:G_"+ind_var_nms[0]+"} \n")
        file.write("\\end{table}\n")
with open(tex_graphtable,"a") as file:
    file.write("\\end{document}\n")




