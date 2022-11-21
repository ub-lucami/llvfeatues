# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:22:44 2022

@author: Andrej Košir
"""


from num2tex import num2tex
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit
#from sklearn.metrics import r2_score
import scipy.stats as scs
from scipy.optimize import minimize, rosen
import pwlf
import os


# -----------------------------------------------------------------------------
# @brief Quadratic optimisation mode
def model_f(p, X_np):
    return p[0]*(X_np-p[1])**2 + p[2]

# @brief quadratic optimisation model
def cost_f(p, X_np, y_np):
    
    #return np.sum(np.abs(y_np - (model_f(p, X_np))))
    return np.sum(np.power(y_np - (model_f(p, X_np)), 2))


# -----------------------------------------------------------------------------
# @brief sort DataFrame rows preseving indices and column names
def sort_rows_pd(data_df):
    
    sorted_data_np = np.sort(data_df.values, axis=1)
    sorted_data_df = pd.DataFrame(sorted_data_np, index=data_df.index, columns=data_df.columns)
    
    return sorted_data_df

#cols = ['f1', 'f2', 'f3']
#inds = [1,3,5,22,6,7,8,9]
#data_np = np.array([[1,3,5], [5,4,6], [3, 11, 8], [9, 7, 3], [2, 6, 1], [12, 14, 8], [6, 7, 3], [11, 4, 6]])
#d_df = pd.DataFrame(data_np, index=inds, columns=cols)
# sd_df = sort_rows_pd(d_df)
# print (sd_df)


# -----------------------------------------------------------------------------
# @brief convert vIDuID to vID, uID
def from_vIDuID_to_vID_uID(vIDuID, code='PsySigs'):
    c = 1000
    return (vIDuID // c), (vIDuID % c)

# @brief convert 
# @Note: in Feature files, users were mapped: if lab Exp2 -> uID=uID, if Clickworker Exp1 -> uID=uID + 100
def from_vID_uID_to_vIDuID(vID, uID, code='PsySigs'):
    
    c = 1000
    if code == 'PsySigs':
        return c*vID + uID
    if code == 'VideoFeats':
        #cvID = pd.DataFrame(data=np.array([int((x)[3:]) for x in vID]), index=vID.index, columns=['uvID'])
        temp = defaultdict(lambda: len(temp))
        cvID = pd.DataFrame([temp[ele] for ele in vID], index=vID.index, columns=['uvID'])+1
        cuID = pd.DataFrame(data=np.array([u if u < 100 else u-100 for u in uID]), index=uID.index, columns=['uvID']) 
        return c*cvID + cuID

#uID, vID = 32, 3
#vIDuID = from_vID_uID_to_vIDuID(vID, uID)
#vIDu, uIDu = from_vIDuID_to_vID_uID(vIDuID)
#vID_uID_df = from_vID_uID_to_vIDuID(d_df['f1'], d_df['f3'])
#d_df.index = vID_uID_df
# [x for ii, x in enumerate(inds) if x in inds[:ii]] # Duplicates


# @brief convert columns to log scale: x -> log(1+x)
def to_log(data_np):
    
    n, m = data_np.shape[0], data_np.shape[1]
    log_data_np = np.zeros((n, m))
    for jj in range(m):
        add_c = 1.0 - min(data_np[:, jj])
        log_data_np[:, jj] = np.log10(add_c + data_np[:, jj])
    
    return log_data_np

# @brief convert columns to log scale: x -> log(1+x)
def to_log_df(data_df):
    
    n, m = data_df.shape[0], data_df.shape[1]
    log_data_df = pd.DataFrame(index=data_df.index, columns=data_df.columns)
    for jj in range(m):
        add_c = 1.0 - min(data_df.iloc[:, jj])
        log_data_df.iloc[:, jj] = np.log10(add_c + data_df.iloc[:, jj])
    
    return log_data_df

#x_df = pd.DataFrame(data=[[1,2], [3, 4], [-1, 5]])
#y_df = to_log_df(x_df)


# @brief remove outlier
def remove_outliers(data_df, ind_var_nms, dep_var_nm, sig_cut):

    X_df, y_df = data_df[ind_var_nms], data_df[dep_var_nm]
    data_outl_df = data_df.copy()
    
    selector_df = (-sig_cut < scs.zscore(X_df)) & (scs.zscore(X_df) < sig_cut)
    selector = np.array(selector_df).flatten()
    X_df = X_df[selector]
    y_df = y_df[selector]
    data_outl_df = data_outl_df[selector]

    return data_outl_df, X_df, y_df


# -----------------------------------------------------------------------------
# @brief plot model fit with confidence interval if given
# @par data_df full data frame
# @par ind_var_nms
# @par dep_var_nm
# @par estY_df model curve
# @par estY_lb_df curve lower bound 
# @par estY_ub_df curve upper bound
def plot_model_fit_df(data_df, ind_var_nms, dep_var_nm, estY_df, estY_lb_df=None, estY_ub_df=None, code='linear', codeAx='lin-lin', colGroup='', xtickNames='', exportPDF=False):
    
    # Sort according to indep. var.
    X_df, y_df = data_df[ind_var_nms], data_df[dep_var_nm]
    if codeAx == 'log-lin':
        X_df = to_log_df(X_df)
    if isinstance(estY_lb_df, pd.DataFrame) and isinstance(estY_ub_df, pd.DataFrame):
        Xyey_df = pd.concat((X_df, y_df, estY_df, estY_lb_df, estY_ub_df), axis=1)
    else:
        Xyey_df = pd.concat((X_df, y_df, estY_df), axis=1)
        
    sXyey_df = Xyey_df.sort_values(by=ind_var_nms)
    isCI="";
    plt.figure()
    if colGroup == "":
        plt.plot(
            sXyey_df[ind_var_nms].values,
            sXyey_df[dep_var_nm].values,
            color="g",
            marker='x', 
            linestyle='', 
            ms=3, 
            label="true: " + dep_var_nm,
        )
    else:
        groups=data_df.groupby(colGroup)
        for name, group in groups:
            plt.plot(group[ind_var_nms[0]], group[dep_var_nm], marker='x', linestyle='', ms=2, label=name)
            
    plt.plot(
        sXyey_df[ind_var_nms].values,
        sXyey_df.iloc[:,2].values,
        "-",
        color="b",
        label="model: " + dep_var_nm,
    )
    if sXyey_df.shape[1] > 3:
        plt.plot(
            sXyey_df[ind_var_nms].values,
            sXyey_df.iloc[:,3].values,
            "--",
            color="r",
            label="lb: " + dep_var_nm,
        )
    if sXyey_df.shape[1] > 4:
        isCI="_CI";
        plt.plot(
            sXyey_df[ind_var_nms].values,
            sXyey_df.iloc[:,4].values,
            "--",
            color="r",
            label="ub: " + dep_var_nm,
        )
    plt.grid()
    plt.xlabel(ind_var_nms[0])
    plt.ylabel(dep_var_nm)
    plt.legend()
    # ticks from video_id
    if xtickNames != '':
        df_xticks=data_df[[ind_var_nms[0],xtickNames]].drop_duplicates(subset=xtickNames)
        plt.xticks(df_xticks[ind_var_nms[0]], df_xticks[xtickNames], rotation = 90,ha='center')
        plt.text(df_xticks[ind_var_nms[0]].min(),-0.2,'${:.2g}$'.format(num2tex(df_xticks[ind_var_nms[0]].min())),ha='center', va='center')
        plt.text(df_xticks[ind_var_nms[0]].max(),-0.2,'${:.2g}$'.format(num2tex(df_xticks[ind_var_nms[0]].max())),ha='center', va='center')
        groups=data_df.groupby(xtickNames)
        for name, group in groups:    
            plt.plot(group[ind_var_nms[0]].mean(), group[dep_var_nm].mean(), marker='_', color="k", linestyle='', ms=6, label=name)
    try:
        os.makedirs(exportPDF+"Figs")
    except FileExistsError:
        # directory already exists
        pass
    plt.ylim(0.5,5.5)
    if exportPDF:
        plt.savefig(exportPDF+'Figs\\'+ind_var_nms[0]+'_'+dep_var_nm+'_'+code+isCI+'.pdf')
    plt.show()
    
    return 0

# -----------------------------------------------------------------------------
# @brief Fit the model and get model estimates
# @par data_df input data
# @par ind_var_nms names of input (independent) variables
# @par dep_var_nm name of dependent variable
# @par folds_num number of folds
# @par folds_q fraction of data points for training
# @par code select model among linera and picewlin
# @par pars a list of parameters passed to fitting function. For 'picewlin' it is the nunmber of line segments;
# @return model_est_df dataframe of model estimates
# @return reg_model regression model
def get_model_estimates(
    data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num=0, folds_q=0.7, code="linear", codeAx='lin-lin', codeOutlIn=False, pars=[], plotQ=False, colGroup="", xtickNames='', exportPDF=''
):
    
    print ('get_model_estimates started ...')
    
    # Remove outliers
    if codeOutlIn:
        if len(pars)>1:
            sig_cut = pars[1]
        else: 
            sig_cut = 2.5
        data_df, X_df, y_df = remove_outliers(data_df, ind_var_nms, dep_var_nm, sig_cut)
    else:
        X_df, y_df = data_df[ind_var_nms], data_df[dep_var_nm]
    
    
    # Rescale independent vairable
    if codeAx == 'log-lin':
        X_df = to_log_df(X_df)
    
    
    X_np = np.array(X_df).astype(float)
    y_np = np.array(y_df).astype(float)
    
    
    
    
    if code == "linear":

        # Get lineat model fit
        lin_model = LinearRegression().fit(X_np, y_np)
        estY = lin_model.predict(X_np)
        
        # Set estimates
        columnsIn = [dep_var_nm] + ['fld' + str(i) for i in range(folds_num)]
        
        model_est_df = pd.DataFrame(index=X_df.index, columns=columnsIn)
        model_est_df[dep_var_nm] = estY

        reg_model = lin_model
        
        
        # Do folding
        if folds_num > 0:
            
            ii = 0
            ss = ShuffleSplit(n_splits=folds_num, test_size=folds_q) #, random_state=0)
            for train_index, test_index in ss.split(X_df):
                
                
                # Select folds
                d_ind_train_df, d_ind_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
                d_dep_train_df, d_dep_test_df = y_df.iloc[train_index], y_df.iloc[test_index]
               
               
                # Training 
                X_train_np = np.array(d_ind_train_df).astype(float)
                y_train_np = np.array(d_dep_train_df).astype(float)
                lin_model = LinearRegression().fit(X_train_np, y_train_np)
               
               
                # Estimate
                X_test_np = np.array(d_ind_test_df).astype(float)
                curr_estY = lin_model.predict(X_test_np)
            
                # Set results 
                curr_est_df = pd.DataFrame(data=curr_estY, index=d_ind_test_df.index, columns=['fld' + str(ii)])
                model_est_df['fld' + str(ii)] = curr_est_df
                
                ii += 1


        
    if code == "quadratic":

        # Get quadratic model fit
        m_x, h_x, M_x = np.min(X_np), np.mean(X_np), np.max(X_np)
        m_y, h_y, M_y = np.min(y_np), np.mean(y_np), np.max(y_np)
        
        # Test CUP
        p0 = [(M_y-h_y)/(M_x-h_x), h_x, h_y]
        cupres = minimize(cost_f, p0, args=(X_np, y_np), method='BFGS',
               options={'gtol': 1e-6, 'disp': True})
        pcup_S = cupres.x
        cup_C = cost_f(pcup_S, X_np, y_np)
        
        # Test CAP
        p0 = [(m_y-h_y)/(M_x-h_x), h_x, h_y]
        capres = minimize(cost_f, p0, args=(X_np, y_np), method='BFGS',
               options={'gtol': 1e-6, 'disp': True})
        pcap_S = capres.x
        cap_C = cost_f(pcap_S, X_np, y_np)
        
        if cap_C < cup_C:
            p_S = pcap_S
        else:
            p_S = pcup_S
        
        quad_model = p_S
        
        
        # reg.score(X, y)
        # reg.coef_
        # reg.intercept_
        estY = model_f(p_S, X_np)
        
        # Set estimates
        columnsIn = [dep_var_nm] + ['fld' + str(i) for i in range(folds_num)]
        
        model_est_df = pd.DataFrame(index=y_df.index, columns=columnsIn)
        model_est_df[dep_var_nm] = estY

        reg_model = quad_model
        
        
        # Do folding
        if folds_num > 0:
            
            ii = 0
            ss = ShuffleSplit(n_splits=folds_num, test_size=folds_q) #, random_state=0)
            for train_index, test_index in ss.split(X_df[ind_var_nms]):
                
                
                # Select folds
                d_ind_train_df, d_ind_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
                d_dep_train_df, d_dep_test_df = y_df.iloc[train_index], y_df.iloc[test_index]
               
               
                # Training 
                X_train_np = np.array(d_ind_train_df).astype(float)
                y_train_np = np.array(d_dep_train_df).astype(float)
                
                
                # Get quadratic model fit
                m_x, h_x, M_x = np.min(X_train_np), np.mean(X_train_np), np.max(X_train_np)
                m_y, h_y, M_y = np.min(y_train_np), np.mean(y_train_np), np.max(y_train_np)
                
                # Test CUP
                p0 = [(M_y-h_y)/(M_x-h_x), h_x, h_y]
                cupres = minimize(cost_f, p0, args=(X_train_np, y_train_np), method='BFGS',
                       options={'gtol': 1e-6, 'disp': True})
                pcup_S = cupres.x
                cup_C = cost_f(pcup_S, X_train_np, y_train_np)
                
                # Test CAP
                p0 = [(m_y-h_y)/(M_x-h_x), h_x, h_y]
                capres = minimize(cost_f, p0, args=(X_train_np, y_train_np), method='BFGS',
                       options={'gtol': 1e-6, 'disp': True})
                pcap_S = capres.x
                cap_C = cost_f(pcap_S, X_train_np, y_train_np)
                
                if cap_C < cup_C:
                    p_S = pcap_S
                else:
                    p_S = pcup_S
               
               
                # Estimate
                X_test_np = np.array(d_ind_test_df).astype(float)
                curr_estY = model_f(p_S, X_test_np)
            
                # Set results 
                curr_est_df = pd.DataFrame(data=curr_estY, index=d_ind_test_df.index, columns=['fld' + str(ii)])
                model_est_df['fld' + str(ii)] = curr_est_df
                
                ii += 1
        
        

    if code == "picewlin":

        # Get nonlineat model fit - pice
        if len(ind_var_nms) > 1:
            print(
                "Nonlinear modeling is not supported for several independent variables."
            )
            return 0, 0

        # fit your data (x and y)
        X_np = X_np.flatten()
        myPWLF = pwlf.PiecewiseLinFit(X_np, y_np, disp_res=True)
        # print ('non-linear: ', )

        # fit the data for n line segments

        if len(pars) > 0:
            num_of_points = pars[0]
        else:
            num_of_points = 3

        z = myPWLF.fit(num_of_points)


        # Number of points used in modeling 
        num_of_p_lst = [((z[ii] <= X_np) & (X_np <= z[ii+1])).sum() for ii in range(len(z)-1)]
        min_num_of_p = min(num_of_p_lst)
        if min_num_of_p < 10:
            print ('Min num of points: ', num_of_p_lst)
        
        # predict for the determined points
        estY = myPWLF.predict(X_np)
        
        
        # Set estimates
        columnsIn = [dep_var_nm] + ['fld' + str(i) for i in range(folds_num)]
        model_est_df = pd.DataFrame(index=data_df.index, columns=columnsIn)
        model_est_df[dep_var_nm] = estY

        # calculate statistics
        # p = myPWLF.p_values(method='non-linear', step_size=1e-4) #p-values
        # se = myPWLF.se  # standard errors

        reg_model = myPWLF
        
        
        # Do folding
        if folds_num > 0:
            
            ii = 0
            ss = ShuffleSplit(n_splits=folds_num, test_size=folds_q) #, random_state=0)
            for train_index, test_index in ss.split(data_df[ind_var_nms]):
                 
                # Select folds
                d_ind_train_df, d_ind_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
                d_dep_train_df, d_dep_test_df = y_df.iloc[train_index], y_df.iloc[test_index]
               
                # Training 
                X_train_np = np.array(d_ind_train_df).astype(float).flatten()
                y_train_np = np.array(d_dep_train_df).astype(float)
                c_PWLF = pwlf.PiecewiseLinFit(X_train_np, y_train_np, disp_res=False)
                z = c_PWLF.fit(num_of_points)
                
                # Test data size for subsegments
                #num_of_p_lst = [((z[ii] <= X_np) & (X_np <= z[ii+1])).sum() for ii in range(len(z)-1)]
                #min_num_of_p = min(num_of_p_lst)
                #if min_num_of_p < 10:
                #    print ('Min num of points: ', num_of_p_lst)
               
                # Estimate
                X_test_np = np.array(d_ind_test_df).astype(float).flatten()
                curr_estY = myPWLF.predict(X_test_np)
            
                # Set results 
                curr_est_df = pd.DataFrame(data=curr_estY, index=d_ind_test_df.index, columns=['fld' + str(ii)])
                model_est_df['fld' + str(ii)] = curr_est_df
                
                
                
                ii += 1
        

    # Plot
    if plotQ and (len(ind_var_nms) == 1):
        estY_df = pd.DataFrame(estY, index=data_df.index)
        print(codeAx+'-----------\n')
        plot_model_fit_df(data_df, ind_var_nms, dep_var_nm, estY_df, code=code, codeAx=codeAx, colGroup=colGroup, xtickNames=xtickNames, exportPDF=exportPDF)
    else:
        print("Plots of the model is not supported for several independent variables.")


    print ('... get_model_estimates started done.')
    
    return model_est_df, data_df, reg_model



# -----------------------------------------------------------------------------
# @brief get curve confident interval: it subsamples the data and estimate curve confidence 
#  interval to test its stability
# @par data_df input dataif codeAx == 'log-lin':
# @par ind_var_nms names of input (independent) variables
# @par dep_var_nm name of dependent variable
# @par code select model among linera and picewlin
# @par pars a list of parameters passed to fitting function. For 'picewlin' it is the nunmber of line segments;
# @par folds_num number of folds, typically 300
# @par folds_q proportion of data taken into a single fold
# @par plotQ plot the result?
# @return mean_CI mean confidence interval
def get_model_conf_curve(data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num, folds_q, code='linear', codeAx='lin-lin', codeOutlIn=False, pars=[], plotQ=False, colGroup='', xtickNames='',exportPDF=''):
    
    print ('get_model_conf_curve started ...')
    
    al = 0.05
    n, N = data_df[ind_var_nms].shape
    estimates_df = pd.DataFrame(index=data_df.index, columns=['fld'+str(i) for i in range(folds_num)])
    
    
    # Remove outliers 
    if codeOutlIn:
        if len(pars)>1:
            sig_cut = pars[1]
        else: 
            sig_cut = 2.5
        data_df, X_df, y_df = remove_outliers(data_df, ind_var_nms, dep_var_nm, sig_cut)
    else:
        X_df, y_df = data_df[ind_var_nms], data_df[dep_var_nm]
    
    # Rescale X
    if codeAx == 'log-lin':
        X_df = to_log_df(X_df)
    
    
    
    if code == "linear":
        
        # Do sampling and modeling
        for ii in range(folds_num):
            
            # Get subsamples
            data_subs_df = data_df.sample(frac=folds_q)
        
            # Get linear model fit
            X_np = np.array(data_subs_df[ind_var_nms]).astype(float)
            if codeAx == 'log-lin':
                X_np = to_log(X_np)
            y_np = np.array(data_subs_df[dep_var_nm]).astype(float)
            lin_model = LinearRegression().fit(X_np, y_np)

            # Get and store linear fit
            estY_np = lin_model.predict(X_np)
            estY_df = pd.DataFrame(data=estY_np, index=data_subs_df.index, columns=['fld'+str(ii)])
            estimates_df['fld'+str(ii)] = estY_df

        # Get bands 
        sorted_estimates_df = sort_rows_pd(estimates_df)
        c_lb = sorted_estimates_df.quantile(q=al, axis=1, numeric_only=True)
        c_mu = sorted_estimates_df.quantile(q=0.5, axis=1, numeric_only=True)
        c_ub = sorted_estimates_df.quantile(q=1.0-al, axis=1, numeric_only=True)
        
        # Get mean CI size
        mean_CI = (c_ub-c_lb).mean()
            

    if code == "quadratic":
        
        # Do sampling and modeling
        for ii in range(folds_num):
            
            # Get subsamples
            data_subs_df = data_df.sample(frac=folds_q)
        
            # Get data
            X_np = np.array(data_subs_df[ind_var_nms]).astype(float)
            if codeAx == 'log-lin':
                X_np = to_log(X_np)
            y_np = np.array(data_subs_df[dep_var_nm]).astype(float)
            
            # Quadratic optimisation
            m_x, h_x, M_x = np.min(X_np), np.mean(X_np), np.max(X_np)
            m_y, h_y, M_y = np.min(y_np), np.mean(y_np), np.max(y_np)
            p0 = [(M_y-h_y)/(M_x-h_x), h_x, h_y]
            cupres = minimize(cost_f, p0, args=(X_np, y_np), method='BFGS',
                   options={'gtol': 1e-6, 'disp': False})
            pcup_S = cupres.x
            cup_C = cost_f(pcup_S, X_np, y_np)
            
            # Test CAP
            p0 = [(m_y-h_y)/(M_x-h_x), h_x, h_y]
            capres = minimize(cost_f, p0, args=(X_np, y_np), method='BFGS',
                   options={'gtol': 1e-6, 'disp': False})
            pcap_S = capres.x
            cap_C = cost_f(pcap_S, X_np, y_np)
            
            if cap_C < cup_C:
                p_S = pcap_S
            else:
                p_S = pcup_S
            
            quad_model = p_S
            
            lin_model = LinearRegression().fit(X_np, y_np)

            # Get and store linear fit
            estY_np = lin_model.predict(X_np)
            estY_df = pd.DataFrame(data=estY_np, index=data_subs_df.index, columns=['fld'+str(ii)])
            estimates_df['fld'+str(ii)] = estY_df

        # Get bands 
        sorted_estimates_df = sort_rows_pd(estimates_df)
        c_lb = sorted_estimates_df.quantile(q=al, axis=1, numeric_only=True)
        c_mu = sorted_estimates_df.quantile(q=0.5, axis=1, numeric_only=True)
        c_ub = sorted_estimates_df.quantile(q=1.0-al, axis=1, numeric_only=True)
        
        # Get mean CI size
        mean_CI = (c_ub-c_lb).mean()
            


    if code == "picewlin":
        
        if len(pars) > 0:
            num_of_points = pars[0]
        else:
            num_of_points = 2

        # Get nonlineat model fit - pice
        if len(ind_var_nms) > 1:
            print(
                "Nonlinear modeling is not supported for several independent variables."
            )
            return 0

        

        # Do sampling and modeling
        for ii in range(folds_num):
            
            # Get subsamples
            data_subs_df = data_df.sample(frac=folds_q)
        
            # Get lineat model fit
            X_np = np.array(data_subs_df[ind_var_nms]).astype(float).flatten()
            if codeAx == 'log-lin':
                X_df = to_log_df(X_df)
            y_np = np.array(data_subs_df[dep_var_nm]).astype(float)
            myPWLF = pwlf.PiecewiseLinFit(X_np, y_np, disp_res=False)
            
            z = myPWLF.fit(num_of_points)
        
            # Get and store linear fit
            estY_np = myPWLF.predict(X_np)
            estY_df = pd.DataFrame(data=estY_np, index=data_subs_df.index, columns=['fld'+str(ii)])
            estimates_df['fld'+str(ii)] = estY_df
            
            # Number of points used in modeling 
            #num_of_p_lst = [((z[ii] <= X_np) & (X_np <= z[ii+1])).sum() for ii in range(len(z)-1)]
            #min_num_of_p = min(num_of_p_lst)
            #if min_num_of_p < 10:
            #    print ('Min num of points: ', num_of_p_lst)
        
        # Get bands 
        sorted_estimates_df = sort_rows_pd(estimates_df)
        c_lb = sorted_estimates_df.quantile(q=al, axis=1, numeric_only=True)
        c_mu = sorted_estimates_df.quantile(q=0.5, axis=1, numeric_only=True)
        c_ub = sorted_estimates_df.quantile(q=1.0-al, axis=1, numeric_only=True)
        
        
        # Get mean CI size
        mean_CI = (c_ub-c_lb).mean()
        

    # Plot
    if plotQ and (len(ind_var_nms) == 1):
        c_mu_df = pd.DataFrame(c_mu, index=data_df.index)
        c_lb_df = pd.DataFrame(c_lb, index=data_df.index)
        c_ub_df = pd.DataFrame(c_ub, index=data_df.index)
        print(codeAx+'++++++++++++\n')
        plot_model_fit_df(data_df, ind_var_nms, dep_var_nm, c_mu_df, c_lb_df, c_ub_df, code=code, colGroup=colGroup, xtickNames=xtickNames, exportPDF=exportPDF)
    else:
        print("Plots of the model is not supported for several independent variables.")
    
    print ('... get_model_conf_curve done.')
    
    return mean_CI


# -----------------------------------------------------------------------------
# @brief get explained variance = R^2 = coefficient of determination and some sumes of squares
# @par data_df
# @par true_var_nm
# @par model_est_df
# @par est_var_nm
# @par k number of predictors
# @return q explained variance
# @return SS_ror total sum of squares
# @return SS_R residual sum of squares
def get_expl_var(data_df, true_var_nm, model_est_df, est_var_nm, k):

    print ('get_expl_var started ...')    

    # Get dfs   
    Y_df = data_df[true_var_nm]
    eY_df = model_est_df[est_var_nm]
    n = Y_df.shape[0]
    
    # Summs of squares
    SS_tot = ((Y_df - Y_df.mean()) ** 2).sum() ## (N - 1) * Y.var()  # Total summs of squares
    SS_R = ((Y_df - eY_df) ** 2).sum() # (N - 1) * (Y - eY).var()  # Residual summs of squares


    # Coefficient of determination - full set
    # print (r2_score(Y, eY))
    R2 = (SS_tot - SS_R) / SS_tot
    R2_adj = 1 - (1-R2)*(n-1)/(n-k-1)

    # Stat test for H0 = [R2 == 0]
    F = (R2 / k) / ((1 - R2) / (n - k - 1))

    p_val = scs.f.sf(F, k, n - k - 1)
    
    # Coefficient of determination - folding
    folds_num = model_est_df.shape[1]-1
    if folds_num > 1:
        R2_lst_np = np.zeros(folds_num)
        for fold_ii in range(folds_num):
            
            Y_df = data_df[true_var_nm].copy()
            curr_eY_df = model_est_df['fld'+str(fold_ii)]
            
            # Clean nans
            nan_selector = curr_eY_df.isna()
            curr_eY_df.dropna(inplace=True)
            
            Y_df[nan_selector] = None
            curr_Y_df = Y_df.dropna()
            
            curr_SS_tot = ((curr_Y_df - curr_Y_df.mean()) ** 2).sum() ## (N - 1) * Y.var()  # Total summs of squares
            curr_SS_R = ((curr_Y_df - curr_eY_df) ** 2).sum() # (N - 1) * (Y - eY).var()  # Residual summs of squares
    
    
            # Coefficient of determination - full set
            R2_lst_np[fold_ii] = (curr_SS_tot - curr_SS_R) / curr_SS_tot
            
        # Get mean R2    
        R2_lst_np[np.where(R2_lst_np<0)] = 0
        R2_mf = np.nanmean(R2_lst_np)
        
        # Get p_val
        F_mf = (R2_mf / k) / ((1 - R2_mf) / (n - k - 1))
        p_mf_val = scs.f.sf(F_mf, k, n - k - 1)
    else:
        R2_mf = 0
        p_mf_val = 1.0

    print ('... get_expl_var.') 
    return R2, R2_mf, SS_tot, SS_R, p_val, p_mf_val

#Y_np = np.array(data_df[true_var_nm])
#eY_np = np.array(model_est_df[est_var_nm])
#N = Y_np.shape[0]

# Summs of squares
#SS_tot = ((Y_np - Y_np.mean()) ** 2).sum() ## (N - 1) * Y.var()  # Total summs of squares
#SS_R = ((Y_np - eY_np) ** 2).sum() # (N - 1) * (Y - eY).var()  # Residual summs of squares