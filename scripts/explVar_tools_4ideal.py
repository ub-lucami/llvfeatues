# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:22:44 2022

@author: Andrej Košir
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit
#from sklearn.metrics import r2_score
import scipy.stats as scs
import pwlf


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
def from_vIDuID_to_vID_uID(vIDuID, code='PsiSigs'):
    c = 1000
    return (vIDuID // c), (vIDuID % c)

# @brief convert 
# @Note: in Feature files, users were mapped: if lab Exp2 -> uID=uID, if Clickworker Exp1 -> uID=uID + 100
def from_vID_uID_to_vIDuID(vID, uID, code='PsiSigs'):
    
    c = 1000
    if code == 'PsiSigs':
        return c*vID + uID
    if code == 'VideoFeats':
        cvID = pd.DataFrame(data=np.array([int(list(x)[1]) for x in vID]), index=vID.index, columns=['uvID'])
        cuID = pd.DataFrame(data=np.array([u if u < 100 else u-100 for u in uID]), index=uID.index, columns=['uvID']) 
        return c*cvID + cuID

#uID, vID = 32, 3
#vIDuID = from_vID_uID_to_vIDuID(vID, uID)
#vIDu, uIDu = from_vIDuID_to_vID_uID(vIDuID)
#vID_uID_df = from_vID_uID_to_vIDuID(d_df['f1'], d_df['f3'])
#d_df.index = vID_uID_df
# [x for ii, x in enumerate(inds) if x in inds[:ii]] # Duplicates



# -----------------------------------------------------------------------------
# @brief Fit the model and get model estimates
# @par data_df input data
# @par ind_var_nms names of input (independent) variables
# @par dep_var_nm name of dependent variable
# @par folds_num number of folds
# @par folds_q fraction of data points for training
# @par code select model among linera and non_linear
# @par pars a list of parameters passed to fitting function. For 'non_linear' it is the nunmber of line segments;
# @return model_est_df dataframe of model estimates
# @return reg_model regression model
def get_model_estimates(
    data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num=0, folds_q=0.7, code="linear", pars=[], plotQ=False, exportPDF=''
):
    
    
    if code == "linear":

        # Get lineat model fit
        X_np = np.array(data_df[ind_var_nms]).astype(float)
        y_np = np.array(data_df[dep_var_nm]).astype(float)
        lin_model = LinearRegression().fit(X_np, y_np)
        # reg.score(X, y)
        # reg.coef_
        # reg.intercept_
        estY = lin_model.predict(X_np)
        
        # Set estimates
        columnsIn = [dep_var_nm] + ['fld' + str(i) for i in range(folds_num)]
        
        model_est_df = pd.DataFrame(index=data_df.index, columns=columnsIn)
        model_est_df[dep_var_nm] = estY

        reg_model = lin_model
        
        
        # Do folding
        if folds_num > 0:
            
            ii = 0
            ss = ShuffleSplit(n_splits=folds_num, test_size=folds_q) #, random_state=0)
            for train_index, test_index in ss.split(data_df[ind_var_nms]):
                
                
                # Select folds
                d_ind_train_df, d_ind_test_df = data_df[ind_var_nms].iloc[train_index], data_df[ind_var_nms].iloc[test_index]
                d_dep_train_df, d_dep_test_df = data_df[dep_var_nm].iloc[train_index], data_df[dep_var_nm].iloc[test_index]
               
               
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
        
        # Plot
        if plotQ and (len(ind_var_nms) == 1):

            # Sort according to indep. var.
            Xyey_np = np.column_stack((X_np, y_np, estY))
            sXyey_np = Xyey_np[Xyey_np[:, 0].argsort()]

            plt.figure()
            # here for color grouping or uncomment block below
            groups=data_df.groupby('videoID')
            for name, group in groups:
                plt.plot(group[ind_var_nms[0]], group[dep_var_nm], marker='.', linestyle='', ms=12, label=name)

            # plt.plot(
            #     sXyey_np[:, 0].values,
            #     sXyey_np[:, 1].values,
            #     "o",
            #     color="g",
            #     label="true: " + dep_var_nm,
            # )
            plt.plot(
                sXyey_np[:, 0],
                sXyey_np[:, 2],
                "-",
                color="b",
                label="est: " + dep_var_nm,
            )
            plt.grid()
            plt.xlabel(ind_var_nms[0])
            plt.ylabel(dep_var_nm)
            plt.legend()
            try:
                os.makedirs(exportPDF+"Figs")
            except FileExistsError:
                # directory already exists
                pass
            plt.ylim(0.5,5.5)
            if exportPDF:
                plt.savefig(exportPDF+'Figs\\'+ind_var_nms[0]+'_'+dep_var_nm+'_L'+'.pdf')
            plt.show()

        else:
            print(
                "Plots of linear model is not supported for several independent variables."
            )

    if code == "non_linear":

        # Get nonlineat model fit - pice
        if len(ind_var_nms) > 1:
            print(
                "Nonlinear modeling is not supported for several independent variables."
            )
            return 0, 0

        # fit your data (x and y)
        X_np = np.array(data_df[ind_var_nms]).astype(float).flatten()
        y_np = np.array(data_df[dep_var_nm]).astype(float)
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
                d_ind_train_df, d_ind_test_df = data_df[ind_var_nms].iloc[train_index], data_df[ind_var_nms].iloc[test_index]
                d_dep_train_df, d_dep_test_df = data_df[dep_var_nm].iloc[train_index], data_df[dep_var_nm].iloc[test_index]
               
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
        

        if plotQ:

            # Sort according to indep. var.
            Xyey_np = np.column_stack((X_np, y_np, estY))
            sXyey_np = Xyey_np[Xyey_np[:, 0].argsort()]

            plt.figure()
            plt.plot(
                sXyey_np[:, 0],
                sXyey_np[:, 1],
                "o",
                color="g",
                label="true: " + dep_var_nm,
            )
            plt.plot(
                sXyey_np[:, 0],
                sXyey_np[:, 2],
                "-",
                color="b",
                label="est: " + dep_var_nm,
            )
            plt.grid()
            plt.xlabel(ind_var_nms[0])
            plt.ylabel(dep_var_nm)
            plt.legend()
            try:
                os.makedirs(exportPDF+"Figs")
            except FileExistsError:
                # directory already exists
                pass
            plt.ylim(0.5,5.5)
            if exportPDF:
                plt.savefig(exportPDF+'Figs\\'+ind_var_nms[0]+'_'+dep_var_nm+'_NL'+'_'+str(num_of_points)+'pts.pdf')
            plt.show()
            
    return model_est_df, reg_model



# -----------------------------------------------------------------------------
# @brief get curve confident interval: it subsamples the data and estimate curve confidence 
#  interval to test its stability
# @par data_df input data
# @par ind_var_nms names of input (independent) variables
# @par dep_var_nm name of dependent variable
# @par code select model among linera and non_linear
# @par pars a list of parameters passed to fitting function. For 'non_linear' it is the nunmber of line segments;
# @par folds_num number of folds, typically 300
# @par folds_q proportion of data taken into a single fold
# @par plotQ plot the result?
# @return mean_CI mean confidence interval
def get_model_conf_curve(data_df, ind_var_nms, dep_var_nm, vID_nm, uID_nm, folds_num, folds_q, code="linear", pars=[], plotQ=False, exportPDF=''):
    
    print ('Curve conf started ...')
    
    al = 0.05
    n, N = data_df[ind_var_nms].shape
    estimates_df = pd.DataFrame(index=data_df.index, columns=['fld'+str(i) for i in range(folds_num)])
    
    
    if code == "linear":
        
        # Do sampling and modeling
        for ii in range(folds_num):
            
            # Get subsamples
            data_subs_df = data_df.sample(frac=folds_q)
        
            # Get linear model fit
            X_np = np.array(data_subs_df[ind_var_nms]).astype(float)
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
            

        if plotQ and (len(ind_var_nms) == 1):

            # Sort according to indep. var.
            Xyey_df = pd.concat((data_df[ind_var_nms], data_df[dep_var_nm], c_lb, c_mu, c_ub), axis=1)
            sXyey_df = Xyey_df.sort_values(by=ind_var_nms)

            plt.figure()
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[dep_var_nm].values,
                "o",
                color="g",
                label="true: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.5].values,
                "-",
                color="b",
                label="mean: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.05].values,
                "--",
                color="r",
                label="lb: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.95].values,
                "--",
                color="r",
                label="ub: " + dep_var_nm,
            )
            plt.grid()
            plt.xlabel(ind_var_nms[0])
            plt.ylabel(dep_var_nm)
            plt.legend()
            try:
                os.makedirs(exportPDF+"Figs")
            except FileExistsError:
                # directory already exists
                pass
            plt.ylim(0.5,5.5)
            if exportPDF:
                plt.savefig(exportPDF+'Figs\\'+ind_var_nms[0]+'_'+dep_var_nm+'_L_CI'+'.pdf')
            plt.show()

        else:
            print(
                "Plots of linear model is not supported for several independent variables."
            )

    if code == "non_linear":
        
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
        
        if plotQ and (len(ind_var_nms) == 1):
        
            # Sort according to indep. var.
            Xyey_df = pd.concat((data_df[ind_var_nms], data_df[dep_var_nm], c_lb, c_mu, c_ub), axis=1)
            sXyey_df = Xyey_df.sort_values(by=ind_var_nms)
        
            plt.figure()
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[dep_var_nm].values,
                "o",
                color="g",
                label="true: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.5].values,
                "-",
                color="b",
                label="mean: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.05].values,
                "--",
                color="r",
                label="lb: " + dep_var_nm,
            )
            plt.plot(
                sXyey_df[ind_var_nms].values,
                sXyey_df[0.95].values,
                "--",
                color="r",
                label="ub: " + dep_var_nm,
            )
            plt.grid()
            plt.xlabel(ind_var_nms[0])
            plt.ylabel(dep_var_nm)
            plt.legend()
            plt.savefig('Figs\\'+ind_var_nms[0]+'_'+dep_var_nm+'_NL_CI'+'_'+str(num_of_points)+'pts.pdf')
            plt.show()
        
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

    print ('Get expl var started ...')    

    # Get dfs   
    Y_df = data_df[true_var_nm]
    eY_df = model_est_df[est_var_nm]
    N = Y_df.shape[0]
    
    # Summs of squares
    SS_tot = ((Y_df - Y_df.mean()) ** 2).sum() ## (N - 1) * Y.var()  # Total summs of squares
    SS_R = ((Y_df - eY_df) ** 2).sum() # (N - 1) * (Y - eY).var()  # Residual summs of squares


    # Coefficient of determination - full set
    # print (r2_score(Y, eY))
    R2 = (SS_tot - SS_R) / SS_tot

    # Stat test for H0 = [R2 == 0]
    F = (R2 / k) / ((1 - R2) / (N - k - 1))

    p_val = scs.f.sf(F, k, N - k - 1)
    
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
        F_mf = (R2_mf / k) / ((1 - R2_mf) / (N - k - 1))
        p_mf_val = scs.f.sf(F_mf, k, N - k - 1)
    else:
        R2_mf = 0
        p_mf_val = 1.0

    return R2, R2_mf, SS_tot, SS_R, p_val, p_mf_val

#Y_np = np.array(data_df[true_var_nm])
#eY_np = np.array(model_est_df[est_var_nm])
#N = Y_np.shape[0]

# Summs of squares
#SS_tot = ((Y_np - Y_np.mean()) ** 2).sum() ## (N - 1) * Y.var()  # Total summs of squares
#SS_R = ((Y_np - eY_np) ** 2).sum() # (N - 1) * (Y - eY).var()  # Residual summs of squares