"""
Anthony Correia
02/01/21
- Plot the distributions of the chosen variables in signal and background (super-imposed)
- Prepare the signal and background sample (merge them, create a 'y' variable for learning, ...)
- Train the BDT with the specified classifier (adaboost or gradientboosting)
- Plot the result of the tests (ROC curve, overtraining check with KS test)
- Apply the BDT to the data and save the result
"""


from bd2dst3pi.locations import loc
from plot.tool import list_into_string, remove_latex

import matplotlib.pyplot as plt

import numpy as np

import plot.tool as pt
import load_save_data as l
import plot.histogram as h


import pandas as pd
import pandas.core.common as com
from pandas.plotting import scatter_matrix
from pandas import Index

# sklearn
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp
 

import os.path as op
from os import makedirs

import pickle

# Parameters of the plot
from matplotlib import rc, rcParams, use
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False
#use('Agg') #no plot.show() --> no display needed

#################################################################################################
###################################### Plotting function ########################################
################################################################################################# 


# it's an adaptation of hist_frame of  pandas.plotting._core
def signal_background(data1, data2, column=None,range_column=None, grid=True,
                      xlabelsize=None, ylabelsize=None,
                      sharex=False,
                      sharey=False, figsize=None,
                      layout=None, n_bins=40,name_file=None,
                      name_folder=None, colors=['red', 'green'], **kwds):
    """Draw histogram of the DataFrame's series comparing the distribution
    in `data1` to `data2`.
    
    @data1        :: DataFrame
    @data2        :: DataFrame
    @column       :: string or sequence
                    If passed, will be used to limit data to a subset of columns
    @grid         :: boolean, default True
                    Whether to show axis grid lines
    @xlabelsize   :: int, default None, if specified changes the x-axis label size
    @ylabelsize   :: int, default None, if specified changes the y-axis label size
    @ax           :: matplotlib axes object, default None
    @sharex       :: bool, if True, the X axis will be shared amongst all subplots.
    @sharey       :: bool, if True, the Y axis will be shared amongst all subplots.
    @figsize      :: tuple, the size of the figure to create in inches by default
    @bins         :: integer, default 10, number of histogram bins to be used
    @name_file    :: name of the saved file
    @name_folder  :: name of the folder where to save the plot
    @kwds         :: other plotting keyword arguments, to be passed to hist function
    """
    if 'alpha' not in kwds:
        kwds['alpha'] = 0.5

    
    if column is not None:
        # column is not a list, convert it into a list.
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]
        
    data1 = data1._get_numeric_data() # select only numbers
    data2 = data2._get_numeric_data() # seject only numbers
    naxes = len(data1.columns) # number of axes = number of available columns
    
    max_nrows = 4
    # subplots
    fig, axes = plt.subplots(nrows=min(naxes,max_nrows), ncols = 1+naxes//max_nrows , squeeze=False,
                                   sharex=sharex,
                                   sharey=sharey,
                                   figsize=figsize)
    
    _axes = axes.flat

    if range_column is None:
        range_column = [[None,None] for i in range(len(column))]
    for i, col in enumerate(data1.columns): # data.columns = the column labels of the DataFrame.
        # col = name of the column/variable
        ax = _axes[i]
        
        if range_column[i] is None:
            range_column[i] = [None,None]
        if range_column[i][0] is None:
            low = min(data1[col].min(), data2[col].min())
        else:
            low = range_column[i][0]
        if range_column[i][1] is None:
            high = max(data1[col].max(), data2[col].max())
        else:
            high = range_column[i][1]
        
        low, high = pt.redefine_low_high(range_column[i][0], range_column[i][1], [data1[col], data2[col]]) 
        _,_,_,_ = h.plot_hist_alone(ax, data1[col].dropna().values, n_bins, low, high, colors[1], mode_hist=True, alpha=0.5, 
                        density=True, label='background', label_ncounts = True)
        _,_,_,_ = h.plot_hist_alone(ax, data2[col].dropna().values, n_bins, low, high, colors[0], mode_hist=True, alpha=0.5, 
                        density=True, label='signal', label_ncounts = True)
        
        bin_width=(high - low)/n_bins
        name_variable, unit_variable = pt.get_name_unit_particule_var(col)
        h.set_label_hist(ax, name_variable, unit_variable, bin_width=bin_width, density=False, fontsize=20)
        pt.fix_plot(ax, ymax=1+0.3, show_leg=True, fontsize_ticks=15., fontsize_leg=20.)
        pt.show_grid(ax, which='major')
                 
    i+=1
    while i<len(_axes):
        ax = _axes[i]
        ax.axis('off')
        i+=1

    #fig.subplots_adjust(wspace=0.3, hspace=0.7)
    if name_file is None:
        name_file = list_into_string(column)
    
    plt.tight_layout()
    pt.save_file(fig, f"1D_hist_{name_file}", name_folder= f'BDT/{name_folder}')
    
    return fig, axes

def correlations(data, name_file=None,name_folder=None, title=None, **kwds):
    """Calculate pairwise correlation between features of the dataframe data
    
    @data         :: DataFrame
    @name_file    :: name of the saved file
    @name_folder  :: name of the folder where to save the plot
    @kwds         :: other plotting keyword arguments, to be passed to DataFrame.corr()
    """

    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds) # correlation

    fig, ax1 = plt.subplots(ncols=1, figsize=(12,10)) # 1 plot
    
    opts = {'cmap': plt.get_cmap("RdBu"), # red blue color mode
            'vmin': -1, 'vmax': +1} # correlation between -1 and 1
    heatmap1 = ax1.pcolor(corrmat, **opts) # create a pseudo color plot
    plt.colorbar(heatmap1, ax=ax1) # color bar

    title = pt.add_text("Correlations", title, ' - ')
    ax1.set_title(title)

    labels = list(corrmat.columns.values) # get the list of labels
    for i, label in enumerate(labels):
        name_variable,_ = pt.get_name_unit_particule_var(label)
        labels[i] = name_variable
    # shift location of ticks to center of the bins
    ax1.set_xticks(np.arange(len(labels))+0.5, minor=False) 
    ax1.set_yticks(np.arange(len(labels))+0.5, minor=False)
    ax1.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax1.set_yticklabels(labels, minor=False)
        
    plt.tight_layout()
    
    
    if name_file is None:
        name_file = list_into_string(column)
        
    pt.save_file(fig, f"corr_matrix_{name_file}", name_folder= f'BDT/{name_folder}')
    
    return fig, ax1
    

#################################################################################################
######################################## BDT training ###########################################
################################################################################################# 

## DATA PROCESSING ------------------------------------------------------

    
def concatenate(dfa_tot_sig, dfa_tot_bkg):
    """
    @dfa_tot    :: dictionnary of dataframes, with
                            * 'MC'      : pandas data frame of the MC data
                            * 'ws_strip': pandas data frame of the background
    @returns    ::
        - X  : numpy array with 'MC' and 'ws_strip' data concatenated
        - y  : new variable: numpy array with 1 for the MC events, and 0 for background events
        - df : pandas dataframe with 'MC' and 'ws_strip' concatenated and with the new variable 'y'
    """
    assert len(dfa_tot_sig.columns) == len(dfa_tot_bkg.columns)
    assert (dfa_tot_sig.columns == dfa_tot_bkg.columns).all()
    
    # Concatenated data
    X = np.concatenate((dfa_tot_sig, dfa_tot_bkg))
    # List of 1 for signal and 0 for background (mask)
    y = np.concatenate((np.ones(dfa_tot_sig.shape[0]),
                        np.zeros(dfa_tot_bkg.shape[0])))

    # Concatened dataframe of signal + background, with a new variable y:
    # y = 0 if background
    # y = 1 if signal
    df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                      columns=list(dfa_tot_sig.columns)+['y'])
    return X, y, df

def bg_sig(y):
    """Return the mask to get the background and the signal (in this order)"""
    return (y<0.5),(y>0.5)

def get_train_test(X, y, test_size=0.5, random_state=15):
    # Separate train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_train_test_df(df, test_size=0.5, random_state=15):
    # Separate train/test data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test


def BDT(X_train, y_train, classifier='adaboost', **hyperparams):
    """ Train the BDT and return the result
    
    @X               :: numpy ndarray,  with signal and background concatenated,
                          The columns of X correspond to the variable the BDT will be trained with
    @y               :: numpy array, 1 if the concatened event is signal, 0 if it is background
    @classifier      :: str, specified the used classifier
        - 'adaboost'
        - 'gradientboosting'
        - 'xgboost' (experimental)
    @hyperparameters :: dict, used hyperparameters. default:
                                    * n_estimators = 800
                                    * learning_rate = 0.1
    @returns ::
       - X_train and y_train: numpy ndarray and numpy array, which the BDT was trained with
       - X_test  and y_test : numpy ndarray and numpy array, left data that might be used for testing
       - bdt                : trained BDT
    """
    
    
    
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    if hyperparams is None:
        hyperparams = {}
    
    l.add_in_dic('n_estimators', hyperparams, 800)
    l.add_in_dic('learning_rate', hyperparams, 0.1) # Learning rate shrinks the contribution of each tree by alpha    
    l.show_dictionnary(hyperparams, "hyperparameters")
            
    
    # Define the BDT
    if classifier == 'adaboost':
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05)
        # The minimum number of samples required to be at a leaf node
        # here, since it's a float, it is expressed in fraction of len(X_train)
        # We need min_samples_leaf samples before deciding to create a new leaf
        bdt = AdaBoostClassifier(dt, algorithm='SAMME', verbose=1, **hyperparams)
        
    elif classifier == 'gradientboosting':
        bdt = GradientBoostingClassifier(max_depth=1, min_samples_split=2, verbose=1, random_state=15, **hyperparams)
    
    elif classifier == 'xgboost': # experimental
        import xgboost as xgb
        bdt = xgb.XGBClassifier(objective="binary:logistic", random_state=15, verbose=1, learning_rate=0.1)
        
        
    ## Learning (fit)
    bdt.fit(X_train, y_train, sample_weight=weights)
    
    return bdt

#################################################################################################
################################### Analysis BDT training #######################################
################################################################################################# 

def classification_report_print(X_test, y_test, bdt, name_BDT=""):
    """ 
    Test the bdt training with the testing sample.
    Print and save the report in {loc.TABLES}/BDT/classification_report{name_BDT}.txt
    
    @X_test        : numpy ndarray, signal and background concatenated, testing sample
    @y_test        : numpy array, signal and background concatenated, testing sample
                        0 if the events is background, 1 if it is signal
    @bdt           : trained BDT
    @name_BDT      : str, name of the BDT, used for the name of the saved .txt file
    """
#     if xgboost:
#         y_predicted = xgbmodel.predict_proba(X)[:,1]
#     else:    
    y_predicted = bdt.predict(X_test)
    
    classification_report_str = classification_report(y_test, y_predicted,
                                target_names=["background", "signal"])
    
    
    print(classification_report_str)
    ROC_AUC_score = roc_auc_score(y_test, # real
                                    bdt.decision_function(X_test)) 
    # bdt.decision_function(X_test) = scores = returns a Numpy array, in which each element 
    # represents whether a predicted sample for x_test by the classifier lies to the right 
    # or left side of the Hyperplane and also how far from the HyperPlane.
    
    print("Area under ROC curve: %.4f"%(ROC_AUC_score))

    ## Write the results -----
    name_file = pt.add_text('classification_report', name_BDT, '_')

    with open(f"{loc.TABLES}/BDT/{name_file}.txt", 'w') as f:
        f.write(classification_report_str)
        f.write("Area under ROC curve: %.4f"%(ROC_AUC_score))

        

def plot_roc(X_test, y_test, bdt,name_BDT = "",name_folder = None):
    """ Plot and save the roc curve in {loc.PLOTS}/BDT/{name_folder}/ROC_{name_BDT}.pdf
    
    @X_test        :: numpy ndarray, signal and background concatenated, testing sample
    @y_test        :: numpy array, signal and background concatenated, testing sample
                        0 if the events is background, 1 if it is signal
    @bdt           :: trained BDT
    @name_BDT      :: str, name of the BDT, used for the name of the saved plot
    @name_folder   :: str, name of the folder where to save the BDT
    """
    
    ## Get the results -----
    decisions = bdt.decision_function(X_test) # result of the BDT of the test sample
    fpr, tpr, thresholds = roc_curve(y_test, decisions) # roc_curve
    # y_test: true results
    # decisions: result found by the BDT
    # fpr: Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
    # tpr: Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
    # thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
    fig, ax = plt.subplots(figsize=(8,6))
    roc_auc = auc(fpr, tpr)

    ## Plot the results -----
    ax.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    title = 'Receiver operating characteristic'
    
    ax.legend(loc="lower right", fontsize=20.)
    pt.show_grid(ax)
    pt.fix_plot(ax, ymax=1.1, show_leg=False, fontsize_ticks=20., ymin_to0=False)
    ## Save the results -----
    
    name_file = pt.add_text('ROC', name_BDT, '_')
    pt.save_file(fig, name_file, name_folder= f'BDT/{name_folder}')
    
    return fig, ax

def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30, name_BDT="", name_folder=None,
                      colors=['red', 'green']):
    """ Plot and save the overtraining plot in {loc.PLOTS}/BDT/{name_folder}/overtraining_{name_BDT}.pdf
    
    @clf           :: trained BDT
    @X_train       :: numpy ndarray, signal and background concatenated, training sample
    @y_train       :: numpy array, signal and background concatenated, training sample
                        0 if the events is background, 1 if it is signal
    @X_test        :: numpy ndarray, signal and background concatenated, testing sample
    @y_test        :: numpy array, signal and background concatenated, testing sample
                        0 if the events is background, 1 if it is signal
    @bins          :: number of bins of the plotted histograms                   
    @name_BDT      :: str, name of the BDT, used for the name of the saved plot
    @name_folder   :: str, name of the folder where to save the BDT
    """
    fig, ax = plt.subplots(figsize=(8,6))
    
    ## decisions = [d(X_train_signal), d(X_train_background),d(X_test_signal), d(X_test_background)]
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2] # [signal, background]
    
    '''
    decisions[0]: train, background
    decisions[1]: train, signal
    decisions[2]: test, background
    decisions[3]: test, signal
    '''
    
    ## Range of the full plot
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    
    ## Plot for the train data the stepfilled histogram of background (y<0.5) and signal (y>0.5) 
    ax.hist(decisions[0],
             color=colors[0], alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    ax.hist(decisions[1],
             color=colors[1], alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    ## Plot for the test data the points with uncertainty of background (y<0.5) and signal (y>0.5) 
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    # Compute and rescale the error
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.errorbar(center, hist, yerr=err, fmt='o', c=colors[0], label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    # Compute and rescale the error
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    ax.errorbar(center, hist, yerr=err, fmt='o', c=colors[1], label='B (test)')

    ax.set_xlabel("BDT output", fontsize=25.)
    ax.set_ylabel("Arbitrary units", fontsize=25.)
    ax.legend(loc='best', fontsize=20.)
    pt.show_grid(ax)
    
    pt.fix_plot(ax, ymax=1.1, show_leg=False, fontsize_ticks=20., ymin_to0=False)
    
    name_file = pt.add_text('overtraining', name_BDT, '_')
    pt.save_file(fig, name_file, name_folder= f'BDT/{name_folder}')
    
    ks_2samp_sig = ks_2samp(decisions[0],decisions[2]).statistic
    ks_2samp_bkg = ks_2samp(decisions[1],decisions[3]).statistic
    pvalue_2samp_sig = ks_2samp(decisions[0],decisions[2]).pvalue
    pvalue_2samp_bkg = ks_2samp(decisions[1],decisions[3]).pvalue
    print('Kolmogorov-Smirnov statistic')
    print(f"signal    : {ks_2samp_sig}")
    print(f"Background: {ks_2samp_bkg}")
    
    print('p-value')
    print(f"signal    : {pvalue_2samp_sig}")
    print(f"Background: {pvalue_2samp_bkg}") 
    return fig, ax, ks_2samp_sig, ks_2samp_bkg, pvalue_2samp_sig, pvalue_2samp_bkg

def apply_BDT(df_tot, df_train, bdt,name_BDT="", save_BDT=False, kind_data='common'):
    """ 
    Apply the BDT to the real data in df_tot['data_strip']
    Add the BDT output as a new variable in df_tot['data_strip']
    Save df_tot['data_strip'] in a root file {loc.OUT}/root/data_strip_{name_BDT}.root (branch 'DecayTreeTuple/DecayTree')
    In addition, save the BDT output in a separated root file {loc.OUT}/tmp/BDT_{name_BDT}.root (branch 'BDT')
    Also save the BDT in a pickle file {loc.OUT}/pickle/bdt_{name_BDT}.pickle
    
    @df_tot        :: pandas dataframe that will be saved together with the BDT output
    df_train       :: pandas dataframe with exactly the variables that have been used for the training 
    @bdt           :: trained BDT                
    @name_BDT      :: str, name of the BDT, used for the name of the saved files
    """
    
    # Apply the BDT to the 
    df_tot['BDT'] = bdt.decision_function(df_train)

    
    name_file = pt.add_text(kind_data, name_BDT, '_')
    
    df = pd.DataFrame()
    df['BDT'] = df_tot['BDT']
    #df.to_root(loc.OUT + f"tmp/BDT{name_BDT}.root",key = 'BDT')
    
    l.save_dataframe(df, 'BDT_'+name_file, 'BDT')
    l.save_dataframe(df_tot, name_file, 'DecayTree')
    
    if save_BDT:
        dump_pickle(bdt, f'bdt_{name_file}')

