from bd2dst3pi.locations import loc
from functions import list_into_string, create_directory

import matplotlib.pyplot as plt

import numpy as np

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

import os.path as op
from os import makedirs

import pickle

# Parameters of the plot
from matplotlib import rc, rcParams, use
#rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=False)
rcParams['axes.unicode_minus'] = False
use('Agg') #no plot.show() --> no display needed

#################################################################################################
###################################### Plotting function ########################################
################################################################################################# 


# it's an adaptation of hist_frame of  pandas.plotting._core
def signal_background(data1, data2, column=None,range_column=None, grid=True,
                      xlabelsize=None, ylabelsize=None,
                      sharex=False,
                      sharey=False, figsize=None,
                      layout=None, bins=40,name_file=None,
                      name_folder=None):
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
    rc('text', usetex=False)
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
        range_column = [[None,None] for i in range(len(columns))]
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
        ax.hist(data1[col].dropna().values, # .dropna() for removing missing values.
                bins=bins, range=(low,high), label = "background",density = True, **kwds)
        ax.hist(data2[col].dropna().values,
                bins=bins, range=(low,high),label = "signal", density = True, **kwds)
        ax.set_title(col, fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(grid)
        ax.legend(fontsize = 20)
    i+=1
    while i<len(_axes):
        ax = _axes[i]
        ax.axis('off')
        i+=1

    #fig.subplots_adjust(wspace=0.3, hspace=0.7)
    if name_file is None:
        name_file = list_into_string(column)
    
    directory = create_directory(f'{loc.PLOTS}/BDT/',name_folder)
    plt.savefig(op.join(directory,f"1D_hist_{name_file}.pdf"))
    
    fig.savefig(f"{loc.PLOTS}/BDT/")
    
    plt.close()
    return axes

def correlations(data, name_file=None,name_folder=None, **kwds):
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

    title = "Correlations"
    if name_file is not None:
        title += f" - {name_file}"
    ax1.set_title(title)

    labels = corrmat.columns.values # get the list of labels
    # shift location of ticks to center of the bins
    ax1.set_xticks(np.arange(len(labels))+0.5, minor=False) 
    ax1.set_yticks(np.arange(len(labels))+0.5, minor=False)
    ax1.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax1.set_yticklabels(labels, minor=False)
        
    plt.tight_layout()
    
    
    if name_file is None:
        name_file = list_into_string(column)
    
    directory = create_directory(f'{loc.PLOTS}/BDT/',name_folder)
    plt.savefig(op.join(directory,f"corr_matrix_{name_file}.pdf"))
    plt.close()

#################################################################################################
######################################## BDT training ###########################################
################################################################################################# 

## DATA PROCESSING ------------------------------------------------------
def concatenate(dfa_tot):
    """
    @dfa_tot    :: dictionnary of dataframes, with
                            * 'MC'      : pandas data frame of the MC data
                            * 'ws_strip': pandas data frame of the background
    @returns    ::
        - X  : numpy array with 'MC' and 'ws_strip' data concatenated
        - y  : new variable: numpy array with 1 for the MC events, and 0 for background events
        - df : pandas dataframe with 'MC' and 'ws_strip' concatenated and with the new variable 'y'
    """
    # Concatenated data
    X = np.concatenate((dfa_tot['MC'], dfa_tot['ws_strip']))
    # List of 1 for signal and 0 for background (mask)
    y = np.concatenate((np.ones(dfa_tot['MC'].shape[0]),
                        np.zeros(dfa_tot['ws_strip'].shape[0])))

    # Concatened dataframe of signal + background, with a new variable y:
    # y = 0 if background
    # y = 1 if signal
    df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                      columns=list(dfa_tot['MC'].columns)+['y'])
    return X, y, df

def bg_sig(y):
    """Return the mask to get the background and the signal (in this order)"""
    return (y<0.5),(y>0.5)

def BDT(X,y):
    """ Train the BDT and return the result
    
    @X       :: numpy ndarray,  with signal and background concatenated,
                The columns of X correspond to the variable the BDT will be trained with
    @y       :: numpy array, 1 if the concatened event is signal, 0 if it is background
    
    @returns ::
       - X_train and y_train: numpy ndarray and numpy array, which the BDT was trained with
       - X_test  and y_test : numpy ndarray and numpy array, left data that might be used for testing
       - bdt                : trained BDT
    """
    
    
    # Separate train/test data
    X_train,X_test, y_train,y_test = train_test_split(X, y,test_size=0.5)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    ## Define the Decision Tree
    dt = DecisionTreeClassifier(max_depth=3,
                                min_samples_leaf=0.05) # The minimum number of samples required to be at a leaf node
    # here, since it's a float, it is expressed in fraction of len(X_train)
    # We need min_samples_leaf samples before deciding to create a new leaf

    # Define the BDT
    bdt = AdaBoostClassifier(dt,
                             algorithm='SAMME',
                             n_estimators=800, # Number of trees 
                             learning_rate=0.1, # before, 0.5
                            ) # Learning rate shrinks the contribution of each tree by alpha
#     bdt = GradientBoostingClassifier(n_estimators=1000, max_depth=1, learning_rate=0.1, min_samples_split=2,verbose=1)
    ## Learning (fit)
    bdt.fit(X_train, y_train,sample_weight=weights)
    
    return X_train, y_train, X_test, y_test, bdt

#################################################################################################
################################### Analysis BDT training #######################################
################################################################################################# 

def classification_report_print(X_test, y_test, bdt,name_BDT=""):
    """ 
    Test the bdt training with the testing sample.
    Print and save the report in {loc.TABLES}/BDT/classification_report{name_BDT}.txt
    
    @X_test        : numpy ndarray, signal and background concatenated, testing sample
    @y_test        : numpy array, signal and background concatenated, testing sample
                        0 if the events is background, 1 if it is signal
    @bdt           : trained BDT
    @name_BDT      : str, name of the BDT, used for the name of the saved .txt file
    """
    
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
    if name_BDT != "":
        name_BDT = '_' + name_BDT
    with open(f"{loc.TABLES}/BDT/classification_report{name_BDT}.txt", 'w') as f:
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
    
    roc_auc = auc(fpr, tpr)

    ## Plot the results -----
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver operating characteristic'
    if name_BDT is not None:
        title += f" - {name_BDT}"
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    
    ## Save the results -----
    if name_BDT != "":
        name_BDT = '_' + name_BDT
    directory = create_directory(f'{loc.PLOTS}/BDT/',name_folder)
    plt.savefig(op.join(directory,f"ROC{name_BDT}.pdf"))
    plt.close()

def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30, name_BDT="",name_folder=None):
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
    ## decisions = [d(X_train_background), d(X_train_signal),d(X_test_background), d(X_test_signal)]
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2] # [background, signal]
    
    
    ## Range of the full plot
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    
    ## Plot for the train data the stepfilled histogram of background (y<0.5) and signal (y>0.5) 
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
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
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    # Compute and rescale the error
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.grid()
    
    
    if name_BDT != "":
        name_BDT = '_' + name_BDT
    directory = create_directory(f'{loc.PLOTS}/BDT/',name_folder)
    plt.savefig(op.join(directory,f"overtraining{name_BDT}.pdf"))
    plt.close()

from root_numpy import array2root

def apply_BDT(df_tot, df_train, bdt,name_BDT=""):
    """ 
    Apply the BDT to the real data in df_tot['data_strip']
    Add the BDT output as a new variable in df_tot['data_strip']
    Save df_tot['data_strip'] in a root file {loc.OUT}/root/data_strip_{name_BDT}.root (branch 'DecayTreeTuple/DecayTree')
    In addition, save the BDT output in a separated root file {loc.OUT}/tmp/BDT_{name_BDT}.root (branch 'BDT')
    Also save the BDT in a pickle file {loc.OUT}/pickle/bdt_{name_BDT}.pickle
    
    @df_tot        :: dictionnary  of pandas dataframe, whose one of the key is  'data_strip'. 
                            df_tot['data_strip'] is the dataframe that contains the real data.
    @bdt           :: trained BDT                
    @name_BDT      :: str, name of the BDT, used for the name of the saved files
    """
    
    # Apply the BDT to the 
    df_tot['data_strip']['BDT'] = bdt.decision_function(df_train['data_strip'])
    if name_BDT != "":
        name_BDT = '_' + name_BDT
    
    df = pd.DataFrame()
    df['BDT'] = df_tot['data_strip']['BDT']
    df.to_root(loc.OUT + f"tmp/BDT{name_BDT}.root",key = 'BDT')
    df_tot['data_strip'].to_root(loc.OUT + f"root/data_strip{name_BDT}.root",key = 'DecayTreeTuple/DecayTree')

    with open(loc.OUT + "pickle/bdt"+name_BDT+".pickle","wb") as f:
        pickle.dump(bdt,f)
