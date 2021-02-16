from root_pandas import read_root
from variables import *

import pickle

from bd2dst3pi.locations import loc
from bd2dst3pi.definitions import years as all_years
from bd2dst3pi.definitions import magnets as all_magnets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.series import Series

from matplotlib.colors import LogNorm #matplotlib.colors.LogNorm()
import matplotlib.gridspec as gridspec

import os.path as op
from os import makedirs



#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False

#################################################################################################
######################################## DATA function ##########################################
################################################################################################# 

def try_makedirs(directory):
    """If they don't not exist, create all the directories
    
    @directory   :: str, path
    
    """
    try:
        makedirs(directory)
    except OSError:
        pass


def list_included(L1,L2):
    ''' Return True if L1 included in L2
    @L1 :: list
    @L2 :: list
    
    @returns :: True if L1 included in L2, else False
    '''
    for l in L1:
        if l not in L2:
            return False
    return True

def apply_cut_DeltaM(df):
    """Cut on DeltaM: 143 < DeltaM < 148
    
    @df      :: pandas dataframe
    
    @returns :: df cut on DeltaM
    
    """
    n_events = len(df)
    df["Delta_M"] = df["Dst_M"] - df["D0_M"]
    df = df.query("Delta_M > 143. and Delta_M < 148.")
    n_cut_events = len(df)
    print(f"cut on DeltaM has removed {n_events - n_cut_events} over {n_events} events")
    return df

def apply_cut_tau_Ds(df, mean=1969., size=50):
    """Cut on df:  abs(tau_M-{mean})<={size}
    
    @df      :: pandas dataframe with the variable tau_M
    @mean    :: float
    @size    :: float
    @returns :: df cut on tau_M
    """
    assert 'tau_M' in df
    
    n_events = len(df)
    df = df.query(f"abs(tau_M-{mean})<={size}")
    n_cut_events = len(df)
    print(f"cut on tau_M has removed {n_events - n_cut_events} over {n_events} events")
    return df


def apply_cut_PIDK(df, cut=4):
    """
    Cut out the events = tau_pion_ID > {cut} if tau_pion and Dst_PID has an opposite charge
    
    @df      :: dataframe with the following variables:
                'tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID',
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                'Dst_ID'
    @cut     :: float, cut that will be applied on the PIKD variables
    
    @returns :: cut dataframe
    
    NB: 
    - tau_pion1_ID is observed to have always the same sign as Dst_ID. 
    - tau_pion0_ID and tau_pion2_ID are observed to have an opposite sign to Dst_ID. We did not use that
    in this function though.
    We did not use that in this function though.
    
    """
    n_events = len(df)
    
    df_cut = df
    for i in range(3): # i = 0, 1, 2
        df_cut = df_cut.query(f"(tau_pion{i}_PIDK < {cut}) | (Dst_ID*tau_pion{i}_ID)>0")
    
    n_cut_events = len(df_cut)
    
    print(f"cut on DeltaM has removed {n_events - n_cut_events} over {n_events} events")
    return df_cut

def apply_cut_allPIDK(df, cut=4):
    """
    Cut out the events = tau_pion_ID > {cut}
    
    @df      :: dataframe with the following variables:
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
    @cut     :: float, cut that will be applied on the PIKD variables
    
    @returns :: cut dataframe    
    """
    n_events = len(df)
    
    df_cut = df
    for i in range(3): # i = 0, 1, 2
        df_cut = df_cut.query(f"(tau_pion{i}_PIDK < {cut})")
    
    n_cut_events = len(df_cut)
    print(f"cut on all PIDKs have removed {n_events - n_cut_events} over {n_events} events")
    return df_cut

def add_flight_distance_tau(df):
    """
    Create new columns in df
    - tau_flight_{axis}, tau_flight_{axis}err and tau_flight_{axis}sig for the 3 axes 'X', 'Y' and 'Z'
    - tau_flight, tau_flight_err and tau_flight_err, a combination for the 3 axes
    
    
    @df    : pandas dataframe whose columns are the variables
                Needed variables in df
                    - B0_ENDVERTEX_X, B0_ENDVERTEX_Y, B0_ENDVERTEX_Z
                    - tau_ENDVERTEX_X, tau_ENDVERTEX_Y, tau_ENDVERTEX_Z
                    - tau_ENDVERTEX_XERR, tau_ENDVERTEX_YERR, tau_ENDVERTEX_ZERR
    @returns : df with those new variables
    """
    
    for axis in 'X','Y','Z':
        variable_flight = f"tau_flight_{axis}"
        df[f"tau_flight_{axis}"] = df[f"tau_ENDVERTEX_{axis}"] - df[f"B0_ENDVERTEX_{axis}"]
        df[f"tau_flight_{axis}err"] = np.sqrt(df[f"tau_ENDVERTEX_{axis}ERR"]**2 + df[f"B0_ENDVERTEX_{axis}ERR"]**2)
        df[f"tau_flight_{axis}sig"] = df[f"tau_flight_{axis}"] / df[f"tau_flight_{axis}err"]
    
    df[f"tau_flight"] = np.sqrt(df[f"tau_flight_X"]**2+df[f"tau_flight_Y"]**2+df[f"tau_flight_Z"]**2)
    
    df[f"tau_flight_err"] = df[f"tau_flight_X"]*df[f"tau_flight_Xerr"]
    df[f"tau_flight_err"] += df[f"tau_flight_Y"]*df[f"tau_flight_Yerr"]
    df[f"tau_flight_err"] += df[f"tau_flight_Z"]*df[f"tau_flight_Zerr"]
    df[f"tau_flight_err"] /= df[f"tau_flight"]
    
    df[f"tau_flight_sig"]=df[f"tau_flight"]/df[f"tau_flight_err"]
    return df

def load_dataframe(path, tree_name, columns, method='read_root'):
    """ load dataframe from a root file
    
    @path      :: str, location of the root file
    @tree_name :: str, name of the tree
    @columns   :: columns of the root files that you want to load
    @method    :: used method to load the data, 'read_root' or 'uproot'
    
    @returns   :: loaded pandas dataframe
    """
    print(path)
    if method == 'read_root':
        return read_root(path, tree_name, columns=columns)
    elif method == 'uproot':
        file = uproot4.open(path)[tree_name]
        df =file.arrays(vars, library = "pd")
        del file
        return df
    
def dump_pickle(data, name_data):
    """ Save the data in a pickle file in {loc.OUT}/pickle/
    @data      :: element to be saved (can be a list)
    @name_data :: str, name of the pickle file
    """
    directory = f'{loc.OUT}/pickle/{name_data}.pickle'
    with open(directory, 'wb') as f:
        pickle.dump(data,f)
    print(f"Pickle file saved in {directory}")
    
def retrieve_pickle(name_data):
    """ Return the data that is in a pickle file in {loc.OUT}/pickle/
    @name_data :: str, name of the pickle file
    
    @returns   :: data saved in the pickle file named {name_data}.pickle
    """
    with open(f'{loc.OUT}/pickle/{name_data}.pickle','br') as file:
        data = pickle.load(file)
    return data
    
def load_data(years=None, magnets=None, type_data='data', vars=None, method='read_root', 
              name_BDT='adaboost_0.8_without_P_cutDeltaM', cut_DeltaM=False, cut_PIDK=None,
             cut_tau_Ds=False, cut_BDT=None):
    """ return the pandas dataframe with the desired data
    
    @years      :: list of the wanted years
    @magnets    :: list of the wanted magnets
    @type_data  :: desired data: 'MC', 'data', 'data_strip' or 'ws_strip' or 'data_KPiPi'
    @vars       :: desired variables
    @method     :: method to retrieve the data ('read_root' or 'uproot')
                        read_root is faster
    @name_BDT   :: if BDT is one of the variable to retrieve (i.e., BDT in vars)
                    this is the str that indicates in which root file 
                    the 'BDT' variable is:
                        '{loc.OUT}/tmp/BDT_{name_BDT}.root'
    @cut_deltaM :: if true (or if BDT is in vars), perform a cut on DeltaM
                        143 < DeltaM < 148
    @cut_PIDK   :: if 'PID', cut out the events with tau_pion_ID < 4 if tau_pion and Dst_PID has an opposite charge
                   if 'ALL', cut out all the events with tau_pion_ID < 4
    @cut_BDT    :: str or float, add '_BDT{cut_BDT}' in the name of file (only for type_data = 'common' and for already saved files)
                            at this stage, this function does not cut on the BDT variable.
    
    @returns    :: df with the desired variables for all specified the years and magnets
    """
    text_cut_BDT = "" if cut_BDT is None else f'_BDT{cut_BDT}'
    
    only_one_file = False # True if we retrieve only one root file, independently of the years/magnets
    retrieve_saved = False
    
    
    if 'BDT' in vars:
        cut_DeltaM = True
    if 'sWeight' in vars:
        cut_tau_Ds = True
        
    tree_name = "DecayTree"
    # MC data -------------------------------------------
    if type_data == 'MC':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = '_Sim09e-ReDecay01.root'
    
    # Clean data ----------------------------------------
    elif type_data == 'data':
        path = f"{loc.DATA}/data_90000000"
        ext = '.root'
    
    # MC data -------------------------------------------
    elif type_data == 'MCc':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = '_Sim09c-ReDecay01.root'
    
    elif type_data == 'MCe':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = '_Sim09e-ReDecay01.root'
    
    # new data strip ------------------------------------
    elif type_data == 'common':
        variables_saved = ['B0_M','tau_M', 'BDT', 'sWeight']
        
        
        if list_included(vars, ['B0_M', 'tau_M', 'BDT']) and cut_DeltaM and magnets == all_magnets and years == all_years and cut_PIDK==None and name_BDT == 'adaboost_0.8_without_P_cutDeltaM' and not cut_tau_Ds: 
            only_one_file = True
            retrieve_saved = True
            complete_path = f"{loc.OUT}root/common/all_common{text_cut_BDT}.root"
            tree_name = 'DecayTreeTuple/DecayTree'
        elif list_included(vars, variables_saved) and cut_DeltaM and magnets == all_magnets and years == all_years and cut_PIDK==None and name_BDT == 'adaboost_0.8_without_P_cutDeltaM' and cut_tau_Ds:
            only_one_file = True
            retrieve_saved = True
            complete_path = f"{loc.OUT}root/common/common_B0toDstDs{text_cut_BDT}.root"
            tree_name = 'DecayTree'
            
        else:
            path = f"{loc.COMMON}/data_90000000"
            ext = '.root'
            tree_name = "DecayTreeTuple/DecayTree"
        
    # Previous data strip -------------------------------
    elif type_data == 'data_strip_p':
        saved_variables_PIDK = ['B0_M', 'tau_M', 'BDT',
                           'tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID','Dst_ID',
                           'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK']
        saved_variables_allPIDK = ['B0_M', 'tau_M', 'BDT', 
                                   'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK']
        
        if list_included(vars, ['B0_M','tau_M']) and magnets == all_magnets and years == all_years and cut_PIDK==None and cut_DeltaM:
            only_one_file = True
            retrieve_saved = True
            complete_path = f"{loc.OUT}root/data_strip_p/all_data_strip.root"
            tree_name = 'all_data_strip_cutDeltaM'
            
        elif list_included(vars, saved_variables_PIDK) and magnets == all_magnets and years == all_years and cut_PIDK=='PID' and cut_DeltaM:
            only_one_file = True
            retrieve_saved = True
            complete_path = f"{loc.OUT}root/data_strip_p/data_strip.root"
            tree_name = 'data_strip_cutDeltaM_cutPID'
            
        elif list_included(vars, saved_variables_allPIDK) and magnets == all_magnets and years == all_years and cut_PIDK=='ALL' and cut_DeltaM:
            only_one_file = True
            retrieve_saved = True
            complete_path = f"{loc.OUT}root/data_strip_p/data_strip.root"
            tree_name = 'data_strip_cutDeltaM_cutallPIDK'
            
        else:
            path = f"{loc.DATA_STRIP_p}/data_90000000"
            ext = '.root'
            tree_name = "DecayTreeTuple/DecayTree"
    
    # wrong sign data strip -------------------------------------
    elif type_data == 'ws_strip':
        path = f"{loc.DATA_WS_STRIP}/dataWS_90000000"
        ext = '.root'
        tree_name = "DecayTreeTuple/DecayTree"
    
    # Simulated wrong-sign data strip ---------------------------
    elif type_data == 'data_KPiPi':
        only_one_file = True
        complete_path = loc.DATA_KPiPi
        tree_name = 'DecayTree'
        
    else:
        print("Possible type of data: 'MC', 'data', 'data_strip_p', 'common', 'ws_strip', 'data_KPiPi")
    
    mode_BDT = ('BDT' in vars)
    mode_sWeight = ('sWeight' in vars)
    
    if not retrieve_saved:
        if mode_BDT:
            vars.remove('BDT')
        if cut_DeltaM:
            vars.append('Dst_M')
            vars.append('D0_M')

        if cut_PIDK == 'PID':
            vars += ['tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID',
                    'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                    'Dst_ID']
        elif cut_PIDK == 'ALL':
            vars += ['tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK']
        if mode_sWeight:
            vars.remove('sWeight')
    
    
    
    
    dfr = {}
    dfr_tot = pd.DataFrame()
    
    if only_one_file:
        dfr_tot = load_dataframe(complete_path, tree_name, vars, method=method)
    
    else:
        for y in years:
            for m in magnets:
                complete_path = f"{path}_{y}_{m}{ext}"
                dfr[f"{y}_{m}"] = load_dataframe(complete_path, tree_name, vars, method=method)
                dfr_tot = dfr_tot.append(dfr[f"{y}_{m}"])
        if cut_DeltaM:
            #print('cut on Delta_M')
            dfr_tot = apply_cut_DeltaM(dfr_tot)
        
        if cut_PIDK == 'PID': # NB: cut the PID after adding the BDT branch :)
            #print('cut on PIDK')
            dfr_tot = apply_cut_PIDK(dfr_tot)
        elif cut_PIDK == 'ALL':
            #print('cut on all PIDK')
            dfr_tot = apply_cut_allPIDK(dfr_tot)
        
        if cut_tau_Ds:
            dfr_tot = apply_cut_tau_Ds(dfr_tot)
        
        if mode_BDT:
            dfr_tot['BDT'] = read_root(loc.OUT+f'tmp/{type_data}/BDT_{name_BDT}.root', 'BDT', columns=['BDT'])
        if mode_sWeight: 
            dfr_tot = dfr_tot.reset_index()
            dfr_tot['sWeight'] = read_root(loc.OUT+f'root/{type_data}/common_B0toDstDs_sWeights.root', 'sWeights', columns=['sig'])
    return dfr_tot


def save_dataframe(df, name_file, name_key, name_folder=None):
    """ save the dataframe in a .root file
    @df        :: dataframe to save
    @name_file :: name of the file that will be savec (.root file)
    @name_key  :: name of the tree where the file will be saved
    """
    path = loc.OUT + 'root/'
    path = create_directory(path,name_folder)
    path+= f"/{name_file}.root"
    print(path)
    df.to_root(path, key=name_key)
    
    
            

#################################################################################################
########################################### Saving ##############################################
################################################################################################# 

def remove_latex(text):
    if text is not None:
        return text.replace('\_','_').replace('$','')
    else:
        return None

def remove_space(text):
    if text is not None:
        return text.replace(' ','_')
    else:
        return None

def create_directory(directory,name_folder):
    ''' if name_folder is not None, created a folder in {directory}/{name_folder}
    
    @directory   :: directory where to create the folder
    @name_folder :: str, name of the folder to create
    '''
    if name_folder is not None:
        directory = op.join(directory,name_folder)
        try:
            makedirs(directory)
        except OSError:
            pass
    return directory


def save_file(fig, name_file,name_folder=None, alt_name_file=None, directory=f"{loc.PLOTS}/"):
    """ Save fig in a file, in {directory}/{name_folder}/{name_file or alt_name_file}.pdf
    @fig            :: figure to save
    @name_file      :: name of the file
    @alt_name_file  :: alternative name of the file of name_fine = None
    @name_folder    :: name of the folder where we save the file
    @directory      :: main directory where to save the plot
    
    """
    name_folder = remove_space(name_folder)
    directory = create_directory(directory,name_folder)
    
    if name_file is None:
        name_file = alt_name_file

    #Save the plot as a PDF document into our PLOTS folder (output/plots as defined in bd2dst3pi/locations.py)
    fig.savefig(op.join(directory, f"{remove_space(remove_latex(name_file))}.pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(op.join(directory, f"{remove_space(remove_latex(name_file))}.png"), dpi=600, bbox_inches="tight")
    print(op.join(directory, f"{remove_space(remove_latex(name_file))}.pdf"))

    
#################################################################################################
############################### Tool functions for plotting #####################################
################################################################################################# 


### Computation -------------------------------------------------------------------
def redefine_low_high(low, high, data):
    '''
    if low or high is not None, return global min (low) or max (high) [respectively] of all the data in data.
    
    @low    :: float
    @high   :: float
    @data   :: Series or list of Series for which we want to define the low/high value
    
    @returns:
        - low  = low  if low  is not None else min of all the data in 'data'
        - high = high if high is not None else max of all the data in 'data'
    '''
    # Transform data into a list of data (if it is not already the case)
    l_data = [data] if isinstance(data,Series) else data
        
    define_low = low is None
    define_high = high is None
    if define_low or define_high:
        if define_low: 
            low = np.inf
        if define_high:
            high = -np.inf
        for el_data in l_data:
            if define_low:
                low = min(low,el_data.min())
            if define_high:
                high = max(high,el_data.max())
    
    return low,high

def get_bin_width(low,high,n_bins):
    """return bin width"""
    return float((high-low)/n_bins)

def el_to_list(el,len_list):
    """ If el is not a list, return a list of size len_list of this element duplicated
    
    @el       :: element
    @len_list :: integer, size of the list (if el is not a list)
    
    @returns ::
          * If el is a list: el
          * Else           : list of size len_list: [el, el, ...]
    """
    if not isinstance(el,list):
        return [el for i in range(len_list)]
    else:
        return el

### Texy formatting -------------------------------------------------------------------

def flattenlist2D(L2):
    if not isinstance(L2,list):
        return L2
    elif not isinstance(L2[0],list):
        return L2
    else:
        return [el for L in L2 for el in L]

def list_into_string(L, sep='_'):
    """Transform a list into a string of elements seperated by sep
    @L    :: list
    @sep  :: str
    
    @returns: str with all the elements separated by the argument sep    
    """
    if not isinstance(L, str):
        string = ""
        for l in L:
            string += str(l)
            string += sep
        return string[:-len(sep)]
    else:
        return L

def latex_format(text):
    """Replace _ by \_ to avoid latex errors with matplotlib"""
    return text.replace('_','\_') # to avoid latex errors when I want a label like 'B0_M'
    

def redefine_format_text(text, bracket = None):
    """ Return the correct text for the label of plots    
    @text    :: str
    @bracket :: type of bracket around text
                * '('  :: parenthetis
                * '['  :: bracket
                * None :: no brackets
    
    @returns text with
                * a space before it
                * between the brackets specified by 'bracket'
    """
    
    if (text is not None) and (text != ''):
        if bracket is None:
            brackets = ['', '']
        elif bracket == '(':
            brackets = ['(', ')']
        elif bracket == '[':
            brackets = ['[', ']']
        
        text = " %s%s%s"%(brackets[0], text, brackets[1])
    else:
        text = ""
    return text

def redefine_unit(unit, show_bracket = True):
    """Return the correct text to show the units in the labels of plots
    
    @unit         :: str
    @show_bracket :: bool
    
    @returns: text with
            * a space before it
            * between the brackets specified if show_bracket is True
    """
    if show_bracket:
        bracket = '['
    else:
        bracket = None
    return redefine_format_text(unit, bracket=bracket)

## Parameters of the plot --------------------------------------------------------

def change_ymax(ax, factor=1.1, ymin_to0=True):
    """ multiple ymax of the plot by factor
    @factor    :: float
    """
    ymin, ymax = ax.get_ylim()
    if ymin_to0:
        ymin=0
    ax.set_ylim(ymin,ymax*factor)

#################################################################################################
################################ subfunctions for plotting ######################################
################################################################################################# 

### Core of the plot -------------------------------------------------------------------

def show_grid(ax, which='major'):
    """show grid
    @ax    :: axis where to show the grid
    which  :: 'major' or 'minor'
    """
    ax.grid(b=True, which=which, color='#666666', linestyle='-', alpha = 0.2)

def plot_hist_alone(ax, data, n_bins, low, high, color, mode_hist, alpha = 1, 
                    density = False, label = None, label_ncounts = False, weights=None):
    ''' 
    Plot histogram
    - If mode_hist: Points with error
    - Else: histogram with bars
    
    @ax            :: axis where to plot
    @data          :: series of data to plot
    @low           :: low limit of the distribution
    @high          :: high limit of the distribution
    @color         :: color of the distribution
    @mode_hist     :: if True, plot with bars,else, plot with points (and error bars)
    @alpha         :: transparency of the bar histogram
    @density       :: if True, divide the numbers of counts in the histogram by the total number of counts
    @label         :: label of the histogram
    @label_ncounts :: if True, show the number of counts in the histogram
    @weights       :: Weights of each element in data
    
    @returns:
            - counts  : counts in every bin
            - edges   : edges of the bins
            - centres : centres of the bins
            - err     : Poisson uncertainty on the counts
    '''
    counts,edges = np.histogram(data, range = (low,high), bins=n_bins, weights=weights)
    centres = (edges[:-1] + edges[1:])/2.    
    n_candidates = counts.sum()
    err = np.sqrt(counts)
    
    bin_width = get_bin_width(low,high,n_bins)
    
    if label_ncounts:
        if label is None:
            label = ""
        else:
            label += ": "
        label += f" {n_candidates}"
        
    if density:
        counts = counts/(n_candidates*bin_width)
        err = err/(n_candidates*bin_width)
    
    if mode_hist:
        ax.bar(centres, counts,centres[1]-centres[0], color=color, alpha=alpha, edgecolor=None, label=label)
        ax.step(centres, counts, color=color, where='mid')
    else:
        ax.errorbar(centres, counts, yerr=err, color=color, ls='', marker='.', label=label)  
    
    ax.set_xlim(low,high)
    
    return counts, edges, centres, err


### Set labels -------------------------------------------------------------------


def set_label_variable(ax, name_variable, unit_variable=None, name_data=None, fontsize=25, axis='x'):
    ''' set the label variable (name_data) [unit] for the axis specified by 'axis'
    
    @ax            :: axis where to plot
    @name_variable :: name of the variable that was plotted
    @unit_variable :: unit of the variable that was plotted
    @fontsize      :: fontsize of the label
    @axis          :: 'x' or 'y'
    '''
    unit_variable_text = redefine_unit(unit_variable)
    name_data_text = redefine_format_text(name_data, bracket = '(')
    
    label = "%s%s%s" %(name_variable,name_data_text,unit_variable_text)
    if axis == 'x':
        ax.set_xlabel(label, fontsize=fontsize)
    if axis == 'y':
        ax.set_ylabel(label, fontsize=fontsize)

def set_label_candidates_hist (ax, bin_width, pre_label, unit_variable=None, fontsize=25, axis='y'):
    ''' set the typical y-label of 1D histogram
    
    @ax            :: axis where to plot
    @bin_width     :: bin width of the histogram
    @pre_label     :: Label to put before showing the width of the bins
    @unit_variable :: unit of the variable that was plotted
    
    @fontsize      :: fontsize of the label
    @axis          :: 'x' or 'y'
    '''
    
    
    label = f"{pre_label} / ({bin_width:.1g}{redefine_unit(unit_variable, show_bracket=False)})"
    if axis == 'x':
        ax.set_xlabel(label, fontsize=fontsize)
    if axis == 'y':
        ax.set_ylabel(label, fontsize=fontsize)

        
    
def set_label_hist(ax, name_variable, unit_variable, bin_width, density=False, name_data=None, fontsize=25):
    ''' 
    Set xlabel and ylabel of a histogram
    
    @ax            :: axis where to plot
    @name_variable :: name of the variable that was plotted
    @unit_variable :: unit of the variable that was plotted
    @density       :: if True, the ylabel will be "Proportion of candidates" instead of "candidates'
    @fontsize      :: fontsize of the label
    @bin_width     :: bin width of the histogram
    '''
    
    #Set the x label
    set_label_variable(ax, name_variable, unit_variable=unit_variable, 
                       name_data=name_data, fontsize=fontsize, axis='x')
    
    pre_label = "Proportion of candidates" if density else "Candidates"
    set_label_candidates_hist (ax, bin_width, pre_label = pre_label, unit_variable=unit_variable, 
                               fontsize=25, axis='y')

def set_label_2Dhist(ax, name_variables, unit_variables, fontsize=25):
    ''' 
    Set xlabel and ylabel of a histogram
    
    @ax             :: axis where to plot
    @name_variables :: name of the variable that was plotted (list of 2 str)
    @unit_variables :: unit of the variable that was plotted (list of 2 str)
    @fontsize       :: fontsize of the label
    '''
        
    set_label_variable(ax, name_variables[0], unit_variable=unit_variables[0], fontsize=fontsize, axis='x')
    set_label_variable(ax, name_variables[1], unit_variable=unit_variables[1], fontsize=fontsize, axis='y')
    
def set_label_divided_hist(ax, name_variable, unit_variable, bin_width, names_data, fontsize=25):
    ''' 
    Set xlabel and ylabel of a 'divided' histogram
    
    @ax             :: axis where to plot
    @name_variable  :: name of the variables that were divided
    @unit_variable  :: common unit of the variables that were plotted
    @names_data     :: list of the 2 names of the data (whose a common variable was divided)
    @fontsize       :: fontsize of the label
    @bin_width      :: bin width of the histogram
    '''
    
    #Set the x label
    set_label_variable(ax, name_variable, unit_variable=unit_variable, 
                       name_data=None, fontsize=fontsize, axis='x')
        
    pre_label = ("candidates[%s] / candidates[%s] \n")%(names_data[0],names_data[1])
    
    set_label_candidates_hist (ax, bin_width, pre_label = pre_label, 
                               unit_variable=unit_variable, 
                               fontsize=25, axis='y')
    
    
def set_label_ticks(ax, labelsize=20):
    """Set label ticks to size given by labelsize"""
    ax.tick_params(axis='both', which='both', labelsize=20)

def add_text(text1,text2, sep = ' ', default=None):
    """ concatenate 2 texts with sep between them, unless one of them is None
    
    @text1       :: str
    @text2       :: str, text to add
    @sep         :: str, separator between text and text_to_add
    @default     :: str, default value to return if both text1 and text2 are None
    
    @returns     :: {text1}_{text2} if they are both not None
                        else, return text1, or text2, 
                        or the default value is they are both None
    """
    
    if text1 is None and text2 is None:
        return None
    elif text1 is None:
        return text2
    elif text2 is None:
        return text1
    else:
        return text1 + sep + text2

def fix_plot(ax, ymax=1.1, show_leg=True, fontsize_ticks=20., fontsize_leg=20.):
    """ Some fixing of the parameters (fontsize, ymax, legend)
    @ax              :: axis where to plot
    @ymax            :: float, multiplicative factor of ymax
    @show_leg        :: Bool, True if show legend
    @fontsize_ticks  :: float, fontsize of the ticks
    @fontsize_leg    :: fontsize of the legend
    """
    
    if ymax is not None:
        change_ymax(ax,ymax)
    
    set_label_ticks(ax)
    if show_leg:
        plt.legend(fontsize = fontsize_leg)
    
    

#################################################################################################
################################# Main plotting function ########################################
################################################################################################# 

def plot_hist(dfs, variable, name_variable=None, unit_variable=None, n_bins=100, mode_hist=False, 
              low=None, high=None, density=None, 
              title=None, name_data_title=False,  
              name_file=None,name_folder=None,colors=None, weights=None, save_fig=True):
    """ Save the histogram(s) of variable of the data given in dfs
    
    @dfs             :: Dictionnary {name of the dataframe : pandas dataframe, ...}
    @variable        :: Variable, in the dataframes
    @name_variable   :: Name of the variable that will be used in the labels of the plots
    @unit_variable   :: Unit of the variable
    @n_bins          :: Desired number of bins of the histogram
    @mode_hist       :: if True, plot with bars,else, plot with points (and error bars) 
    @low             :: low value of the distribution
    @high            :: high value of the distribution
    @density         :: if True, divide the numbers of counts in the histogram by the total number of counts
    @title           :: str, title of the figure
    @name_data_title :: Bool, if true, show the name of the data in the title
    @name_file       :: name of the plot that will be saved
    @name_folder     :: name of the folder where to save the plot
    @colors          :: str or list of str, colors used in the histogram of the variable of the corresponding
                            dataframe in dfs
    """
    if not isinstance(dfs,dict):
        dfs = {"":dfs}
    
    if density is None:
        density = len(dfs)>1 # if there are more than 2 histograms
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    if isinstance(dfs,dict):
        name_datas = list(dfs.keys())
    
    if name_variable is None:
        name_variable = latex_format(variable)
    
    if name_data_title:
        title = add_text(list_into_string(name_datas),title,' - ', default=None)
    
    ax.set_title(title, fontsize=25)
    
    #First loop to determine the low and high value
    low, high = redefine_low_high(low,high,[df[variable] for df in dfs.values()])
    bin_width = get_bin_width(low,high,n_bins)
    
    if colors is None:
        colors = ['r','b','g','k']
    if not isinstance(colors,list):
        colors = [colors]
    
    k_col = 0
    for name_data, df in dfs.items():
        alpha = 0.5 if len(dfs)>1 else 1
        _,_,_,_ = plot_hist_alone(ax, df[variable], n_bins, low, high, colors[k_col], mode_hist, alpha = alpha, 
                        density = density, label = name_data, label_ncounts = True, weights=weights)
        k_col += 1
              
              
    #Some plot style stuff
    set_label_hist(ax, name_variable, unit_variable, bin_width, density=density, fontsize=25)
    fix_plot(ax, ymax=1+0.1*len(name_datas), show_leg=len(dfs)>1)
    
    #Remove any space not needed around the plot
    plt.tight_layout()
    
    if save_fig:
        directory = f"{loc.PLOTS}/"    
        save_file(fig, name_file,name_folder,f'{variable}_{list_into_string(name_datas)}',f"{loc.PLOTS}/")
    return fig, ax
    
    
def plot_divide(dfs, variable, name_variable,unit_variable, n_bins=100, low=None, high=None, 
                name_data_title=False,
                name_file=None, name_folder=None, save_fig=True):
    """
    plot the (histogram of the dataframe 1 of variable)/(histogram of the dataframe 1 of variable) after normalisation
        
    @dfs             :: Dictionnary of 2 couple (key:value) 
                            {name_dataframe_1 : pandas_dataframe_1, name_dataframe_2 : pandas_dataframe_2}
    @variable        :: Variable, in the dataframes
    @name_variable   :: Name of the variable that will be used in the labels of the plots
    @unit_variable   :: Unit of the variable
    @n_bins          :: Desired number of bins of the histogram
    @low             :: low value of the distribution
    @high            :: high value of the distribution
    @name_data_title :: Bool, if true, show the name of the data in the title
    @name_file       :: name of the plot that will be saved
    @name_folder     :: name of the folder where to save the plot
    """
       
    fig, ax = plt.subplots(figsize=(8,6))
    name_datas = list(dfs.keys())
    
    
    # Compute the number of bins
    low, high = redefine_low_high(low,high,[df[variable] for df in dfs.values()])
    bin_width = get_bin_width(low,high,n_bins)
    
    # Make the histogram, and get the bin centres and error on the counts in each bin
    list_dfs = list(dfs.values())
    names_data = list(dfs.keys())
    
    counts1, bin_edges = np.histogram(list_dfs[0][variable], n_bins, range=(low,high))
    counts2, _ = np.histogram(list_dfs[1][variable], n_bins, range=(low,high))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    
    err1 = np.sqrt(counts1)
    err2 = np.sqrt(counts2)
    
    #division
    with np.errstate(divide='ignore', invalid='ignore'): 
        division = counts1*counts2.sum()/(counts2*counts1.sum()) 
    err = division*np.sqrt((err1/counts1)**2+(err2/counts2)**2)

    ax.errorbar(bin_centres, division, yerr=err, fmt='o', color='k')
    ax.plot([low,high], [1.,1.], linestyle='--', color='b',marker='')
    
    # Set lower and upper range of the x and y axes
    plt.xlim(low,high)
    ymin, ymax = plt.ylim()
    plt.ylim(0.,ymax*1.2)
    
    # Labels
    set_label_divided_hist(ax, name_variable, unit_variable, bin_width, names_data, fontsize=25)    

    # Set the font size of the axis numbers
    set_label_ticks(ax)

    
    #R emove any space not needed around the plot
    plt.tight_layout()

    plt.show()

    #Save the plot as a PDF document into our PLOTS folder (output/plots as defined in bd2dst3pi/locations.py)    
    if save_fig:
        save_file(fig, name_file,name_folder,f"{variable}_{list_into_string(name_datas,'_d_')}",f"{loc.PLOTS}/")
    
    return fig, ax


def plot_hist2d(df, variables, name_variables, unit_variables, n_bins = 100,
                low=None, high=None, 
                title=None, name_data_title=False, 
                name_file=None, name_folder=None,
                name_data=None, log_scale=False,
               save_fig=True):
    '''  Plot a 2D histogram of 2 variables.
    @df                :: dataframe (only one)
    @variables         :: list of 2 str, variables in the dataframe
    @name_variables    :: list of 2 str, names of the variables 
    @unit_variables    :: str (common unit) or list of 2 str (units of variable[0] and variable[1])
    @n_bins            :: integer or list of 2 integers
    @low               :: float or list of 2 floats ; low  value(s) of variables
    @high              :: float or list of 2 floats ; high value(s) of variables
    @title             :: str, title of the figure
    @name_data_title   :: Bool, if true, show the name of the data in the title
    
    @name_file         :: name of the plot that will be saved
    @name_folder       :: name of the folder where to save the plot
    @name_data         :: str, name of the data, this is isued to define the name of the plot,
                              in the case name_file is not defined.
    @log_scale         :: if true, the colorbar is in logscale
    '''
    
    
    ## low, high and unit_variables into a list of size 2
    low = el_to_list(low,2)
    high = el_to_list(high,2)
    
    unit_variables = el_to_list(unit_variables,2)
    
    for i in range(2):
        low[i],high[i] = redefine_low_high(low[i],high[i], df[variables[i]])
    
    ## Plotting
    fig, ax = plt.subplots(figsize=(8,6))
    
    title = add_text(name_data, title, default=None)
    
    ax.set_title(title, fontsize=25)
    
    if log_scale:
        _,_,_,h = ax.hist2d(df[variables[0]], df[variables[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins,norm=LogNorm())
    else:
        _,_,_,h = ax.hist2d(df[variables[0]], df[variables[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins)
    
    ## Label, color bar
    set_label_ticks(ax)
    set_label_2Dhist(ax, name_variables, unit_variables, fontsize=25)
    cbar = plt.colorbar(h)
    cbar.ax.tick_params(labelsize=20)
    
    ## Save the data
    if save_fig:
        directory = f"{loc.PLOTS}/"
        save_file(fig, name_file,name_folder,add_text(list_into_string(variables,'_vs_'),name_data,'_'),directory)
    
    return fig, ax
        

#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################         


def add_in_dic(key, dic):
    ''' if key is not in dic, add it with value None. In place.
    @key   :: key of dictionnary
    @dic   :: dictionnary
    
    @returns :: nothing - in place
    '''
    if key not in dic:
        dic[key] = None

def retrieve_particle_variable(variable):
    '''
    Retrieve the name of the particle and of the variable from 'variable'
    @variable    :: str, variable (for instance: 'B0_M')
    
    @returns:
        - particle : str, name of the particle ('B0', 'tau', ...), key in the dictionnary particle_names
        - variable : str, name of the variable ('P', 'M', ...)
    
    Hypothesis: the particle is in the dictionnary particle_names in variables.py
    '''

    list_particles = list(particle_names.keys())
    particle = variable
    marker = None
    marker_before = 0
    
    while particle not in list_particles and marker != marker_before:
        marker_before = marker
        cut_variable = variable[:marker]
        marker = len(cut_variable)-1-cut_variable[::-1].find('_') # find the last '_'
        particle = variable[:marker]
        
    if marker != marker_before:
        var = variable[marker+1:]
        return particle, var
    else:
        return None, None

def get_name_unit_particule_var(variable):
    """ return the name of particle, and the name and unit of the variable
    @variable   :: str, variable (for instance: 'B0_M')
    
    @returns    ::
                    - str, name of the variable (for instance, 'm($D^{*-}3\pi$)')
                    - str, unit of the variable (for instance, 'MeV/$c^2$')
    """
    particle, var = retrieve_particle_variable(variable)
    if particle is not None and var is not None:
        if particle in particle_names:
            name_particle = particle_names[particle]
        else:
            name_particle = None
        if var in variables_params:
            name_var = variables_params[var]['name']
            unit_var = variables_params[var]['unit']
        else:
            name_var = None
            unit_var = None

        if name_var is None:
            name_var = latex_format(var)
        if name_particle is None:
            name_variable = var
        else:
            name_variable = f"{name_var}({name_particle})"
    else:
        name_variable = variable
        unit_var = None
    return name_variable, unit_var 

def get_name_file_title_BDT(name_file, title, cut_BDT, variable, name_data):
    """ Return the new name_file and title given the cut on the BDT 
    @name_file   :: str, initial name of the file
    @title       :: str, initial title
    @cut_BDT     :: float, cut on the BDT (we keep BDT > {cut_BDT})
    @variable    :: float, variable (e.g., 'B0_M')
    @name_data   :: name of the plotted data
    
    @returns     ::
                    - new name_file
                    - new title
    """
    if name_file is None:
        name_file = add_text(variable,name_data,'_')
    
    # Title with BDT
    if cut_BDT is not None:
        title = add_text(title, f"BDT $>$ {cut_BDT}", ' - ')
        name_file = add_text(name_file, f'BDT{cut_BDT}')
    
    return name_file, title

   

def plot_hist_particle(dfs, variable, cut_BDT=None, **kwargs):
    """ 
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    
    Then, plot histogram with plot_hist
    
    @dfs      :: Dictionnary of 2 couple (key:value) 
                            {name_dataframe_1 : pandas_dataframe_1, name_dataframe_2 : pandas_dataframe_2}
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variable, name_var, unit_var
    """
    
    ## Retrieve particle name, and variable name and unit.
#     particle, var = retrieve_particle_variable(variable)
    
#     name_var = variables_params[var]['name']
#     unit_var = variables_params[var]['unit']
#     name_particle = particle_names[particle]
    
    name_variable, unit_var = get_name_unit_particule_var(variable)
    name_datas = list_into_string(list(dfs.keys()))
    
    add_in_dic('name_file', kwargs)
    add_in_dic('title', kwargs)
    kwargs['name_file'], kwargs['title'] = get_name_file_title_BDT(kwargs['name_file'], kwargs['title'], 
                                                                   cut_BDT, variable, name_datas)

    # Name of the folder = list of the names of the data
    add_in_dic('name_folder', kwargs)
    if kwargs['name_folder'] is None:
        name_folder = name_datas
    
    return plot_hist(dfs, variable, name_variable, unit_var, **kwargs)


def plot_divide_particle(dfs, variable, **kwargs): 
    """
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    
    Then, plot with plot_divide.
    
    @dfs      :: Dictionnary of 2 couple (key:value) 
                            {name_dataframe_1 : pandas_dataframe_1, name_dataframe_2 : pandas_dataframe_2}
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variable, name_var, unit_var
    """
    name_variable, unit_var = get_name_unit_particule_var(variable)
    
    add_in_dic('name_folder', kwargs)
    if kwargs['name_folder'] is None:
        kwargs['name_folder'] = list_into_string(list(dfs.keys()))
        
    return plot_divide(dfs, variable, name_variable, unit_var, **kwargs)
    
    
def plot_hist2d_particle(df, variables, **kwargs):
    """
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    
    Then, plot 2d histogram with plot_hist2d.
    
    @df       :: pandas dataframe
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variables, name_vars, unit_vars)
    """
    name_variables = [None, None]
    unit_vars      = [None, None]
    for i in range(2):
        name_variables[i], unit_vars[i] = get_name_unit_particule_var(variables[i])
    
    add_in_dic('name_folder', kwargs)
    add_in_dic('name_data', kwargs)
    
    if kwargs['name_folder'] is None :
        kwargs['name_folder'] = kwargs['name_data']
    
    return plot_hist2d(df, variables, name_variables=name_variables, unit_variables=unit_vars, **kwargs)

    