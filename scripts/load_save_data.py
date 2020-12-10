from root_pandas import read_root

import pickle

from bd2dst3pi.locations import loc
from bd2dst3pi.definitions import years as all_years
from bd2dst3pi.definitions import magnets as all_magnets

import numpy as np
import pandas as pd

from os import makedirs
import os.path as op

from copy import deepcopy

import variables as v

#################################################################################################
######################################## TOOLS function #########################################
################################################################################################# 

def try_makedirs(path):
    """If they don't not exist, create all the necessary directories in the path
    
    @path   :: str, path
    
    """
    try:
        makedirs(path)
    except OSError:
        pass

def create_directory(directory,name_folder):
    ''' if name_folder is not None, created a folder in {directory}/{name_folder}
    
    @directory   :: directory where to create the folder
    @name_folder :: str, name of the folder to create
    '''
    if name_folder is not None:
        directory = op.join(directory,name_folder)
        try_makedirs(directory)
    return directory

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


def add_in_dic(key, dic, default=None):
    ''' if key is not in dic, add it with value specified by default. In place.
    @key     :: key of dictionnary
    @dic     :: dictionnary
    @default :: python object
    
    @returns :: nothing - in place
    '''
    if key not in dic:
        dic[key] = default

def show_dictionnary(dic, name_dic):
    print(f"{name_dic}:")
    for key, value in dic.items():
        print(f"{key}: {value}")
    


#################################################################################################
######################################## CUTS function ##########################################
################################################################################################# 

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


#################################################################################################
##################################### DATAFRAME function ########################################
################################################################################################# 

## ADD COLUMNS ===============================================================
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

def add_constr_Dst(df, col='B0_M', Dst_M_PDG=v.Dst_M_PDG):
    """ Add the column 'Ds_constr_{col}' in the df
    @df     :: pandas dataframe with the variablaes 'B0_M' and 'Dst_M'
    @col    :: str, name of the column of dataframe from which is computed the constrained column
    @returns:: dataframe with the supplementary column
    """
    df[f"Dst_constr_{col}"] = df[col] - df['Dst_M'] + Dst_M_PDG
    return df

## LOAD DATAFRAMES ===============================================================

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
        import uproot4 # not imported by default...
        file = uproot4.open(path)[tree_name]
        df =file.arrays(vars, library = "pd")
        del file
        return df

    
def load_saved_root(name_data, vars=None, name_folder="", tree_name=None, cut_BDT=None, method='read_root'):
    """
    @name_data  :: name of the root file
    @vars       :: desired variables
    @method     :: method to retrieve the data ('read_root' or 'uproot')
                        read_root is faster
    @cut_BDT    :: str or float, corresponding cut to load
    @cut_deltaM :: if true (or if BDT is in vars), perform a cut on DeltaM
                        143 < DeltaM < 148
    
    @returns    :: df with the desired variables
    """
    
    text_cut_BDT = "" if cut_BDT is None else f'_BDT{cut_BDT}'
    if tree_name is None:
        tree_name = v.data_tree_names[name_data]
    complete_path = f"{loc.OUT}/root/{name_folder}/{name_data}{text_cut_BDT}.root"
    return load_dataframe(complete_path, tree_name, vars, method=method)
    
    
    

def load_data(years=None, magnets=None, type_data='common', vars=None, method='read_root', 
              name_BDT='adaboost_0.8_without_P_cutDeltaM', cut_DeltaM=False, cut_PIDK=None,
             cut_tau_Ds=False):
    """ return the pandas dataframe with the desired data
    
    @years      :: list of the wanted years
    @magnets    :: list of the wanted magnets
    @type_data  :: desired data: 'MC', 'data', 'data_strip' or 'ws_strip' or 'data_KPiPi' or 'common'
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
    
    @returns    :: df with the desired variables for all specified the years and magnets
    """
    assert type_data in ['MC', 'data_strip', 'data', 'common', 'ws_strip', 'data_KPiPi']
    mode_BDT = False
    
    variables = deepcopy(vars)
    if 'BDT' in variables:
        cut_DeltaM = True
        mode_BDT = True
        
    if 'sWeight' in variables:
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
        path = f"{loc.COMMON}/data_90000000"
        ext = '.root'
        tree_name = "DecayTreeTuple/DecayTree"
        
    # Previous data strip -------------------------------
    elif type_data == 'data_strip_p':
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
    
    mode_sWeight = ('sWeight' in variables)
    
    # Remove / add some variables ------------------------------
    if mode_BDT:
            variables.remove('BDT')
    if cut_DeltaM:
        variables.append('Dst_M')
        variables.append('D0_M')
    if cut_PIDK == 'PID':
        variables += ['tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID',
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                'Dst_ID']
    elif cut_PIDK == 'ALL':
        variables += ['tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK']
    
    # Load variables -------------------------------------------    
    dfr = {}
    dfr_tot = pd.DataFrame()      
    
    for y in years:
        for m in magnets:
            complete_path = f"{path}_{y}_{m}{ext}"
            dfr[f"{y}_{m}"] = load_dataframe(complete_path, tree_name, variables, method=method)
            dfr_tot = dfr_tot.append(dfr[f"{y}_{m}"])
    
    # CUTS --------------------
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
    
    dfr_tot = dfr_tot.reset_index()
    return dfr_tot


### SAVE DATAFRAMES ==========================================================
def save_dataframe(df, name_file, name_key, name_folder=None):
    """ save the dataframe in a .root file
    @df        :: dataframe to save
    @name_file :: name of the file that will be savec (.root file)
    @name_key  :: name of the tree where the file will be saved
    """
    path = loc.OUT + 'root/'
    path = create_directory(path, name_folder)
    path+= f"/{name_file}.root"
    print(path)
    df.to_root(path, key=name_key)


#################################################################################################
###################################### PICKLE functions #########################################
################################################################################################# 

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
    


    
    