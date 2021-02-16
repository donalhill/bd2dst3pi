"""
Anthony Correia
07/02/21
Handle root files with Pandas for the bd2dst3pi project.
- load a root file into a Pandas dataframe
- Apply cuts the a Pandas dataframe (on DeltaM, tau_M, PIDK)
- Save a Pandas dataframe into a root file
"""

from bd2dst3pi.locations import loc
from bd2dst3pi.definition import years as all_years
from bd2dst3pi.definition import magnets as all_magnets
from bd2dst3pi.definition import Dst_M_PDG

import pandas as pd
from copy import deepcopy
from root_pandas import read_root
import numpy as np

import sys
sys.path.append(loc.ROOT + 'library/')
from HEA.pandas_root import assert_needed_variables, show_cut, load_dataframe
from HEA.tools.da import el_to_list

#################################################################################################
######################################## CUTS function ##########################################
################################################################################################# 

def apply_cut_DeltaM(df):
    """Perform the following cut on DeltaM: `143 < DeltaM < 148` with `DeltaM = Dst_M - D0_M`
    
    Parameters
    ----------
    df      : pandas.Dataframe
    
    Returns
    -------
    pandas.Dataframe
        cut on `DeltaM`
    
    """
    assert_needed_variables(['Dst_M', 'D0_M'], df)
    
    df["Delta_M"] = df["Dst_M"] - df["D0_M"]
    df = show_cut(df, "Delta_M > 143. and Delta_M < 148.")
    
    return df

def apply_cut_tau_Ds(df, mean=1969., size=50):
    """Cut out the events with `abs(tau_M - {mean}) > {size}` in a pandas dataframe
    
    Parameters
    ----------
    df      :: pandas dataframe with the variable `tau_M`
    mean    :: float
    size    :: float
    
    returns
    -------
    pandas.Dataframe
        Dataframe cut on `tau_M`
    """
    
    assert_needed_variables('tau_M', df)
    
    df = show_cut(f"abs(tau_M - {mean}) <= {size}")
    return df


def apply_cut_PIDK(df, cut=4):
    """ Cut out the events `tau_pion{i}_ID > {cut}` if `tau_pion` and `Dst_PID` have opposite charges
    
    Parameters
    ----------
    df      : pandas.Dataframe 
        dataframe with the following variables:
                ```
                'tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID',
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                'Dst_ID'
                ```
    cut     : float
        cut that will be applied to the PIKD variables
    
    Returns
    -------
    pandas.Dataframe
        cut on the `tau_pion{i}_ID` variables
    
    Notes
    -----
    - tau_pion1_ID is observed to have always the same sign as Dst_ID. 
    - tau_pion0_ID and tau_pion2_ID are observed to have an opposite sign to Dst_ID.
    
    We did not use that in this function though.
    
    """
    needed_variables = ['tau_pion0_ID', 'tau_pion1_ID', 'tau_pion2_ID',
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                'Dst_ID']
    
    assert_needed_variables(needed_variables, df)
    
    for i in range(3): # i = 0, 1, 2
        df = show_cut(df, f"(tau_pion{i}_PIDK < {cut}) | (Dst_ID*tau_pion{i}_ID)>0")
    
    
    return df

def apply_cut_allPIDK(df, cut=4):
    """ Cut out the events with `tau_pion_ID > {cut}`
    
    
    Parameters
    ----------
    df      : pandas.Dataframe 
        dataframe with the following variables:
                ```
                'tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK',
                ```
    cut     : float
        cut that will be applied to the PIKD variables
    
    Returns
    -------
    pandas.Dataframe
        cut on the `tau_pion{i}_ID` variables 
    """
    needed_variables = ['tau_pion0_PIDK', 'tau_pion1_PIDK', 'tau_pion2_PIDK']
    assert_needed_variables(needed_variables, df)
    
    for i in range(3): # i = 0, 1, 2
        df = show_cut(df, f"(tau_pion{i}_PIDK < {cut})")
    
    return df


## ADD COLUMNS ===============================================================

def add_flight_distance_tau(df):
    """ Create new columns in df
    - `tau_flight_{axis}`, `tau_flight_{axis}err` and `tau_flight_{axis}sig` for the 3 `axes` 'X', 'Y' and 'Z'
    - `tau_flight`, `tau_flight_err` and `tau_flight_err`, taking into account the 3 axes
    
    Parameters
    ----------
    df    : pandas.Dataframe 
                Needed variables in df
                    - B0_ENDVERTEX_X, B0_ENDVERTEX_Y, B0_ENDVERTEX_Z
                    - tau_ENDVERTEX_X, tau_ENDVERTEX_Y, tau_ENDVERTEX_Z
                    - tau_ENDVERTEX_XERR, tau_ENDVERTEX_YERR, tau_ENDVERTEX_ZERR
    Returns
    -------
    pandas.Dataframe
        with the new variables
    """
    
    needed_variables = ['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']
    needed_variables+= ['tau_ENDVERTEX_X', 'tau_ENDVERTEX_Y', 'tau_ENDVERTEX_Z']
    needed_variables+= ['tau_ENDVERTEX_XERR', 'tau_ENDVERTEX_YERR', 'tau_ENDVERTEX_ZERR']
    assert_needed_variables(needed_variables, df)
    
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


def add_constr_Dst(df, col='B0_M', Dst_M_PDG=Dst_M_PDG):
    """ Add the column `'Ds_constr_{col}'` in the df
    
    Parameters
    ----------
    df     :: pandas dataframe with the variablaes 'B0_M' and 'Dst_M'
    col    :: str, name of the column of dataframe from which is computed the constrained column
    
    Returns
    -------
    pandas.Dataframe
        with the new column `'Ds_constr_{col}'`
    """
    
    df[f"Dst_constr_{col}"] = df[col].values - df['Dst_M'].values + Dst_M_PDG
    return df

#################################################################################################
##################################### DATAFRAME function ########################################
################################################################################################# 



def load_data(years=None, magnets=None, type_data='common', vars=None, method='read_root', 
              name_BDT='adaboost_0.8_without_P_cutDeltaM', cut_DeltaM=False, cut_PIDK=None,
             cut_tau_Ds=False):
    """ return the pandas dataframe with the desired data
    
    Parameters
    ----------
    years      : list of int
        list of the wanted years
    magnets    : list
        list of the wanted magnet polarisations
    type_data  : str
        desired data: 'MC', 'data', 'data_strip' or 'ws_strip' or 'data_KPiPi' or 'common'
    vars       : list of str
        variables to load from the root file
    method    : str 
        library to retrieve the data ('read_root' or 'uproot') (read_root is faster)
    name_BDT   : str
        if BDT is one of the variable to retrieve (i.e., BDT in vars)
                    this is the string that indicates in which root file 
                    the 'BDT' variable is:
                        `'{
                        .OUT}/tmp/BDT_{name_BDT}.root'`
    cut_deltaM : Bool
        if true (or if BDT is in vars), perform a cut on `DeltaM`: `143 < DeltaM < 148`
                        
    cut_PIDK   : str
        if 'PID', cut out the events with `tau_pion_ID > 4` if `tau_pion` and `Dst_PID` have opposite signs
        if 'ALL', cut out all the events with `tau_pion_ID > 4`
    cut_tau_Ds : Bool
        if True, cut on `tau_M` around the mass of the Ds meson (+/- 50 MeV)
    
    returns
    -------
    pandas.Dataframe
        with the desired variables for all specified the years and magnets
    """
    assert type_data in ['MC', 'MCc', 'MCe', 'all_MC', 'data_strip', 'data', 'common', 'ws_strip', 'data_KPiPi']
    assert cut_PIDK in [None, 'PID', 'ALL']
    assert method in ['read_root', 'uproot']
    
    mode_BDT = False
    only_one_file = False
    
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
    # MC data -------------------------------------------
    elif type_data == 'MCc':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = '_Sim09c-ReDecay01.root'
    
    elif type_data == 'MCe':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = '_Sim09e-ReDecay01.root'
    elif type_data == 'all_MC':
        path = f"{loc.MC}/Bd_Dst3pi_11266018"
        ext = ['_Sim09e-ReDecay01.root', '_Sim09c-ReDecay01.root']
    
    # Clean data ----------------------------------------
    elif type_data == 'data':
        path = f"{loc.DATA}/data_90000000"
        ext = '.root'
    
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
    
    
    
    if only_one_file:
        dfr_tot = load_dataframe(complete_path, tree_name, vars, method=method)
    else:
        ext = el_to_list(ext, 1)
        for y in years:
            for m in magnets:
                for e in ext:
                    complete_path = f"{path}_{y}_{m}{e}"
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
#     if mode_BDT: (outdated)
#         dfr_tot['BDT'] = load_saved_root(f'../tmp/{type_data}/BDT_{name_BDT}.root', 'BDT', ['BDT'])
    
    dfr_tot = dfr_tot.reset_index()
    return dfr_tot


def load_saved_root(data_name, vars=None, name_folder="", tree_name=None, cut_BDT=None, method='read_root'):
    """ Load a saved root file, with the tree_name specified in the variable.py file.
    
    Parameters
    ----------
    data_name  : str, name of the root file
    vars       : list of str,
        list of the desired variables
    method     : str,
        method to retrieve the data ('read_root' or 'uproot') (read_root is faster)
    cut_BDT    : str or float or None
        if not None, the root file with the BDT cut (`BDT > cut_BDT`) applied
        the root file file is saved as with the extension `'_BDT{cutBDT}"` in this case
    cut_deltaM : Bool
        if True (or if BDT is in vars), perform a cut the `143 < DeltaM < 148` cut
    
    Returns
    -------
    pandas.Dataframe 
        loaded, with the desired variables
    """    
    return load_saved_root(data_name, vars=None, name_folder="", tree_name=tree_name, cut_BDT=None, method='read_root')

