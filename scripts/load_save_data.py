"""
Anthony Correia
02/01/21
- Create directories
- Add a key-value couple to a dictionnary if the key is not in the dictionnary
- Apply cuts the a Pandas dataframe (on DeltaM, tau_M, PIDK)
- Add some variables inside a dataframe (fight distance of the 3pi system, Ds-constrained invariant mass)
- Load a dataframe
- Load a saved root file
- Save a Pandas dataframe into a root file
- Dump/retrieve a pickle File
- Save a json file
- format the name of the parameters
- Retrieve a json file
- get a json latex table from parameters that are saved in a json file

"""

from root_pandas import read_root

import pickle
import json
from uncertainties import ufloat

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

def el_to_list(el, len_list):
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
    # DeltaM = Dst_M - D0_M
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

## VARIABLES AND FUNCTION ====================================================

def get_name_var(variable, name_function):
    """
    @variable      :: str or list of str, variable or list of variables
    @name_function :: str, name of the function
    """
    if name_function is not None:
        name_variable = str(variable).replace('(', '').replace(')', '').replace(' ','').replace("'",'')
        new_variable = name_variable + ':' + name_function
    
        return new_variable
    else:
        return variable

def get_needed_vars(variables_functions):
    """
    @variables_functions ::  list of:
                                - variable
                                - tuple (variable, function), where function is the name of the function
                                - tuple (variables, function), where variables is a tuple of variables,
                                    input of the function
    
    @returns :: list of all the needed variables (that need to loaded)
    """
    variables = []
    for variable_function in variables_functions:
        if isinstance(variable_function, tuple):
            variable = variable_function[0]
        else:
            variable = variable_function
        
        if isinstance(variable, tuple):
            variables += variable
        else:
            variables.append(variable)
    
    return variables

def get_real_vars(variables_functions):
    """
    @variables_functions ::  list of:
                                - variable
                                - tuple (variable, function), where function is the name of the function
                                - tuple (variables, function), where variables is a tuple of variables,
                                    input of the function
    
    @returns :: list of all the variables
    """
    variables = []
    for variable_function in variables_functions:
        if isinstance(variable_function, tuple):
            variable = variable_function[0]
            name_function = variable_function[1]
        else:
            variable = variable_function
            name_function = None
        
        new_variable = get_name_var(variable, name_function)
        variables.append(new_variable)
    
    return variables

def get_df_variables(df, variables_functions, mode='new', functions=v.functions):
    '''
    @variables_functions ::  list of:
                                - variable
                                - tuple (variable, function), where function is the name of the function
                                - tuple (variables, function), where variables is a tuple of variables,
                                    input of the function
    @df                  :: Pandas dataframe, original dataframe
    @mode                :: 3 modes:
                - 'add': add the variables to the dataframe
                - 'new': create a new dataframe with the new variables only
                - 'both' : do both
    
    @returns :: list of all the variables
    '''
    
    new_df = pd.DataFrame()
    
    new_df_required = (mode=='new' or mode=='both')
    
    for variable_function in variables_functions:
        if isinstance(variable_function, tuple):
            variable = variable_function[0]
            name_function = variable_function[1]
        else:
            variable = variable_function     
            name_function = None
                
        if name_function is None and new_df_required:
            new_df[variable] = df[variable].values
        
        if name_function is not None:
            new_variable = get_name_var(variable, name_function)
            
            if isinstance(variable, tuple) or isinstance(variable, list):
                data = tuple([df[var] for var in variable])
            else:
                data = df[variable]
            
            new_data = functions[name_function](data).values
            
            if new_df_required:
                new_df[new_variable] = new_data
            
            if mode=='add' or mode=='both':
                df[new_variable] = new_data
        
        
    if new_df_required:
        return new_df

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
        if tree_name in v.data_tree_names:
            tree_name = v.data_tree_names[name_data]
        else:
            tree_name = 'DecayTree'
    complete_path = f"{loc.OUT}/root/{name_folder}/{name_data}{text_cut_BDT}.root"
    return load_dataframe(complete_path, tree_name, vars, method=method)
    
    
    

def load_data(years=None, magnets=None, type_data='common', vars=None, method='read_root', 
              name_BDT='adaboost_0.8_without_P_cutDeltaM', cut_DeltaM=False, cut_PIDK=None,
             cut_tau_Ds=False):
    """ return the pandas dataframe with the desired data
    
    @years      :: list of the wanted years
    @magnets    :: list of the wanted magnets
    @type_data  :: str, desired data: 'MC', 'data', 'data_strip' or 'ws_strip' or 'data_KPiPi' or 'common'
    @vars       :: list of str, desired variables
    @method     :: str, method to retrieve the data ('read_root' or 'uproot')
                        read_root is faster
    @name_BDT   :: str, if BDT is one of the variable to retrieve (i.e., BDT in vars)
                    this is the str that indicates in which root file 
                    the 'BDT' variable is:
                        '{loc.OUT}/tmp/BDT_{name_BDT}.root'
    @cut_deltaM :: Bool, if true (or if BDT is in vars), perform a cut on DeltaM
                        143 < DeltaM < 148
    @cut_PIDK   :: Bool, if 'PID', cut out the events with tau_pion_ID < 4 if tau_pion and Dst_PID has an opposite charge
                   if 'ALL', cut out all the events with tau_pion_ID < 4
    @cut_tau_Ds :: Bool, if true, cut on tau_M around the mass of the Ds meson
    
    @returns    :: Pandas df with the desired variables for all specified the years and magnets
    """
    assert type_data in ['MC', 'MCc', 'MCe', 'all_MC', 'data_strip', 'data', 'common', 'ws_strip', 'data_KPiPi']
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
#################################### PICKLE/JSON functions ######################################
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

def save_json(dic, name_data, name_folder=None):
    """
    @dic         :: dict, to save in the json file
    @name_data   :: str, name of the json file
    @name_folder :: str, name of the folder where the json file is saved
                        if None: no folder
    
    Save dic in a json file in {loc.JSON}/{name_folder}/{name_data}.json
    """
    directory = create_directory(loc.JSON, name_folder)
    path = f"{directory}/{name_data}_params.json"
    
    with open(path,'w') as f:
        json.dump(dic, f, sort_keys = True, indent = 4)
    print(f"parameters saved in {path}")
    
#################################################################################################
######################################## FIT functions ##########################################
################################################################################################# 

def format_previous_params(df_params, retrieve_err=False):
    """ Remove the element in the dictionnary that ends with '_err'.
    For the other ones, removes what is after | in the keys.
    In particular: variable|BDT-0.2 will become variable
    
    @df_params_recup  :: dataframe with the result of the file 
                            this is the df saved in .json after the fit.
                            * {'alphaL': value, 'alphaL_err': value ...}
                            or
                            * {'alphaL|BDT-0.5': value, 'alphaL|BDT-0.5'_err': value ...}
    @retrieve_err     :: bool, if True, include the error of the variables in the file
                            
    @returns          :: new dataframe
    """
    
    df_params_formatted = {}
    for key, value in df_params.items():
        index = key.find('|')
        if index==-1:
            index = None
        variable = key[:index]
        if key.endswith('_err') and retrieve_err:
            if not variable.endswith('_err'):
                variable+='_err'
            
        if not key.endswith('_err') or retrieve_err:
            df_params_formatted[variable] = value
    return df_params_formatted

def retrieve_params(name_data, name_folder=None):
    """ retrieve the content of a json file
    @name_data   :: str, name of the fit
    @name_folder :: str, name of folder where the json file is
                        (if None, there is no folder)
    
    @return      :: dictionnary that contains the variables stored in the json file
                        in {loc.JSON}/{name_folder}/{name_data}_params.json
    """
    directory = create_directory(loc.JSON, name_folder)
    path = f"{directory}/{name_data}_params.json"
    
    with open(path,'r') as f:
        params = json.load(f)
    
    return params


def get_list_without_err(params):
    """ get the list of variables from the list of fitted parameters
    @params   :: dict, fitted parameters {'alphaL': value, 'alphaL_err': value ...}
    @returns  :: list of variables (remove the keys with '_err')
    """
    keys = list(params.keys())
    variables = []
    for key in keys:
        if not key.endswith('_err'):
            variables.append(key)
    return variables

def json_to_latex_table(name_json, path, name_params, show=True):
    """ transform a json file that contains the fitted parameters and uncertainties into a latex table
    @name_json     :: str, name of the json file
    @path          :: str, path of the json file compared to loc.JSON
    @name_params   :: str, alternative name of the parameters (in latex)
    @show          :: bool, if True, shows the content of the created latex table code
    """
    
    # Open the json file
    directory = create_directory(loc.JSON, path)
    with open(f'{directory}/{name_json}_params.json', 'r') as f:
        params = json.load(f)
    params = format_previous_params(params, True)
    # Load the variables into ufloats
    variables = get_list_without_err(params)
    ufloats = {}
    for variable in variables:
        if f'{variable}_err' in params:
            ufloats[variable] = ufloat(params[variable], params[f"{variable}_err"])
    
    # Write the .tex file
    directory = create_directory(loc.TABLES, path)
    file_path = f'{directory}/{name_json}_params.tex'
    with open(file_path, 'w') as f:
        f.write('\\begin{tabular}[t]{lc}')
        f.write('\n')
        f.write('\\hline')
        f.write('\n')
        f.write('Variable &Fitted Value\\\\')
        f.write('\n')
        f.write('\\hline\\hline')
        f.write('\n')
        
        for variable, value in ufloats.items():
            formatted_value = f'{value:.2u}'.replace('+/-','\pm').replace('e+0','\\times 10^')
            name_var = name_params[variable]
            f.write(f"{name_var}&${formatted_value}$\\\\")
            f.write('\n')
            f.write('\\hline')
            f.write('\n')
        f.write("\\end{tabular}")
    if show:
        show_latex_table(name_json, path)
        
def show_latex_table(name, path):
    """ Print the latex table that contains the result of a fit. It must have been already generated, with json_to_latex_table. 
    @name  :: name of the fit which we want to get the latex table with its result
    @path  :: path of the .tex file from loc.TABLES, where the .tex file is
    
    print the content of the tex file in {loc.TABLES}/{path}/{name}_params.tex
    """
    directory = create_directory(loc.TABLES, path)
    file_path = f'{directory}/{name}_params.tex'
    print(file_path)
    with open(file_path, 'r') as f:
        print(f.read())
        

    