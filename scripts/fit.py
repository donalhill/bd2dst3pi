from bd2dst3pi.locations import loc
from load_save_data import create_directory, add_in_dic


import zfit
import json

#################################################################################################
############################################# PDFs ##############################################
#################################################################################################

def crystall_ball_gaussian(mu, sigma, obs, alpha=None, n=None):
    if alpha is None or n is None:
        pdf = zfit.pdf.Gauss(mu, sigma, obs=obs)
    else:
        pdf = zfit.pdf.CrystalBall(mu, sigma, alpha, n, obs=obs)
    return pdf

def sum_crystalball(muL, muR, sigmaL, sigmaR, frac, obs, alphaL=None, alphaR=None, nL=None, nR=None):
    """ Return the sum of 2 crystall ball PDFs.
    If the alpha or n is None, the corresponding distribution is a gaussian.
    """
    pdfL = crystall_ball_gaussian(muL, sigmaL, obs,
                                         alphaL,nL)
    pdfR = crystall_ball_gaussian(muR, sigmaR, obs,
                                         alphaR, nR)
    
    model = zfit.pdf.SumPDF([pdfL, pdfR], fracs=frac)
    
    return model, pdfL, pdfR

#################################################################################################
########################################## Parameters ###########################################
#################################################################################################

def format_previous_params(df_params_recup, recup_err=False):
    """ Remove the element in the dictionnary that ends with '_err'.
    For the other ones, removes what is after | in the keys.
    In particular: variable|BDT-0.2 will become variable
    
    @df_params_recup  :: dataframe with the result of the file 
                            this is the df saved in .json after the fit.
                            
    @returns          :: new dataframe
    """
    df_params_recup_formatted = {}
    for key, value in df_params_recup.items():
        if recup_err or not key.endswith('_err'):
            index = key.find('|')
            df_params_recup_formatted[key[:index]] = value
    
    return df_params_recup_formatted

def define_zparams(initial_values, cut_BDT=None):
    """Define zparams from the dictionnary initial_values
    
    @initial_values :: dictionnary {"name_variable": {"value":, "low":, "high":, "floating":}}
    @cut_BDT        :: float, performed cut on the BDT
    
    @returns        :: dictionnary zparams. To access to one of the variable
                                * if cut_BDT is None, zparams[{name_variable}]
                                * else, zparams[{name_variable}|BDT{cut_BDT}]
    """
    zparams = {}
    for var in initial_values.keys():
        if cut_BDT is not None:
            name_var = f"{var}|BDT{cut_BDT}"
        else:
            name_var = var
        
        init = initial_values[var]
        add_in_dic('value', init, default=None)
        add_in_dic('low', init, default=None)
        add_in_dic('high', init, default=None)
        add_in_dic('floating', init, default=True)
        
        zparams[var] = zfit.Parameter(name_var, init['value'], init['low'], init['high'],
                                          floating = init['floating'])
        
    return zparams

#################################################################################################
###################################### Fitting functions ########################################
################################################################################################# 
  
def save_params(params,name_data, uncertainty=False, dic_add=None, name_folder=None):
    """ Save the parameters of the fit in {loc.JSON}/{name_data}_params.json
    
    @params        :: Result 'result.params' of the minimisation of the loss function
                         given by launch_fit
    @name_data     :: name of the data (used for the name of the file where the parameters
                        will be saved    
    @uncertainty   :: boolean, if True, save also the uncertainties
    dic_add        :: dict, if not None, other parameter to save in the json file.
    
    """
    param_results = {}
    for p in list(params.keys()): # loop through the parameters
        # Retrieve name, value and error
        name_param = p.name
        value_param = params[p]['value']
        param_results[name_param] = value_param
        if uncertainty:
            error_param = params[p]['minuit_hesse']['error']
            param_results[name_param+'_err'] = error_param
    
    if dic_add is not None:
        for key, value in dic_add.items():
            param_results[key] = value
    
    directory = create_directory(loc.JSON, name_folder)
    path = f"{directory}/{name_data}_params.json"
    
    with open(path,'w') as f:
        json.dump(param_results, f, sort_keys = True, indent = 4)
    print(f"parameters saved in {path}")

def retrieve_params(name_data, name_folder=None):
    directory = create_directory(loc.JSON, name_folder)
    path = f"{directory}/{name_data}_params.json"
    
    with open(path,'r') as f:
        params = json.load(f)
    
    return params
    
    
def launch_fit(model, data, extended = False):
    """Fit the data with the model
    
    @model    :: zfit model
    @data     :: zfit data
    @extended :: if True, define an extended loss function
    
    @returns  ::
         - result: result of the minimisation of the loss function
         - param : result.params, the fitted parameters (zfit format)
    """
    
    ## Minimisation of the loss function 
    if extended:
        nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    else:
        nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    
    # create a minimizer
    minimizer = zfit.minimize.Minuit()
    # minimise with nll the model with the data
    result = minimizer.minimize(nll)

    # do the error calculations, here with Hesse
    param_hesse = result.hesse() # get he hessien
    #param_errors, _ = result.errors(method='minuit_minos') # get the errors (gaussian)
    param = result.params
    print(param)        
    return result, param

#################################################################################################
####################################### TABLE function ##########################################
################################################################################################# 

def json_to_latex_table(path, name_json):
    with open(f'{loc.JSON}/{path}/{name_json}.json', 'r') as f:
        params = json.load(f)
    
    directory = create_directory(loc.JSON, path)
    
    