"""
Anthony Correia
02/01/21
- Define some PDFs (sum of CB or gaussian PDFs)
- Define parameters
- Check if a fit converged
- launch a fit
"""

from bd2dst3pi.locations import loc
from load_save_data import create_directory, add_in_dic, save_json
from plot.tool import el_to_list


import zfit
import json

#################################################################################################
############################################# PDFs ##############################################
#################################################################################################

def crystall_ball_gaussian(mu, sigma, obs, alpha=None, n=None):
    """ return a CB or gaussian PDF:
    - Gaussian if alpha or n is None
    - CB if none of them is None
    """
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

def define_zparams(initial_values, cut_BDT=None, num=None):
    """Define zparams from the dictionnary initial_values
    
    @initial_values :: dictionnary {"name_variable": {"value":, "low":, "high":, "floating":}}
    @cut_BDT        :: float, performed cut on the BDT
    @num            :: integer, index of the fit --> add ;{num} at the end of the variable
                            the other functions I wrote allow to ignore the ';{num}' in the name of
                            the variable
    
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
        if num is not None:
            name_var += f';{num}'
        
        init = initial_values[var]
        add_in_dic('value', init, default=None)
        add_in_dic('low', init, default=None)
        add_in_dic('high', init, default=None)
        add_in_dic('floating', init, default=True)
        
        zparams[var] = zfit.Parameter(name_var, init['value'], init['low'], init['high'],
                                          floating = init['floating'])
        
    return zparams

def check_fit(result):
    """ Check if the fit has gone well. If a check has gone wrong, point it out.
    @result  :: result of the minimisation of the likelihood
    
    @returns :: False if something has gone wrong in the fit. 
    
    """
    fit_ok = True
    info_fit = result.info['original']
    
    checks = {}
    checks['is_valid'] = True
    checks['has_valid_parameters'] = True
    checks['has_accurate_covar'] = True
    checks['has_posdef_covar'] = True
    checks['has_made_posdef_covar'] = False
    checks['hesse_failed'] = False
    checks['has_covariance'] = True
    checks['is_above_max_edm'] = False
    checks['has_reached_call_limit'] = False
    
    for check, desired in checks.items():
        if info_fit[check]!=desired:
            print(f'Problem: {check} is '+ str(info_fit[check]))
            fit_ok = False
    
    edm = info_fit['edm']
    if edm > 0.001:
        print(f'edm = {edm} > 0.001')
        fit_ok = False
    
    if result.params_at_limit:
        print(f'param at limit')
        fit_ok = False
    
    return fit_ok
    
        
#################################################################################################
###################################### Fitting functions ########################################
################################################################################################# 
  
def save_params(params,name_data, uncertainty=False, dic_add=None, name_folder=None, remove=None):
    """ Save the parameters of the fit in {loc.JSON}/{name_data}_params.json
    
    @params        :: Result 'result.params' of the minimisation of the loss function
                         given by launch_fit
    @name_data     :: name of the data (used for the name of the file where the parameters
                        will be saved    
    @uncertainty   :: boolean, if True, save also the uncertainties
    dic_add        :: dict, if not None, other parameter to save in the json file.
    
    """
    if remove is not None:
        remove = el_to_list(remove,1)
    
    
    param_results = {}
    for p in list(params.keys()): # loop through the parameters
        # Retrieve name, value and error
        name_param = p.name
        if remove is not None:
            for t in remove:
                name_param = name_param.replace(t, '')
            
        value_param = params[p]['value']
        param_results[name_param] = value_param
        if uncertainty:
            error_param = params[p]['minuit_hesse']['error']
            param_results[name_param+'_err'] = error_param
    
    if dic_add is not None:
        for key, value in dic_add.items():
            param_results[key] = value
    
    save_json(param_results, name_data, name_folder=directory)

   
    
def launch_fit(model, data, extended=False, verbose=True):
    """Fit the data with the model
    
    @model    :: zfit model
    @data     :: zfit data
    @extended :: if True, define an extended loss function
    @verbose  :: Bool, print or not the result of the fit
    
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
    minimizer = zfit.minimize.Minuit(verbosity=verbose*5)
    # minimise with nll the model with the data
    result = minimizer.minimize(nll)

    # do the error calculations, here with Hesse
    param_hesse = result.hesse() # get he hessien
    #param_errors, _ = result.errors(method='minuit_minos') # get the errors (gaussian)
    
    param = result.params
    if verbose:
        print(param)        
    return result, param


    
    