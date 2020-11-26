import functions as fct

from bd2dst3pi.locations import loc


import zfit
import json

from uncertainties import ufloat, unumpy
from uncertainties.core import Variable as ufloat_type    


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False



#################################################################################################
######################################## Tools functions ########################################
#################################################################################################


def get_plot_scaling(counts, obs, n_bins):
    """Return plot_scaling, the factor to scale the curve fit to unormalised histogram"""
    return counts.sum() * obs.area() / n_bins 
    
def get_frac_model(model, n_tot):
    ''' Return the yield of the PDF of model, relatively to the total PDF
    
    @model   :: extended zfit model
    @n_tot   :: total number of events (or yield of the total pdf)
    
    @return  :: number_element_model / n_tot
    '''
    
    return model.get_yield()/n_tot

def frac_model(x, model, frac=None):
    """ 
    Return the list of the values of the pdf of the model evaluated in x.
    If frac is not None, multiply by frac the list.
    
    @x       :: list of float
    @model   :: zfit model
    @frac    :: float
    
    @returns :: list of the values of the pdf of the model evaluated in x
                    multiplied by frac (if it is not None)
    
    """
    if frac is None:
        return model.pdf(x).numpy()
    else:
        return (model.pdf(x)*frac).numpy()

def el_or_list_to_group(l, type_el=np.ndarray):
    """
    @l       :: element of type type_el, or list of element of type type_el, or list of list of element of type type_el
    @type_el :: type
    
    @returns :: list of list of element of type type_el
    """
    if isinstance(l,type_el):
        groups_l = [[l]]
    elif isinstance(l[0],type_el):
        groups_l = [l]
    else:
        groups_l = l
    
    return groups_l

def get_chi2(fit_counts, counts):
    """ chi2 of the fit
    
    @fit_counts :: fitted number of counts
    @counts     :: number of counts in the histogram of the fitted data
    
    @returns :: float, chi2 of the fit
    """
    diff = np.square(fit_counts-counts)
    
    n_bins = len(counts)
    diff = np.divide(diff,counts,out=np.zeros_like(diff), where=counts!=0)
    chi2 = np.sum(diff)/n_bins # sigma_i^2 = mu_i
    return chi2

#################################################################################################
#################################### Sub-plotting functions #####################################
#################################################################################################
    
    
def plot_pull_diagram(ax, model, counts, centres, err, low=None, high=None, 
                      plot_scaling=None, fontsize=25, color='b', color_lines='r', show_chi2=False):
    """
    Plot pull diagram of 'model' compared to the data given by (counts, centres)
    
    @ax            :: axis where to plot
    @model         :: zfit model
    @counts        :: counts of the bins given by centres, in the histogram
    @centres       :: bin centres of the histogram
    @fontsize      :: fontsize of the labels
    @color         :: color of the pull diagram
    @color_lines   :: color of the lines y=0, y=2 and y=-2 (default is red)
    @show_chi2     :: Boolean, if True, show chi2
    """
    
    ## Computing
    if low is None:
        low = centres[0]-(centres[1]-centres[0])/2
    if high is None:
        high = centres[-1]-(centres[1]-centres[0])/2
    
    if plot_scaling is None:
        n_bins = len(centres)
        plot_scaling = counts.sum() * (high-low) / n_bins
        
    fit = model.pdf(centres).numpy()*plot_scaling
    with np.errstate(divide='ignore', invalid='ignore'): # ignore divide-by-0 warning
        pull = np.divide(counts-fit,err)
    
    ## Plotting
    ax.errorbar(centres,pull, yerr = np.ones(len(centres)),color =color, ls='', marker='.')
    ax.plot([low,high],[2,2],color='r',ls='--')
    ax.plot([low,high],[-2,-2],color='r',ls='--')
    ax.plot([low,high],[0,0],color='r')
    
    ## Label and ticks
    ax.set_ylabel('residuals / $\\sigma$', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticks([-2,0,2])
    #ax[1].grid()
    ax.set_xlim([low,high]) 
    
    if show_chi2:
        chi2 = get_chi2(fit, counts)
        ax.set_title(f'($\\chi^2$={chi2:.2f})', fontsize=fontsize)
    
def plot_fitted_curve(ax, model, plot_scaling, frac=None, line_width=2.5,
                      color='b',linestyle='-', low=None, high=None, label=None, x=None):
    """
    Plot fitted curve given by 'model'
    
    @ax            :: axis where to plot
    @model         :: zfit model
    @plot_scaling  :: scaling to get the scale of the curve right
    @low           :: low limit of the plot (x-axis)
    @high          :: high limit of the plot (x-axis)
    @line_width    :: width of the curve line
    @color         :: color of the curve
    @x             :: points of the x-axis where to evaluate the pdf of the model to plot
    """
    if x is None:
        x = np.linspace(low, high, 1000)
    y = frac_model(x, model, frac=frac)* plot_scaling
    ax.plot(x, y, linewidth=line_width, color=color, ls=linestyle, label=label)


# Alternative names for the models
name_PDF = {
    'DoubleCB': 'Double CB',
    'SumPDF'  : 'Sum',
    'Exponential': 'Exponential',
    'Gauss' : 'Gaussian',
    'CrystalBall': 'Crystal Ball'
}
    
    
def plot_fitted_curves(ax, models, plot_scaling, low, high, name_models=None, type_models=None, frac=None,
                       line_width=2.5, colors=None, fontsize_legend=16, loc_leg='upper left'):
    """
    Plot fitted curve given by 'models', with labels given by name_models
    
    @ax            :: axis where to plot
    @models        :: zfit model of list of zfit models
    @plot_scaling  :: scaling to get the scale of the curve right
    @low           :: low limit of the plot (x-axis)
    @high          :: high limit of the plot (x-axis)
    @name_models   :: str or list of str - name of the model that will appear in the label of the curve
    @type_models   :: str, type of each mode (one character for each model):
                        - 'm' : model (sum)
                        - 's' : signal
                        - 'b' : background
                     initially None, which means that the first is 'm', the second is 's' and the other ones are 'b'
    @line_width    :: width of the curve lines
    @colors        :: 1 color or list of colors (one color for each model)
    @fontsize      :: fontsize of the legend
    """
    models = fct.el_to_list(models, 1)
    show_legend = len(models) > 1
    name_models = fct.el_to_list(name_models, len(models))

    if colors is None:
        colors = ['b', 'g', 'gold', 'magenta', 'orange']
        
    x = np.linspace(low, high, 1000)
    
    # Get the total number of events in the models
    if len(models) > 1:
        model = models[0] # total PDF
        if frac is None:
            n_tot = 0
            for yield_model in model.get_yield().params.values():
                n_tot += float(yield_model.value())
    
    # Plot the models
    if type_models is None:
        type_models = 'm'
        if len(name_models) >= 2:
            type_models += 's'
            type_models += 'b'*(len(name_models)-2)
    
    name_type_models = {
        'm' : 'model',
        's' : 'signal',
        'b' : 'background'
    }
    
    for k, model in enumerate(models):
        marker = model.name.find('_') 
        if marker == -1:
            marker = None
        label_model = name_PDF[model.name[:marker]]
        if k != 0:
            
            if frac is None:
                applied_frac = get_frac_model(model, n_tot)
            elif k == 1:
                applied_frac = frac
            elif k == 2 :
                applied_frac = 1 - frac
            
            ls = '--'
            
        else: # k = 0 is the total PDF
            applied_frac = None
            ls ='-'
        
        label_model += f' - {name_type_models[type_models[k]]}'
            
            
        label_model = fct.add_text(label_model, name_models[k])
        plot_fitted_curve(ax, model, plot_scaling, frac=applied_frac, line_width=2.5, color=colors[k], 
                          linestyle = ls, label=label_model, x=x)
    
    if show_legend:
        ax.legend(fontsize=fontsize_legend, loc=loc_leg)
   
        
def plot_result_fit(ax, params, name_params=None, fontsize=20, colWidths=[0.04,0.01,0.08,0.06], loc='upper right'):
    """
    Plot fitted the results of the fit in a table
    
    @ax            :: axis where to plot
    @params        :: Result 'result.params' of the minimisation of the loss function
                         given by launch_fit
    @name_params   :: Dictionnary with the name of the params
                        Also indicated the variables to plot among all the variables in params
    @fontsize      :: Font size of the text in the table
    @colWidths     :: Width of the four columns of the table
    """
    
   #result_fit = ""
    result_fit = []
    for p in list(params.keys()): # loop through the parameters
        
        name_param = p.name
        
        # if name_params not None, it specifies the variables we want to show
        if (name_params is None) or (name_param in name_params): 
            # Retrieve value and error
            value_param = params[p]['value']
            error_param = params[p]['minuit_hesse']['error']
            
            # Retrieve alt name
            if name_params is not None:
                surname_param = name_params[name_param]
            else:
                surname_param = fct.latex_format(name_param)

            # Table --> name_param   ::   value_param +/- error_param
            size_int_part = len(str(int(value_param)))
            if size_int_part>=4:
                value_param_text = f"{value_param:.0f}"
            else:
                value_param_text = f"{value_param:.4f}"
            result_fit.append([surname_param,":",value_param_text,f'$\pm$ {error_param:.2g}']) 

    # Plot the table with the fitted parameters in the upper right part of the plot  
    table = ax.table(result_fit,loc=loc, edges='open', cellLoc='left',
              colWidths=colWidths)                 
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(2, 2) 

    
def plot_xys (ax, x, ly, xlabel, labels=None, colors=['b','g','r','y'], fontsize=25,
             linewidth=2.5):
    """
    @ax        :: axis where to plot
    @x         :: list of float, points of the x-axis
    @ly        :: list of list of float, list of the y-points of the curves
    @labels    :: labels of the curves
    @xlabel    :: str, xlabel
    @ylabel    :: str, ylabel
    @colors    :: list of str, colors of each curve in ly
    @fontsize  :: fontsize of the labels
    @linewidth :: linewidth of the plotted curves
    """
    colors = fct.el_to_list(colors, len(ly))
    
    plot_legend = False
    
    for i, y in enumerate(ly):
        label = labels[i] if len(ly) > 1 else None
        ax.errorbar(x, unumpy.nominal_values(y), yerr=unumpy.std_devs(y), linestyle='-', color=colors[i], 
                    linewidth=linewidth, label=label, marker='.')
        
        if label is not None:
            plot_legend = True
    
    ax.set_xlabel(xlabel, fontsize=25)
    
    if len(ly)==1:
        ax.set_ylabel(labels[0], fontsize=25)
    else:
        ax.set_ylabel('value', fontsize=25)
    
    if plot_legend:
        ax.legend(fontsize=25)
    
#################################################################################################
###################################### Fitting functions ########################################
################################################################################################# 
  
def save_params(params,name_data,uncertainty=False):
    """ Save the parameters of the fit in {loc.JSON}/{name_data}_params.json
    
    @params        :: Result 'result.params' of the minimisation of the loss function
                         given by launch_fit
    @name_data     :: name of the data (used for the name of the file where the parameters
                        will be saved    
    @uncertainty   :: boolean, if True, save also the uncertainties
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
    with open(f"{loc.JSON}/{name_data}_params.json",'w') as f:
        json.dump(param_results, f, sort_keys = True, indent = 4)
    print(f"parameters saved in {loc.JSON}/{name_data}_params.json")

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
    param_errors, _ = result.errors(method='minuit_minos') # get the errors (gaussian)
    param = result.params
    print(param)        
    return result, param


#################################################################################################
###################################### Plotting function ########################################
#################################################################################################   
    
def plot_hist_fit(df, variable, name_var=None, unit_var=None,models=None, obs=None, n_bins=50, color='black', 
                  name_models=None, type_models=None, frac=None,
                  mode_hist=True, linewidth=2.5, colors=None,
                  name_data_title=False, title=None, fontsize_leg=20,
                  name_file=None,name_folder=None,name_data=None, show_chi2=False,
                  params=None,name_params=None, colWidths=[0.04,0.01,0.06,0.06], loc_res='upper right', loc_leg='upper left'):
    """ Plot complete histogram with fitted curve, pull histogram and results of the fits, save it in plots/
    @df            :: pandas dataframe that contains all the variables, including 'variable'
    @variable      :: name of the variable to plot and fit
    @name_var      :: name of the variable, to be used in the plots
    @models        :: zfit model of list of zfit models
    @obs           :: zfit space (used for the fit)
    @n_bins        :: number of desired bins of the histogram
    @color         :: color of the histogram
    @name_models   :: str or list of str - name of the model that will appear in the label of the corresponding curves
    @type_models   :: str, type of each mode (one character for each model):
                        - 'm' : model (sum)
                        - 's' : signal
                        - 'b' : background
                     initially None, which means that the first is 'm', the second is 's' and the other ones are 'b'
    @model_hist    :: Result 'result.params' of the minimisation of the loss function
                         given by launch_fit
    @mode_hist     :: if True, plot with bars,else, plot with points (and error)
    @linewidth     :: width of the fitted curve line
    @colors        :: colors of the fitted curves
    @name_file     :: name of the saved file
    @name_folder   :: name of the folder where to save the plot
    @name_data     :: name of the data used to constitute the name of the saved file, if name_file is not specified. 
    @cut_BDT       :: float, cut performed in the BDT > {cut_BDT}. This is used for the title and the saved file.
    @params        :: Result 'result.params' of the minimisation of the loss function, given by launch_fit
    @name_params   :: Dictionnary with the name of the params
                        Also indicated the variables to plot among all the variables in params
    @colWidths     :: Width of the four columns of the table
    """
    
    ## Create figure
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    ax = [plt.subplot(gs[i]) for i in range(2)]

    ## Retrieve low,high (of x-axis)
    low = float(obs.limits[0])
    high = float(obs.limits[1])
    
    if name_var is None:
        name_var = fct.latex_format(variable)
    
    if name_data_title and name_data is not None:
        title = fct.add_text(title, name_data, 'before')
    
    ax[0].set_title(title, fontsize=25)
    
    ## plot 1D histogram of data
    # Histogram 
    counts,_,centres,err = fct.plot_hist_alone(ax[0], df[variable], n_bins,
                                               low, high, color, mode_hist, alpha = 0.1)
    
    
    # Label
    bin_width = fct.get_bin_width(low, high, n_bins)
    fct.set_label_hist(ax[0], name_var, unit_var, bin_width, fontsize=25)
    
    
    
    # Ticks
    fct.set_label_ticks(ax[0])
    
    ## Plot fitted curve 
    if isinstance(models,list):
        model = models[0] # the first model is the "global" one
    else:
        model = models
        
    plot_scaling = get_plot_scaling(counts, obs, n_bins)
    #plot_fitted_curve(ax[0], model, plot_scaling, low, high)
    plot_fitted_curves(ax[0], models, plot_scaling, low, high, name_models=name_models, type_models=type_models,
                       line_width=2.5, colors=colors, fontsize_legend=fontsize_leg, loc_leg=loc_leg, frac=frac)
    
    fct.change_ymax(ax[0], factor=1.1)
    
    ## Plot pull histogram
    plot_pull_diagram(ax[1], model, counts, centres, err, 
                      low=low, high=high, plot_scaling=plot_scaling, show_chi2=show_chi2)
    
    
    ## Plot the fitted parameters of the fit
    if params is not None:
        plot_result_fit(ax[0], params, name_params=name_params, fontsize=20, colWidths=colWidths, loc=loc_res)
    
    # Save result
    plt.tight_layout()
    plt.show()  
    directory = f"{loc.PLOTS}/"    
    fct.save_file(fig, name_file,name_folder,f'{variable}_{name_data}_fit',f"{loc.PLOTS}/")



def plot_x_list_ys(x, y, name_x, names_y, surnames_y=None, linewidth=2.5,fontsize=25, name_file=None, name_folder=None):
    """ plot x as a function of the y of the list l_y
    
    @x          :: list or array of floats, points in the x-axis
    @y          :: list of list of numpy arrays of ufloat, list of numpy arrays of ufloat, or one numpy array of ufloat
    @name_y     :: list of str or str, name of each list in l_y
    @surname_y  :: list of list of str, list of str or str, surname of each list in l_y
    colors      :: list of list of str, list of str or str, colors of each graph in l_y
    linewidth   :: float, linewdith
    name_file   :: str, name of the file to save
    name_folder :: str, name of the folder where the image is saved
    """
    
    groups_ly         = el_or_list_to_group(y)
    groups_names_y    = el_or_list_to_group(names_y,str)
    
    if surnames_y is not None:
        groups_surnames_y = el_or_list_to_group(surnames_y,str)
    else:
        groups_surnames_y = groups_names_y
        
    fig, axs = plt.subplots(len(groups_ly),1, figsize=(8,3*len(groups_ly)))
    
    for k, ly in enumerate(groups_ly):
        if len(groups_ly)==1:
            ax = axs
        else:
            ax = axs[k]
        # In the same groups_ly, we plot the curves in the same plot
        plot_xys (ax, x, groups_ly[k], xlabel=name_x, labels=groups_surnames_y[k], fontsize=fontsize,
             linewidth=linewidth)
    
        # Grid
        fct.show_grid(ax, which='major')
        fct.show_grid(ax, which='minor')
        
        # Ticks
        fct.set_label_ticks(ax)
        fct.change_ymax(ax, factor=1.1)
        
    
    plt.tight_layout()
    plt.show()  
    directory = f"{loc.PLOTS}/"    
    fct.save_file(fig, name_file, name_folder, f'BDT_vs_{fct.list_into_string(fct.flattenlist2D(names_y))}', f"{loc.PLOTS}/")

#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################  

def plot_hist_fit_particle(df, variable, cut_BDT=None, **kwargs):
    """ 
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    Then, plot with plot_hist_fit
    
    @df       :: dataframe
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variable, name_var, unit_var
    """
    
    ## Retrieve particle name, and variable name and unit.
#     particle, var = retrieve_particle_variable(variable)
    
#     name_var = variables_params[var]['name']
#     unit_var = variables_params[var]['unit']
#     name_particle = particle_names[particle]
    
    
    
    name_variable, unit_var = fct.get_name_unit_particule_var(variable)
       
    fct.add_in_dic('name_file', kwargs)
    fct.add_in_dic('title', kwargs)
    fct.add_in_dic('name_data', kwargs)
    # Title and name of the file with BDT    
    kwargs['name_file'], kwargs['title'] = fct.get_name_file_title_BDT(kwargs['name_file'], kwargs['title'], cut_BDT, 
                                                                       variable, fct.add_text(kwargs['name_data'],
                                                                                              'fit','_',None))
    
    
    # Name of the folder = name of the data
    fct.add_in_dic('name_folder', kwargs)
    fct.add_in_dic('name_data', kwargs)
    if kwargs['name_folder'] is None and kwargs['name_data'] is not None:
        kwargs['name_folder'] = kwargs['name_data']
    
    plot_hist_fit(df, variable, name_var=name_variable, unit_var=unit_var, **kwargs)