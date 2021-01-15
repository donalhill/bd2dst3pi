"""
Anthony Correia
02/01/21
- Compute the number of d.o.f. of a model
- Compute the reduced chi2 of a model
- plot the pull diagram ot a fit
- plot the histogram, with the fitted PDF and the fitted parameters and the pull diagram
"""

import plot.tool as pt
from plot.histogram import plot_hist_alone, set_label_hist
from load_save_data import add_in_dic, el_to_list

from zfit.core.parameter import ComposedParameter
from zfit.core.parameter import Parameter as SimpleParameter
from zfit.models.functor import SumPDF

#from zfit.core.parameter import Parameter

import numpy as np

from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False



#################################################################################################
######################################## Tools functions ########################################
#################################################################################################

### Computation ===============================================

def get_plot_scaling(counts, low, high, n_bins):
    """Return plot_scaling, the factor to scale the curve fit to unormalised histogram"""
    return counts.sum() *(high-low) / n_bins 
    

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

### number of degrees of freedom =============================

def get_n_dof_params_seen(results, n_dof=0):
    n_dof += results[0]
    params_seen = results[1]
    return n_dof, params_seen

def count_n_dof_params_recurs(params, params_seen=[]):
    n_dof = 0
    for param in params.values():
        name_param = param.name
        if name_param not in params_seen :
            params_seen.append(name_param)
            if isinstance(param, ComposedParameter):
                n_dof, params_seen = get_n_dof_params_seen(
                    count_n_dof_params_recurs(param.params, params_seen), n_dof)
            else:
                if param.floating:
                    n_dof+=1
    return n_dof, params_seen

def count_n_dof_model_recurs(model, params_seen=[]):
    n_dof = 0
    
    # Explore the parameters of model
    n_dof, params_seen = get_n_dof_params_seen(
        count_n_dof_params_recurs(model.params, params_seen), n_dof)

    
    
    # Explore the parameters of the submodels of model
    if isinstance(model, SumPDF):
        for submodel in model.models:
            n_dof, params_seen = get_n_dof_params_seen(
                count_n_dof_model_recurs(submodel, params_seen), n_dof)
    
    return n_dof, params_seen

def count_n_dof_model(model):
    n_dof, _ = count_n_dof_model_recurs(model, params_seen=[])
    return n_dof


def get_chi2(fit_counts, counts):
    """ chi2 of the fit
    
    @fit_counts :: fitted number of counts
    @counts     :: number of counts in the histogram of the fitted data
    
    @returns :: float, chi2 of the fit
    """
    diff = np.square(fit_counts-counts)
    
    #n_bins = len(counts)
    diff = np.divide(diff, np.abs(counts), out=np.zeros_like(diff), where=counts!=0)
    chi2 = np.sum(diff) # sigma_i^2 = mu_i
    return chi2

def reduced_chi2(fit_counts, counts, ndof):
    n_bins = np.abs(len(counts))
    return get_chi2(fit_counts, counts)/(n_bins - ndof)

#################################################################################################
#################################### Sub-plotting functions #####################################
#################################################################################################
    
    
def plot_pull_diagram(ax, model, counts, edges, centres, err, low=None, high=None, line=3, mode_hist_pull=True,
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
    if mode_hist_pull:
        ax.bar(centres, pull, centres[1]-centres[0], color=color, edgecolor=None)
        ax.step(edges[1:], pull, color=color)
    else:
        ax.errorbar(centres,pull, yerr = np.ones(len(centres)),color=color, ls='', marker='.')
    ax.plot([low,high],[line, line],color='r',ls='--')
    ax.plot([low,high],[-line, -line],color='r',ls='--')
    ax.plot([low,high],[0,0],color='r')
    
    ## Symmetric pull diagram
    low_y, high_y = ax.get_ylim() 
    maxi = max(4., -low_y, high_y) 
    low_y = -maxi
    high_y = maxi
    
    ax.set_ylim([low_y, high_y])
    
    ## Label and ticks
    ax.set_ylabel('residuals / $\\sigma$', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(3.))
    ax.yaxis.set_minor_locator(MultipleLocator(1.))
    
    pt.show_grid(ax, which='minor', axis='y')
    
    ax.set_xlim([low,high]) 
    
    ndof = count_n_dof_model(model)
    
    ## Fit quality
    print(f"Number of bins: {len(centres)}")
    print(f"Width of the bins: {centres[1]-centres[0]}")
    print("")
    chi2 = reduced_chi2(fit, counts, ndof)
    print("Number of d.o.f. in the model: ", ndof)
    print('Reduced chi2: ', chi2)
    print("")
    print(f"Mean of the normalised residuals: {np.mean(pull[np.isfinite(pull)])}")
    print(f"Std of the normalised residuals: {np.std(pull[np.isfinite(pull)])}")
    
    if show_chi2:
        ax.set_xlabel(f'(reduced $\\chi^2$={chi2:.2f})', fontsize=fontsize)
    
def plot_fitted_curve(ax, model, plot_scaling, frac=None, line_width=2.5,
                      color='b',linestyle='-', low=None, high=None, label=None, x=None, alpha=1):
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
    ax.plot(x, y, linewidth=line_width, color=color, ls=linestyle, label=label, alpha=alpha)


# Alternative names for the models
name_PDF = {
    'DoubleCB': 'Double CB',
    'SumPDF'  : 'Sum',
    'Exponential': 'Exponential',
    'Gauss' : 'Gaussian',
    'CrystalBall': 'Crystal Ball'
}

def number_events_model(models):
    """ return the frac or the total number of events in the model
    @models  :: list of zFit models, whose first element is the sum of the others, weighted by frac or by extended models.
    
    @returns  ::
        - If frac, is used, it directly returns frac
        - Else, it returns the number of events in the model
    
    NB: this functions assumes that there is only 1 frac (I don't need more that 1 fracs yet)
    """
    
    model=models[0]
    assert isinstance(model, SumPDF)

    mode_frac = False
    parameters = list(model.params.values())
    i = 0
    while not mode_frac and i<len(parameters):
        # if one of the parameter is not a ComposedParameter, this it is a frac parameter
        mode_frac = not isinstance(parameters[i],ComposedParameter)
        if mode_frac:
            assert isinstance(parameters[i], SimpleParameter)
        i+= 1
    if mode_frac:
        # parameters[i-1] is a SimpleParameter, i.e., frac
        frac = float(parameters[i-1])
        return frac, mode_frac
    else:
        n_tot = 0
        for sub_model in model.models:
            assert sub_model.is_extended
            n_tot += float(sub_model.get_yield().value())
        return n_tot, mode_frac

        
name_type_models = {
    'm' : 'model',
    's' : 'signal',
    'b' : 'background'
}

def get_name_model(model):
    """ return the name of the model
    @model  :: zfit PDF
    
    @returns :: name of the model (specified by the dictionnary name_PDF)
    """
    # get the name of the model, removing  '_extended' when the PDF is extended
    marker = model.name.find('_') 
    if marker == -1:
        marker = None
    label_model = name_PDF[model.name[:marker]]
    
    return label_model

def get_element_list(liste_candidate, index, if_not_list=None):
    """ return an element of list, but process the case where liste is not a list
    @liste       :: python element
    @index       :: integer
    @if_not_list :: returned element if liste is not a list:
            if_not_list=None  : return None
            if_not_list='el' : return the liste_candidate
    @results: if liste is a liste, return liste_candidate[index], 
                else return according to what is specified is if_not_list
    """
    if isinstance(liste_candidate, list):
        return liste_candidate[index]
    else:
        if if_not_list is None:
            return None
        elif if_not_list == "el":
            return liste_candidate

def plot_models(ax, x, models, plot_scaling, type_models, name_models=None, 
                frac=1., l=0, colors=['b', 'g', 'gold', 'magenta', 'orange'], 
                linestyles=['-', '--', ':', '-.'], line_width=2.5):
    """ Plot the models recursively
    @ax          :: axis where to plot
    @models      :: list of models or list of list of models
            * just one PDF
            * a list of PDFs, whose first PDF is the composite PDF and the other ones are their components
            * list of list of PDFs, if composite of composite of PDFs
            * ... (recursive)
    @plot_scaling:: float, scaling to get the scale of the curve right
    @type_models :: str, type of each mode (one character for each model):
                      - 'm' : model (sum) --> should always be the FIRST ONE !!
                      - 's' : signal
                      - 'b' : background
                      used in the legend
    @name_models :: name of the models - used in the legend
            * if there is only one model, just a str
            * if there is a list of PDFs,
                    * list of the same size as name_models to give a different to each
                    * only one str, name of the first composite PDFs. The other PDFs are plotted but they aren't legended
            * ... (recursive)
    @frac        :: float, frac is multiplied to the PDF to get the correct scale due to composite PDFs
    @colors      :: list of colors, same structure as name_models
    @linestyles  :: list of linestyles to use, the one among them to use is specified by 'l'
    @l           :: =0 is first sumPDF, 1 if component of this sumPDF, 2 if component of a sumPDF component of sumPDF
    @line_width  :: float, width of the plotted lines
    """             
        
    if isinstance(models, list):
        if len(models)>1:
            frac_or_ntot, mode_frac = number_events_model(models)
        else:
            mode_frac=None
        if mode_frac:
            assert len(models)==3

        for k, model in enumerate(models):
            if k==1:
                l+=1
            # Compute frac
            applied_frac = frac
            if mode_frac is not None:
                if mode_frac:
                    if k == 1:
                        applied_frac = frac * frac_or_ntot
                    elif k == 2 :
                        applied_frac = frac * (1 - frac_or_ntot)
                else:
                    #frac_or_ntot is ntot
                    if k>=1:
                        main_model = get_element_list(model,0,if_not_list='el')
                        applied_frac = frac * float(main_model.get_yield().value()) / frac_or_ntot
            # labels
            if len(type_models)>1:
                type_model = type_models[k]
            else:
                type_model = type_models

            color = get_element_list(colors, k, if_not_list='el')
            if not isinstance(name_models,list): 
                # if the name of the subsubmodel is not specified, put it to None
                if k==0:
                    name_model = name_models
                else:
                    name_model = None
            else:
                name_model = name_models[k]


            plot_models(ax, x, model, plot_scaling, type_model, name_model, applied_frac, l, color, 
                        linestyles, line_width)
    
    else:
        model = get_element_list(models,0,if_not_list='el')
        assert not isinstance(model,list)
        

        # Label
        if name_models is not None:
            label_model = get_name_model(model)
            label_model += f' - {name_type_models[type_models]}'
            label_model = pt.add_text(label_model, name_models)
        else:
            label_model=None
        if l>=2:
            alpha = 0.5
        else:
            alpha = 1

        plot_fitted_curve(ax, model, plot_scaling, frac=frac, line_width=line_width, color=colors, 
                          linestyle = linestyles[l], label=label_model, x=x, alpha=alpha)
        
        
    

def plot_fitted_curves(ax, models, plot_scaling, low, high, name_models=None, type_models=None,
                       line_width=2.5, colors=['b', 'g', 'gold', 'magenta', 'orange'],
                       fontsize_legend=16, loc_leg='upper left', show_legend=None):
    """
    Plot fitted curve given by 'models', with labels given by name_models
    
    @ax            :: axis where to plot
    @models        :: zfit model of list of zfit models
    @plot_scaling  :: scaling to get the scale of the curve right
    @low           :: low limit of the plot (x-axis)
    @high          :: high limit of the plot (x-axis)
    @name_models, type_models, line_width, colors passed to plot_models
    @fontsize      :: fontsize of the legend
    """
    models = el_to_list(models, 1)
    if show_legend is None:
        show_legend = name_models is not None
    name_models = el_to_list(name_models, len(models))

    x = np.linspace(low, high, 1000)
    
    # Plot the models
    if type_models is None:
        type_models = 'm'
        if len(name_models) >= 2:
            type_models += 's'
            type_models += 'b'*(len(name_models)-2)

    plot_models(ax, x, models, plot_scaling, type_models, name_models=name_models, colors=colors, line_width=2.5)
    

    
    if show_legend:
        ax.legend(fontsize=fontsize_legend, loc=loc_leg)

def plot_result_fit(ax, params, name_params=None, fontsize=20, colWidths=[0.06,0.01,0.05,0.06], loc='upper right'):
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
        
        # if name_param as ';', remove everything from ';' onwards
        index = name_param.find(';')
        if index!=-1:
            name_param = name_param[:index]
        
        # if name_params not None, it specifies the variables we want to show
        if (name_params is None) or (name_param in name_params): 
            # Retrieve value and error
            value_param = params[p]['value']
            error_param = params[p]['minuit_hesse']['error']
            
            # Retrieve alt name
            if name_params is not None:
                surname_param = name_params[name_param]
            else:
                surname_param = pt.latex_format(name_param)

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



#################################################################################################
###################################### Plotting function ########################################
#################################################################################################   
    
def plot_hist_fit (df, variable, name_var=None, unit_var=None,models=None, obs=None, n_bins=50, color='black', 
                  name_models=None, type_models=None, low_hist=None, high_hist=None,
                  mode_hist=True, linewidth=2.5, colors=None,
                  name_data_title=False, title=None, fontsize_leg=20,
                  name_file=None,name_folder=None,name_data=None, show_chi2=False,
                  params=None,name_params=None, colWidths=[0.04,0.01,0.06,0.06], fontsize_res=20.,
                  loc_res='upper right', loc_leg='upper left',
                  weights=None, save_fig=True, pos_text_LHC=None, show_leg=None,
                  plot_pull=True, mode_hist_pull=True):
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
    fontsize_res   :: float, fontsize of the text in the result table
    @loc_res       :: str position of the result table, loc argument specified in in plt.table 
    @loc_leg       :: str position of the legend, loc argument specified by loc in plt.legend
    @weights       :: weights passed to plt.hist  
    
    @returns       :: fig, axs
    """
    
    ## Create figure
    if plot_pull:
        fig = plt.figure(figsize=(12,10))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        ax = [plt.subplot(gs[i]) for i in range(2)]
    else:
        fig, ax = plt.subplots(figsize=(8,6))
        ax = [ax]    
    
    
    ## Retrieve low,high (of x-axis)
    low = float(obs.limits[0])
    high = float(obs.limits[1])
    
    if low_hist is None:
        low_hist = low
    if high_hist is None:
        high_hist = high
    
    if name_var is None:
        name_var = pt.latex_format(variable)
    
    if name_data_title and name_data is not None:
        title = pt.add_text(title, name_data, 'before')
    
    ax[0].set_title(title, fontsize=25)
    
    ## plot 1D histogram of data
    # Histogram 
    counts, edges, centres, err = plot_hist_alone(ax[0], df[variable], n_bins,
                                               low_hist, high_hist, color, mode_hist, alpha = 0.1, weights=weights)
    
    
    # Label
    bin_width = pt.get_bin_width(low_hist, high_hist, n_bins)
    set_label_hist(ax[0], name_var, unit_var, bin_width, fontsize=25)
    
    # Ticks
    pt.set_label_ticks(ax[0])
    pt.set_text_LHCb(ax[0], pos=pos_text_LHC)
    
    ## Plot fitted curve 
    if isinstance(models,list):
        model = models[0] # the first model is the "global" one
    else:
        model = models
    
        
    plot_scaling = get_plot_scaling(counts, low_hist, high_hist, n_bins)
    plot_fitted_curves(ax[0], models, plot_scaling, low, high, name_models=name_models, type_models=type_models,
                       line_width=2.5, colors=colors, fontsize_legend=fontsize_leg, loc_leg=loc_leg, show_legend=show_leg)
    
    pt.change_ymax(ax[0], factor=1.1)    
    
    color_pull = colors if not isinstance(colors, list) else colors[0]
    ## Plot pull histogram
    if plot_pull:
        plot_pull_diagram(ax[1], model, counts, edges, centres, err, color=color_pull,
                          low=low, high=high, plot_scaling=plot_scaling, show_chi2=show_chi2, mode_hist_pull=mode_hist_pull)
    
    
    ## Plot the fitted parameters of the fit
    if params is not None:
        plot_result_fit(ax[0], params, name_params=name_params, fontsize=fontsize_res, colWidths=colWidths, loc=loc_res)
    
    # Save result
    plt.tight_layout()
    if save_fig:
        pt.save_file(fig, name_file,name_folder,f'{variable}_{name_data}_fit')
        
    if plot_pull:
        return fig, ax[0], ax[1]
    else:
        return fig, ax[0]

def plot_hist_fit_var (data, variable, name_var=None, unit_var=None, **kwargs):
    ''' plot data with his fit
    @data      :: pandas Series or numpy ndarray
    @variable  :: str, name of the variable, for the name of the file
    @name_var  :: str, name of the variable, for the label/legend
    @unit_var  :: str, unit of the variable
    @kwargs    :: parameters passed to plot_hist_fit
    '''
    df = DataFrame()
    df[variable] = data
    
    return plot_hist_fit(df, variable, name_var, unit_var, **kwargs) 
    



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
    
    @returns  :: fig, axs
    """
    
    ## Retrieve particle name, and variable name and unit.
#     particle, var = retrieve_particle_variable(variable)
    
#     name_var = variables_params[var]['name']
#     unit_var = variables_params[var]['unit']
#     name_particle = particle_names[particle]
    
    
    
    name_variable, unit_var = pt.get_name_unit_particule_var(variable)
       
    add_in_dic('name_file', kwargs)
    add_in_dic('title', kwargs)
    add_in_dic('name_data', kwargs)
    # Title and name of the file with BDT    
    kwargs['name_file'], kwargs['title'] = pt.get_name_file_title_BDT(kwargs['name_file'], kwargs['title'], cut_BDT, 
                                                                      variable, pt.add_text(kwargs['name_data'],
                                                                                            'fit','_',None))
    
    
    # Name of the folder = name of the data
    add_in_dic('name_folder', kwargs)
    add_in_dic('name_data', kwargs)
    if kwargs['name_folder'] is None and kwargs['name_data'] is not None:
        kwargs['name_folder'] = kwargs['name_data']
    
    return plot_hist_fit(df, variable, name_var=name_variable, unit_var=unit_var, **kwargs)