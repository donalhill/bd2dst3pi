"""
Anthony Correia
02/01/21
- Plot histograms
- Plot 2D histograms
- Plot histograms of the quotient of 2 variables
"""


import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pandas.core.series import Series


from matplotlib.colors import LogNorm #matplotlib.colors.LogNorm()

import plot.tool as pt
from load_save_data import add_in_dic

#Gives us nice LaTeX fonts in the plots
#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False



#################################################################################################
################################ subfunctions for plotting ######################################
################################################################################################# 



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
    
    bin_width = pt.get_bin_width(low,high,n_bins)
    
    if label_ncounts:
        if label is None:
            label = ""
        else:
            label += ": "
        label += f" {n_candidates} events"
        
    if density:
        counts = counts/(n_candidates*bin_width)
        err = err/(n_candidates*bin_width)
    
    if mode_hist:
        ax.bar(centres, counts,centres[1]-centres[0], color=color, alpha=alpha, edgecolor=None, label=label)
        ax.step(edges[1:],counts, color=color)
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
    unit_variable_text = pt.redefine_unit(unit_variable)
    name_data_text = pt.redefine_format_text(name_data, bracket = '(')
    
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
    
    
    label = f"{pre_label} / ({bin_width:.1g}{pt.redefine_unit(unit_variable, show_bracket=False)})"
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
    set_label_candidates_hist(ax, bin_width, pre_label = pre_label, unit_variable=unit_variable, 
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
    
    set_label_candidates_hist(ax, bin_width, pre_label = pre_label, 
                              unit_variable=unit_variable, 
                              fontsize=25, axis='y')
    
    

#################################################################################################
################################# Main plotting function ########################################
################################################################################################# 

def plot_hist(dfs, variable, name_variable=None, unit_variable=None, n_bins=100, mode_hist=False, 
              low=None, high=None, density=None, 
              title=None, name_data_title=False, label_ncounts=True,
              name_file=None,name_folder=None,colors=None, weights=None, save_fig=True,
              pos_text_LHC=None, ymax=None, show_leg=None, alpha=None):
    """ Save the histogram(s) of variable of the data given in dfs
    
    @dfs             :: Dictionnary {name of the dataframe : pandas dataframe, ...}
    @variable        :: str, name of the variable, in the dataframes
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
    @weights         :: weights passed to plt.hist 
    @save_fig        :: Bool, specifies is the figure is saved
    @alpha           :: transparancy of the histograms
    
    @returns         :: fig, ax
    """
    if not isinstance(dfs,dict):
        dfs = {"":dfs}
    
    if density is None:
        density = len(dfs)>1 # if there are more than 2 histograms
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    if isinstance(dfs,dict):
        name_datas = list(dfs.keys())
    
    if name_variable is None:
        name_variable = pt.latex_format(variable)
    
    if name_data_title:
        title = pt.add_text(pt.list_into_string(name_datas),title,' - ', default=None)
    
    ax.set_title(title, fontsize=25)
    
    #First loop to determine the low and high value
    low, high = pt.redefine_low_high(low,high,[df[variable] for df in dfs.values()])
    bin_width = pt.get_bin_width(low,high,n_bins)
    
    if colors is None:
        colors = ['r','b','g','k']
    if not isinstance(colors,list):
        colors = [colors]
    
    k_col = 0
    for name_data, df in dfs.items():
        if alpha is None:
            alpha = 0.5 if len(dfs)>1 else 1
        _,_,_,_ = plot_hist_alone(ax, df[variable], n_bins, low, high, colors[k_col], mode_hist, alpha = alpha, 
                        density = density, label = name_data, label_ncounts=label_ncounts, weights=weights)
        k_col += 1
              
              
    #Some plot style stuff
    if ymax is None:
        ymax=1+0.15*len(name_datas)
    if show_leg is None:
        show_leg = len(dfs)>1
    set_label_hist(ax, name_variable, unit_variable, bin_width, density=density, fontsize=25)
    pt.fix_plot(ax, ymax=ymax, show_leg=show_leg, pos_text_LHC=pos_text_LHC)
    
    #Remove any space not needed around the plot
    plt.tight_layout()
    
    if save_fig:
        pt.save_file(fig, name_file,name_folder,f'{variable}_{pt.list_into_string(name_datas)}')
    return fig, ax

    
def plot_hist_var (datas, variable, name_variable=None, unit_variable=None, name_datas=None, **kwargs):
    ''' plot the histogram(s) of data
    
    @datas      :: data or list of data (data = list of float)
    @variable   :: str, name of the variable (for the name of the saved file)
    @name_datas :: str or list of str, name of the datas
    @kwargs     :: passed to plot_hist
    @returns    :: fig, ax
    
    '''
    
    # data into list of data
    if not isinstance(datas[0], np.ndarray) and not isinstance(datas[0], Series):
        datas = [datas]
        
    name_datas = pt.el_to_list(name_datas, len(datas))
    assert len(name_datas) == len(datas)
    
    dfs = {}
    for data, name_data in zip(datas, name_datas):
        df = DataFrame()
        df[variable] = np.array(data)
        dfs[name_data] = df
    
    return plot_hist(dfs, variable, name_variable=name_variable, unit_variable=unit_variable, **kwargs)   

def plot_divide(dfs, variable, name_variable,unit_variable, n_bins=100, low=None, high=None, 
                name_file=None, name_folder=None, save_fig=True,
                pos_text_LHC=None):
    """
    plot the (histogram of the dataframe 1 of variable)/(histogram of the dataframe 1 of variable) after normalisation
        
    @dfs             :: Dictionnary of 2 couple (key:value) 
                            {name_dataframe_1 : pandas_dataframe_1, name_dataframe_2 : pandas_dataframe_2}
    @variable        :: str, name of the variable, in the dataframes
    @name_variable   :: Name of the variable that will be used in the labels of the plots
    @unit_variable   :: Unit of the variable
    @n_bins          :: Desired number of bins of the histogram
    @low             :: low value of the distribution
    @high            :: high value of the distribution
    @name_file       :: name of the plot that will be saved
    @name_folder     :: name of the folder where to save the plot
    @save_fig        :: Bool, specifies is the figure is saved
    
    @returns         :: fig, ax
    """
       
    fig, ax = plt.subplots(figsize=(8,6))
    name_datas = list(dfs.keys())
    
    
    # Compute the number of bins
    low, high = pt.redefine_low_high(low,high,[df[variable] for df in dfs.values()])
    bin_width = pt.get_bin_width(low,high,n_bins)
    
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
    pt.fix_plot(ax, ymax=1.1, show_leg=False, fontsize_ticks=20., ymin_to0=False, pos_text_LHC=pos_text_LHC)
    
    # Labels
    set_label_divided_hist(ax, name_variable, unit_variable, bin_width, names_data, fontsize=25)    

    

    
    #R emove any space not needed around the plot
    plt.tight_layout()
    plt.show()

    #Save the plot as a PDF document into our PLOTS folder (output/plots as defined in bd2dst3pi/locations.py)    
    if save_fig:
        pt.save_file(fig, name_file,name_folder,f"{variable}_{pt.list_into_string(name_datas,'_d_')}")
    
    return fig, ax


def plot_hist2d(df, variables, name_variables, unit_variables, n_bins = 100,
                low=None, high=None, 
                title=None, 
                name_file=None, name_folder=None,
                name_data=None, log_scale=False,
               save_fig=True, pos_text_LHC=None):
    '''  Plot a 2D histogram of 2 variables.
    @df                :: dataframe (only one)
    @variables         :: list of 2 str, variables in the dataframe
    @name_variables    :: list of 2 str, names of the variables 
    @unit_variables    :: str (common unit) or list of 2 str (units of variable[0] and variable[1])
    @n_bins            :: integer or list of 2 integers
    @low               :: float or list of 2 floats ; low  value(s) of variables
    @high              :: float or list of 2 floats ; high value(s) of variables
    @title             :: str, title of the figure
    
    @name_file         :: name of the plot that will be saved
    @name_folder       :: name of the folder where to save the plot
    @name_data         :: str, name of the data, this is isued to define the name of the plot,
                              in the case name_file is not defined.
    @log_scale         :: if true, the colorbar is in logscale
    @save_fig        :: Bool, specifies is the figure is saved
    
    @returns         :: fig, ax
    '''
    
    
    ## low, high and unit_variables into a list of size 2
    low = pt.el_to_list(low,2)
    high = pt.el_to_list(high,2)
    
    unit_variables = pt.el_to_list(unit_variables,2)
    
    for i in range(2):
        low[i],high[i] = pt.redefine_low_high(low[i],high[i], df[variables[i]])
    
    ## Plotting
    fig, ax = plt.subplots(figsize=(8,6))
    
    title = pt.add_text(name_data, title, default=None)
    
    ax.set_title(title, fontsize=25)
    
    if log_scale:
        _,_,_,h = ax.hist2d(df[variables[0]], df[variables[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins,norm=LogNorm())
    else:
        _,_,_,h = ax.hist2d(df[variables[0]], df[variables[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins)
    
    ## Label, color bar
    pt.set_label_ticks(ax)
    pt.set_text_LHCb(ax, pos = pos_text_LHC)
    
    set_label_2Dhist(ax, name_variables, unit_variables, fontsize=25)
    cbar = plt.colorbar(h)
    cbar.ax.tick_params(labelsize=20)
    
    ## Save the data
    if save_fig:
        pt.save_file(fig, name_file,name_folder,pt.add_text(pt.list_into_string(variables,'_vs_'),name_data,'_'))
    
    return fig, ax
    

#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################         


def plot_hist_particle(dfs, variable, cut_BDT=None, **kwargs):
    """ 
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    
    Then, plot histogram with plot_hist
    
    @dfs      :: Dictionnary of 2 couple (key:value) 
                            {name_dataframe_1 : pandas_dataframe_1, name_dataframe_2 : pandas_dataframe_2}
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variable, name_var, unit_var
    @returns  :: fig, ax
    """
    
    ## Retrieve particle name, and variable name and unit.
#     particle, var = retrieve_particle_variable(variable)
    
#     name_var = variables_params[var]['name']
#     unit_var = variables_params[var]['unit']
#     name_particle = particle_names[particle]
    
    name_variable, unit_var = pt.get_name_unit_particule_var(variable)
    name_datas = pt.list_into_string(list(dfs.keys()))
    
    add_in_dic('name_file', kwargs)
    add_in_dic('title', kwargs)
    kwargs['name_file'], kwargs['title'] = pt.get_name_file_title_BDT(kwargs['name_file'], kwargs['title'], 
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
    @returns  :: fig, ax
    """
    name_variable, unit_var = pt.get_name_unit_particule_var(variable)
    
    add_in_dic('name_folder', kwargs)
    if kwargs['name_folder'] is None:
        kwargs['name_folder'] = pt.list_into_string(list(dfs.keys()))
        
    return plot_divide(dfs, variable, name_variable, unit_var, **kwargs)
    
    
def plot_hist2d_particle(df, variables, **kwargs):
    """
    Retrieve name_variable, unit_variable and name_particle directly from variables.py.
    (in order not to have to type it every time)
    
    Then, plot 2d histogram with plot_hist2d.
    
    @df       :: pandas dataframe
    @variable :: str, variable (for instance: 'B0_M'), in dataframe
    @kwargs   :: arguments passed in plot_hist_fit (except variables, name_vars, unit_vars)
    @returns  :: fig, ax
    """
    name_variables = [None, None]
    unit_vars      = [None, None]
    for i in range(2):
        name_variables[i], unit_vars[i] = pt.get_name_unit_particule_var(variables[i])
    
    add_in_dic('name_folder', kwargs)
    add_in_dic('name_data', kwargs)
    
    if kwargs['name_folder'] is None :
        kwargs['name_folder'] = kwargs['name_data']
    
    return plot_hist2d(df, variables, name_variables=name_variables, unit_variables=unit_vars, **kwargs)