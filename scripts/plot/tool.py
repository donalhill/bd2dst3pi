"""
Anthony Correia
02/01/21
- Save a plot
- Some functions to change how the plot looks (grid, logscale, label of the ticks, text ['LHCb preliminary']
- Functions which, from the root variable name, gets the unit, name of the variable and particle (from variable.py)
    which is used for the label/legend of the plots in plot/histogram.py, plot/fit.py and plot/line.py
"""

from bd2dst3pi.locations import loc

import numpy as np
from pandas.core.series import Series

import os.path as op
from os import makedirs

from variables import variables_params, particle_names, functions, name_variables_functions
from load_save_data import create_directory, add_in_dic, el_to_list
from copy import deepcopy



#################################################################################################
########################################### Saving ##############################################
################################################################################################# 

## FORMATING ===================================================

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

# SAVING FIGURE ================================================

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

    name_file = remove_space(remove_latex(name_file)).replace('/','d')
    path = op.join(directory, f"{name_file}")
    #Save the plot as a PDF document into our PLOTS folder (output/plots as defined in bd2dst3pi/locations.py)
    fig.savefig(path + '.pdf', dpi=600, bbox_inches="tight")
    fig.savefig(path + '.jpg', dpi=600, bbox_inches="tight")
    print(path)

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
    
### Core of the plot -------------------------------------------------------------------
def show_grid(ax, which='major', axis='both'):
    """show grid
    @ax    :: axis where to show the grid
    which  :: 'major' or 'minor'
    """
    ax.grid(b=True, axis=axis, which=which, color='#666666', linestyle='-', alpha = 0.2)

def change_ymax(ax, factor=1.1, ymin_to0=True):
    """ multiple ymax of the plot by factor
    @ax        :: axis where to plot
    @factor    :: float, factor by which ymax is multiplied
    @ymin_to0  :: bool, if True, ymin is set at 0
    """
    ymin, ymax = ax.get_ylim()
    if ymin_to0:
        ymin=0
    ax.set_ylim(ymin,ymax*factor)

def change_max(ax, factor=1.1, min_to0=True, axis='y'):
    """ multiple ymax of the plot by factor
    @ax        :: axis where to plot
    @factor    :: float, factor by which ymax is multiplied
    @ymin_to0  :: bool, if True, ymin is set at 0
    """
    if axis=='x' or axis=='both':
        xmin, xmax = ax.get_xlim()
        if min_to0:
            xmin=0
        ax.set_xlim(xmin,xmax*factor)
    
    if axis=='y' or axis=='both':
        ymin, ymax = ax.get_ylim()
        if min_to0:
            ymin=0
        ax.set_ylim(ymin,ymax*factor)
    
    
def set_label_ticks(ax, labelsize=20):
    """Set label ticks to size given by labelsize"""
    ax.tick_params(axis='both', which='both', labelsize=20)


def fix_plot(ax, ymax=1.1, show_leg=True, fontsize_ticks=20., fontsize_leg=20., loc_leg='best', ymin_to0=True, pos_text_LHC=None, axis='y'):
    """ Some fixing of the parameters (fontsize, ymax, legend)
    @ax              :: axis where to plot
    @ymax            :: float, multiplicative factor of ymax
    @show_leg        :: Bool, True if show legend
    @fontsize_ticks  :: float, fontsize of the ticks
    @fontsize_leg    :: int, fontsize of the legend
    @ymin_to0        :: bool, if True, ymin = 0
    pos_text_LHC     :: pos, passed to set_text_LHCb 
    """
    
    if ymax is not None:
        change_max(ax,ymax, ymin_to0, axis)
    
    set_label_ticks(ax)
    if show_leg:
        ax.legend(fontsize = fontsize_leg, loc=loc_leg)
    
    set_text_LHCb(ax, pos=pos_text_LHC)

def set_log_scale(ax, axis='both'):
    """Set label ticks to size given by labelsize
    @ax    :: axis where to change the scale into a log one
    @axis  :: 'both, 'x' or 'y', axis with a log scale
    """
    if axis == 'both' or axis == 'x':
        ax.set_xscale('log')
    if axis == 'both' or axis == 'y':
        ax.set_yscale('log')

def set_text_LHCb(ax, text='LHCb preliminary \n 2 fb$^{-1}$', fontsize=25., pos=None):
    """ Put a text on a plot 
    
    @ax       :: axis where to plot
    @text     :: str, text to plot
    @fontsize :: float, fontsize of the text
    @pos      ::
    
    @returns  :: the text element that plt.text return
    """
    if pos is not None:
        info = deepcopy(pos)
        if isinstance(pos, dict):
            ha = info['ha']
            if ha=='left' :
                x = 0.02 if 'x' not in info else info['x']
                y = 0.95 if 'y' not in info else info['y']
            elif ha=='right':
                x = 0.98 if 'x' not in info else info['x']
                y = 0.95 if 'y' not in info else info['y']
            
            add_in_dic('fontsize', pos, fontsize)
            add_in_dic('text', pos, text)
            
            fontsize = pos['fontsize']
            text = pos['text']
            
        elif isinstance(pos, str):
            if pos=='left':
                x = 0.02
                y = 0.95
                ha = 'left'
            elif pos=='right':
                x = 0.98
                y = 0.95
            ha = 'right'
        elif isinstance(pos, list):
            x = pos[0]
            y = pos[1]
            ha = pos[2]
    
        return ax.text(x, y, text, verticalalignment='top', horizontalalignment=ha, 
                    transform=ax.transAxes, fontsize=fontsize)

    
    
#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################         

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
    
    ## Get the name of the particle and deduce the name of the var
    particle = variable
    marker = None
    marker_before = 0
    
    # We must have: variable = particle_var
    # As long as particle is not in the list of the name of particles
    # get the next '_' (starting from the end)
    # cut variable in before '_' and after '_'
    # What is before should be the name of the variable
    # unless we need to select the next '_' to get the name of the particle
    # because this '_' is part of the name of the var
    while particle not in list_particles and marker != marker_before:
        marker_before = marker
        cut_variable = variable[:marker]
        marker = len(cut_variable)-1-cut_variable[::-1].find('_') # find the last '_'
        particle = variable[:marker]
    
    # if there were a '_' in variable, we assume the separation was done
    if marker != marker_before:
        var = variable[marker+1:]
        return particle, var
    
    # if we did not find a particle and var
    else:
        return None, None

def get_name_unit_from_particle_var(particle, var, variable, get_particle=False):
    '''
    @particle     :: str/None, name of the particle
    @var          :: str/None, name of the variable
    @variable     :: str, name of the full variable (particle_var in general)
    @get_particle :: Bool, if True, return (particle, var)
    '''
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
            name_variable = latex_format(var)
        else:
            if get_particle:
                name_variable = name_var
            else:
                name_variable = f"{name_var}({name_particle})"
    else:
        name_variable = variable
        unit_var = None
    
    if get_particle:
        return name_particle, name_variable, unit_var
    else:
        return name_variable, unit_var



def get_name_unit_particule_var(variable, get_particle=False):
    """ return the name of particle, and the name and unit of the variable
    @variable     :: str, variable (for instance: 'B0_M')
    @get_particle :: Bool, if True, return the name of the particle  and of the var separetately
    @returns    :: if get_particle is False
                    - str, name of the variable (for instance, 'm($D^{*-}3\pi$)')
                    - str, unit of the variable (for instance, 'MeV/$c^2$')
                   if get_particle is True
                    - str, name of the particle (for instance, 'm($D^{*-}3\pi$)')
                    - str, name of the var (for instance, 'm')
                    - str, unit of the var (for instance, 'MeV/$c^2$')
    """
    
    # If the variable is expressed in a function, there must be a ':'
    index_underscore = variable.find(':')
    variable_with_function = (index_underscore != -1)
    
    have_predefinite_name = False
    
    if variable_with_function:
        
        full_variable = variable
        variable = full_variable[:index_underscore]
        name_function = full_variable[index_underscore+1:]
        
        ## GET ALL THE VARIABLES ==============================================================
        no_comma = False
        variables = [variable]
        while not no_comma:
            variable = variables[-1]
            index_comma = variable.find(',')
            no_comma = (index_comma==-1)
            if not no_comma: # if there is a comma, it means there is more than 1 variable
                variables[-1] = variable[:index_comma]
                variables.append(variable[index_comma+1:])
        
        ## FOR ALL THE VARS, SEPARATE PARTICLE AND VAR ========================================
        particles = [None]*len(variables)
        varis = [None]*len(variables)
        all_particles_same = True # True if there is only one variable
        for i, variable in enumerate(variables):
            particles[i], varis[i] =  retrieve_particle_variable(variables[i])
            # check if all the particles are the same
            if i > 0:
                all_particles_same = all_particles_same and (particles[i] == particles[i-1])
        
        
        ## IF THE PARTICLE ARE THE SAME, CHECK IF var1,var2:function HAS A SURNAME ============
        if all_particles_same:
            
            particle = particles[0]
            var = list_into_string(varis, ',') + ':' + name_function

            # at this stage:
            # var = var1,var2:function
            have_predefinite_name = (var in variables_params)
            if all_particles_same:
                if get_particle:
                    name_particle, name_variable, unit_var = get_name_unit_from_particle_var(particle, var, variable, 
                                                                          get_particle=get_particle)
                else:
                    name_variable, unit_var = get_name_unit_from_particle_var(particle, var, variable, 
                                                                          get_particle=get_particle)
            
        # all_particles_same = False is the var cannot be found in the list of known variables
        # or if the particles aren't the same
        
        ## IF THE PARTICLES AREN'T THE SAME OR IT DOES NOT HAVE A PREDEFINITE NAME ================================================
        if not all_particles_same or not have_predefinite_name:
            if name_function in name_variables_functions:
                var = name_variables_functions[name_function](variables)
                
            else:
                name_variables = [None]*len(varis)
                for i in range(len(varis)):
                    if get_particle or have_predefinite_name:
                        name_particle, name_variables[i], _ = get_name_unit_from_particle_var(particles[i], varis[i], variable,
                                                                           get_particle=True)
                    else:
                        name_particle = False
                        name_variables[i], _ = get_name_unit_from_particle_var(particles[i], varis[i], variable,
                                                                           get_particle=False)
                variables_text = list_into_string(name_variables, ' and the ')
                if variables_text[0] != '$' and variables_text[1].islower(): 
                    #if not '$' at the beg, and the 2nd character is not in upper case (ex: 'DIRA angle')
                    variables_text = variables_text[0].lower() + variables_text[1:]
                name_variable = f"${name_function}$" + ' of the ' + variables_text
                unit_var = None
        
    if not variable_with_function:
        
        particle, var = retrieve_particle_variable(variable)
        if get_particle:
            name_particle, name_variable, unit_var = get_name_unit_from_particle_var(particle, var, variable, 
                                                                  get_particle=get_particle)
        else:
            name_variable, unit_var = get_name_unit_from_particle_var(particle, var, variable, 
                                                                  get_particle=get_particle)
    if get_particle:
        return name_particle, name_variable, unit_var
    else:
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
        name_file = add_text(variable, name_data, '_')
    
    # Title with BDT
    if cut_BDT is not None:
        title = add_text(title, f"BDT $>$ {cut_BDT}", ' - ')
        name_file = add_text(name_file, f'BDT{cut_BDT}')
    
    return name_file, title

