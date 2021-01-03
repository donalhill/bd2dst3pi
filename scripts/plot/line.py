"""
Anthony Correia
02/01/21
- plot x vs y
- plot x vs y1, y2, y3, ... (several curves with the same x)
"""

from . import tool as pt

from uncertainties import unumpy
import numpy as np
import matplotlib.pyplot as plt

#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False


#################################################################################################
######################################## Tool function ##########################################
################################################################################################# 

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

def add_value_labels(ax, lx, ly, labels, space_x=-10, space_y=5, labelsize=12):
    """ Annotate points with a text whose distance to the point is specified by (space_x, space_y)
    
    @ax          :: axis where to plot
    @lx          :: list of float, abcissa of the points
    @ly          :: list of float, ordinate of the points
    @labels      :: list of str, annotation of the specified points
    @space_x     :: float, space in pixel from the point to the annotation text, projected in the x-axis
    @space_y     :: float, space in pixel from the point to the annotation text, projected in the y-axis
    @labelsize   :: float, fontsize of the annotation
    """  
    assert len(lx)==len(ly)
    assert len(labels)==len(lx)

    # For each bar: Place a label
    for x, y, label in zip(lx, ly, labels):
        # Vertical alignment for positive values
        ha = 'center'
        va = 'center'       
        if x!=0 and y!=0:    
            ax.annotate(
                label,          
                (x, y), #xycoords='data',         
                xytext=(space_x, space_y),          # Vertically shift label by `space`
                textcoords='offset pixels', # Interpret `xytext` as offset in points
                va=va, ha=ha,
                size = labelsize)

#################################################################################################
###################################### Plotting function ########################################
#################################################################################################  

def plot_xys (ax, x, ly, xlabel, labels=None, colors=['b','g','r','y'], fontsize=25, markersize=1,
             linewidth=1., linestyle='-', factor_ymax=1.,
              annotations=None, fontsize_annot=15., space_x=-15, space_y=5, pos_text_LHC=None):
    """
    @ax               :: axis where to plot
    @x                :: list of float, points of the x-axis
    @ly               :: list of list of float, list of the y-points of the curves
    @labels           :: labels of the curves
    @xlabel           :: str, xlabel
    @ylabel           :: str, ylabel
    @colors           :: list of str, colors of each curve in ly
    @fontsize         :: fontsize of the labels
    @markersize       :: size of the markers
    @linestyle        :: linestyle of the plotted curves
    @linewidth        :: linewidth of the plotted curves
    @fontsize_annot   :: float, fontsize of the annotations
    @fontsize         :: fontsize of the labels
    @space_x          :: float, space in pixel from the point to the annotation text, projected in the x-axis
    @space_y          :: float, space in pixel from the point to the annotation text, projected in the y-axis
    """
    colors = pt.el_to_list(colors, len(ly))
    
    plot_legend = False
    
    for i, y in enumerate(ly):
        label = labels[i] if len(ly) > 1 else None
        x = np.array(x)
        y = np.array(y)
        x_n = unumpy.nominal_values(x)
        y_n = unumpy.nominal_values(y)
        ax.errorbar(x_n, y_n, 
                    xerr = unumpy.std_devs(x), yerr=unumpy.std_devs(y), 
                    linestyle=linestyle, color=colors[i], 
                    markersize=markersize, elinewidth=markersize,
                    linewidth=linewidth, label=label, marker='.')
        
        if label is not None:
            plot_legend = True
        
    ax.set_xlabel(xlabel, fontsize=25)
    
    if len(ly)==1:
        ax.set_ylabel(labels[0], fontsize=25)
    else:
        ax.set_ylabel('value', fontsize=25)
    
    # Grid
    pt.show_grid(ax, which='major')
    pt.show_grid(ax, which='minor')
        
    # Ticks
    pt.fix_plot(ax, ymax=factor_ymax, show_leg=plot_legend, fontsize_leg=25, ymin_to0=False, pos_text_LHC=pos_text_LHC)    
    
    
    if annotations is not None:
        assert len(ly)==1
        add_value_labels(ax, x_n, y_n, annotations, labelsize=fontsize_annot, space_x=space_x, space_y=space_y)

def plot_x_list_ys(x, y, name_x, names_y, surname_x=None, surnames_y=None, 
                   annotations=None, markersize=1,
                   linewidth=1.,fontsize=25, name_file=None, name_folder=None,
                   factor_ymax=1., linestyle='-', fontsize_annot=15.,
                   space_x=-15, space_y=5, log_scale=None, save_fig=True, pos_text_LHC=None):
    """ plot x as a function of the y of the list l_y
    
    @x          :: list or array of floats, points in the x-axis
    @y          :: list of list of numpy arrays of ufloat, list of numpy arrays of ufloat, or one numpy array of ufloat
    @name_x     :: str, name of the list used for the saving
    @name_y     :: list of str or str, name of each list in l_y - use for the saved filename
    @name_x     :: str or list of strs - use for the saved file name
    @surname_y  :: list of list of str, list of str or str, surname of each list in l_y - use for the label
    @colors     :: list of list of str, list of str or str, colors of each graph in l_y 
    @linewidth  :: float, linewdith
    @name_file  :: str, name of the file to save
    @name_folder:: str, name of the folder where the image is saved
    @factor_ymax:: float, ymax is multiplied by factor_ymax
    @log_scale  :: 'both', 'x' ot 'y', specify which axis is wanted to be in log scale
    @linestyle, fontsize_annot, space_x, space_y --> passed to plot_xys
    
    @returns    :: fig, axs
    """
    
    if surname_x is None:
        surname_x = name_x
            
    groups_ly         = el_or_list_to_group(y)
    groups_names_y    = el_or_list_to_group(names_y, str)
    
    
    
    if surnames_y is not None:
        groups_surnames_y = el_or_list_to_group(surnames_y, str)
    else:
        groups_surnames_y = groups_names_y
    
    
    fig, axs = plt.subplots(len(groups_ly),1, figsize=(8,4*len(groups_ly)))
    
    for k, ly in enumerate(groups_ly):
        if len(groups_ly)==1:
            ax = axs
        else:
            ax = axs[k]
        
        # In the same groups_ly, we plot the curves in the same plot
        plot_xys (ax, x, ly, xlabel=name_x, labels=groups_surnames_y[k], 
                  annotations=annotations, markersize=markersize,
                  fontsize=fontsize, linewidth=linewidth, linestyle=linestyle, factor_ymax=factor_ymax,
                  fontsize_annot=fontsize_annot, space_x=space_x, space_y=space_y, pos_text_LHC=pos_text_LHC)
        
        pt.set_log_scale(ax, axis=log_scale)
        
        
    plt.tight_layout()
    plt.show()
    if save_fig:
        pt.save_file(fig, name_file, name_folder, f'{name_x}_vs_{pt.list_into_string(pt.flattenlist2D(names_y))}')
    
    return fig, axs