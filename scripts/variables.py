"""
Anthony Correia
02/01/21
Some global variables
- name and unit of root variables
- name of particles, specified in the root files
- Name of fitted parameters
"""


# Mass of the Dst meson
Dst_M_PDG = 2010.26


# Inside a saved root file, whose name is specified by the key of "data_tree_names"
# there is a tree, whose name is specified by the value of "data_tree_names"
# I have sometimes chosen different names for the trees, which are stored in this dictionnary
data_tree_names = {
    'all_common'                                  : 'DecayTreeTuple/DecayTree',
    'common_adaboost_without_P_cutDeltaM_highB0M' : 'DecayTree',
    'common_gradient_bkgHighB0M'                  : 'DecayTree',
    'common_B0toDstDs'                            : 'DecayTree',
    'Ds23pi_bkg_high_B0M'                         : 'DecayTree',
    'common_Dstto3pi'                             : 'DecayTree',
    'common_Dstto3pi_sWeights'                    : 'sWeights',
    'all_data_strip'                              : 'all_data_strip_cutDeltaM',
    'data_strip'                                  : 'data_strip_cutDeltaM_cutPID'
}

## INFORMATION ABOUT THE VARIABLES/PARTICLES ===================================================


# Name and unit of some variables
variables_params = {
    'P' : {
        'name': "p",
        'unit': "MeV/c"
    },
    'PT': 
    {
        'name': "$p_T$",
        'unit': "MeV/c"
    },
    'M': 
    {
        'name': "$m$",
        'unit': "MeV/$c^2$"
    },
    'flight_z':
    {
        'name': "$\Delta z$",
        'unit': "mm"
    },
    'flight_zsig':
    {
        'name': "$\Delta z$ significance",
        'unit': None
    },
    'flight':
    {
        'name': "Flight distance",
        'unit': 'mm'
    },
    'TRACK_CHI2NDOF':
    {
        'name': '$\\chi^2$ by d.o.f. of the track',
        'unit': None
    },
    'ENDVERTEX_CHI2':
    {
        'name': '$\\chi^2$ of the end vertex',
        'unit': None
    },
    'OWNPV_CHI2':
    {
        'name': '$\\chi^2$ of the primary vertex',
        'unit': None
    },
    'FDCHI2_OWNPV':
    {
        'name': '$\\chi^2$ of the flight distance',
        'unit': None
    },
    'FD_OWNPV':
    {
        'name': 'Flight distance',
        'unit': None
    },
    'IPCHI2_OWNPV':
    {
        'name': '$\\chi^2$ of the impact parameter',
        'unit': None
    },
    'IP_OWNPV':
    {
        'name': 'Impact parameter',
        'unit': None
    },
    'DIRA_OWNPV':
    {
        'name': 'Cosine of the DIRA angle',
        'unit': None
    },
    'M_Tau_Pi12pip':
    {
        'name': 'm',
        'unit': 'GeV/$c^2$'
    },
    'ENDVERTEX_CHI2,ENDVERTEX_NDOF:x/y':{
        'name': "$\chi^2$ of the end vertex per d.o.f.",
        'unit': None
    },
    'ETA':{
        'name': "$\\eta$",
        'unit': None
    },
    
    
}

# Name of the particles specified in the root file
particle_names = {
    'B0'            : '$D^{*}3\pi$',
    'Dst'           : '$D^*$',
    'D0'            : '$D^0$',
    'tau'           : '$3\pi$',
    'tau_pion0'     : '$\pi_0$',
    'tau_pion1'     : '$\pi_1$',
    'tau_pion2'     : '$\pi_2$',
    'Dst_constr_B0' : f'$D^*3\pi|m(D_s)$={Dst_M_PDG} MeV/$c^2$',
    'D0_pion'       : '$\pi$ of $D^0$',
    'D0_kaon'       : '$K$ from $D^0$',
    'Dst_pion'      : '$\pi$ from $D^*$'
}

from numpy import log
# Name of the functions and functions
functions = {
    'ln(1-x)': (lambda x: log(1.-x)),
    'ln(x)'  : (lambda x: log(x)),
    'x/y'     : (lambda x: x[0]/x[1]),
}

name_variables_functions = {
    'x/y'     : (lambda x: f'$\\frac{x[0]}{x[1]}$')
}


## NAME OF THE PARTICLES ===================================================
# used for the saved latex table as well as the table that contains the result of the fit
# which is shown inside the plot that contains the histogram and the fitted PDF

# LHCb data
name_params = {
    # Signal
    'mu'      : '$\\mu_S$',
    'sigmaL'  : '$\\sigma_{L, S}$',
    'sigmaR'  : '$\\sigma_{R, S}$',    
    'alphaL'  : '$\\alpha_{L, S}$',
    'alphaR'  : '$\\alpha_{R, S}$',
    'nL'      : '$n_{L_S}$',
    'nR'      : '$n_{R_S}$',
    'fraction': '$f_{S, \\frac{L}{R}}$',
    'n_sig'   : '$n_S$',
    'frac': '$f_{\\frac{L}{R}}$',
    # Combinatorial background
    'n_bkg'   : '$n_{B,c}$',
    'lambda'  : '$\\lambda_{B,c}$',
    # Partially reconstructed brackground
    'mu2'     : '$\\mu_{B,D^*3\\pi h}$',
    'sigma2'  : '$\\sigma_{B,D^*3\\pi h}$',
    'n_bkg2'  : '$n_{B,D^*3\\pi h}$',
    # Background decay D*Kpipi    
    'n_bkgK'  : '$n_{B,D^*K\\pi\\pi}$',
    'r_bkgK'  : '$\\frac{n_{B,D^*K\\pi\\pi}}{n_S}$',
    'n_B'     : 'B'
}

# MC
name_params_MC = {
    'mu_MC':'$\mu$',
    'muL_MC':'$\\mu_L$',
    'muR_MC':'$\\mu_R$',
    'sigma_MC':'$\sigma$',
    'sigmaL_MC':'$\\sigma_L$',
    'sigmaR_MC':'$\\sigma_R$',
    'alphaL_MC':'$\\alpha_L$',
    'alphaR_MC':'$\\alpha_R$',
    'nL_MC':'$n_L$',
    'nR_MC':'$n_R$',
    'frac_MC': '$f_{\\frac{L}{R}}$'
}

# B0 -> D* K pi pi RapidSim sample
name_params_KPiPi = {
    'muL':'$\\mu_L$',
    'sigmaL':'$\\sigma_L$',
    'muR':'$\\mu_R$',
    'sigmaR':'$\\sigma_R$',
    'alphaL':'$\\alpha_L$',
    'alphaR':'$\\alpha_R$',
    'nL':'$n_L$',
    'nR':'$n_R$',
    'fraction': '$f_{\\frac{L}{R}}$',
    'frac': '$f_{\\frac{L}{R}}$'
}

# B0 -> D* D_s component
# name_params_B0toDstDs = {
#     # Signal
#     f'mu_B0Ds'      : '$\mu_S$',
#     f'sigma_B0Ds'   : '$\\sigma_S$',
#     f'n_sig_B0Ds'   : '$n_S$',
#     # Combinatorial background
#     f'n_bkg_B0Ds'   : '$n_{B,c}$',
#     f'lambda_B0Ds'  : '$\\lambda_{B,c}$',
#     # Partially reconstructed background
#     f'mu2_B0Ds'     : '$\\mu_{B, D^* D_s h}$',
#     f'sigma2_B0Ds'  : '$\\sigma_{B, D^* D_s h}$',
#     f'n_bkg2_B0Ds'  : '$n_{B, D^* D_s h}$',
# }
name_params_B0toDstDs = {
    # Signal
    f'mu_DstDs'      : '$\mu_S$',
    f'sigma_DstDs'   : '$\\sigma_S$',
    f'n_sig_DstDs'   : '$n_S$',
    # Combinatorial background
    f'n_bkg_DstDs'   : '$n_{B,c}$',
    f'lambda_DstDs'  : '$\\lambda_{B,c}$',
    # Partially reconstructed background
    f'mu2_DstDs'     : '$\\mu_{B, D^* D_s h}$',
    f'sigma2_DstDs'  : '$\\sigma_{B, D^* D_s h}$',
    f'n_bkg2_DstDs'  : '$n_{B, D^* D_s h}$',
}


# D_s -> 3pi component
name_params_tau = {
    # Signal
    f'mu_tau':'$\mu_{S}$',
    f'sigma_tau':'$\sigma_{S}$',
    f'n_sig_tau':'$n_{S}$',
    # Combinatorial background
    f'n_bkg_tau':'$n_{bkg,c}$',
    f'lambda_tau':'$\\lambda_{bkg,c}$',
}


