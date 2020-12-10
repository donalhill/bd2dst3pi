Dst_M_PDG = 2010.26

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
        'unit': "$\mu m$"
    },
    'flight_zsig':
    {
        'name': "$\Delta z$ significance",
        'unit': None
    },
    'flight':
    {
        'name': "$\Delta z$",
        'unit': "$\mu m$"
    },
    'flight':
    {
        'name': "flight distance",
        'unit': '$\mu$m'
    },
    'TRACK_CHI2NDOF':
    {
        'name': '$\\chi^2$ by d.o.f. of the track',
        'unit': None
    },
    'ENDVERTEX_CHI2':
    {
        'name': '$\\chi^2$ of the endvertex',
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
    'IPCHI2_OWNPV':
    {
        'name': '$\\chi^2$ of the impact parameter',
        'unit': None
    },
    'DIRA_OWNPV':
    {
        'name': 'Dira angle',
        'unit': None
    },
    'M_Tau_Pi12pip':
    {
        'name': 'm',
        'unit': 'GeV/$c^2$'
    },
    
}

particle_names = {
    'B0'            : '$D^{*}3\pi$',
    'Dst'           : '$D^*$',
    'D0'            : '$D^0$',
    'tau'           : '$3\pi$',
    'tau_pion0'     : '$\pi_0$',
    'tau_pion1'     : '$\pi_1$',
    'tau_pion2'     : '$\pi_2$',
    'Dst_constr_B0' : f'$B^0|m(D_s)$={Dst_M_PDG} MeV/$c^2$'
}


name_params = {
    # Signal
    'mu'      : '$\\mu_S$',
    'sigmaL'  : '$\\sigma_{L_S}$',
    'sigmaR'  : '$\\sigma_{R_S}$',    
    'alphaL'  : '$\\alpha_{L_S}$',
    'alphaR'  : '$\alpha_{R_S}$',
    'nL'      : '$n_L_S$',
    'nR'      : '$n_R_S$',
    'fraction': '$f_{S, \\frac{L}{R}}$',
    'n_sig'   : '$n_S$',
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
}
name_params_Kpipi = {
    'muL':'$\\mu_L$',
    'sigmaL':'$\\sigma_L$',
    'muR':'$\\mu_R$',
    'sigmaR':'$\\sigma_R$',
    'alphaL':'$\\alpha_L$',
    'alphaR':'$\\alpha_R$',
    'nL':'$n_L$',
    'nR':'$n_R$',
    'fraction': '$f_{\\frac{L}{R}}$'
}




