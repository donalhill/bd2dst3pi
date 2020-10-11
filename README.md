# Code for the analysis of B0 -> D* 3pi decays

This GitHub repository houses code for the analysis of B0 -> D* 3pi decays using LHCb data and MC. This decay is used as a normalisation mode in B0 -> D* (tau -> 3pi nu) nu analyses.

## Setting up

To clone this project, do this once you have logged into `lxplus`:
```
git clone https://github.com/donalrinho/bd2dst3pi.git
```
Then we need to set up a Python environment with all of the packages we require. To do this:
```
cd bd2dst3pi
source setup/setup_env.sh
```
which will install a Conda environment called `bd2dst3pi_env`. You will be placed inside this env after the process completes. To leave the env at any time, do:
```
conda deactivate
```
and to re-enter the env, do:
```
source setup/setup.sh
```

## Some background information on the B0 -> D* 3pi decay 

A hot topic in the study of B-meson decays is the comparison of decays involving heavy tau leptons and lighter leptons (muon, electron). We can measure how often decays like B0 -> D* tau nu occur compared to how often B0 -> D* mu nu decays occur. The ratio of branching fractions for these two decays, `R(D*)`, is very precisely predicted in the Standard Model (SM). But measurements of `R(D*)` at LHCb and the B-factories are consistently higher than the SM prediction. This could be evidence of New Physics, or experimental systematic effects! 

To understand better, we want to measure `R(D*)` on a larger amount of LHCb data. To study B0 -> D* tau nu decays, we can look at cases where the tau decays to three pions and a neutrino, tau -> 3pi nu. The experimental signature of this decay is the production of a D* at the B decay vertex, with three pions originiating from a vertex disaplced from the B vertex (since the tau flies before it decays).

When measuring decays like this, it is important to have a reference decay to compare to; we call this a `normalisation mode`. This helps us to cancel out a lot of systematic uncertainties in the measurement, which come from imperfect knowledge of the detector and our modelling of the detector and physics in the MC simulations. The normalisation mode we use for B0 -> D* (tau -> 3pi nu) nu is `B0 -> D* 3pi`. This decay involves the same visible particles as the B0 -> D* tau nu decay (a D* and three pions) , but does not involve any non-detected neutrinos. We call such decays `fully reconstructed`, since all of the particles produced in the decay are measured in the detector. As such, we should be able to see a nice peak at the B0 mass when we combine a D* and three pions together.

## Project objectives

In this project, we will study LHCb data and MC of the B0 -> D* 3pi decay, which will be helpful in studies of the B0 -> D* tau nu signal in measurements of `R(D*)`. We will:
 
 - learn how to work with Jupyter notebooks for interactive analysis in Python.
 
 - learn how to load `ROOT` files using the `root_pandas` package, putting the data into `pandas` DataFrame format.
 
 - learn how to calculate new variables in `pandas`, to study the flight distance of the tau candidate before it decays to three pions.
 
 - plot variable distributions in B0 -> D* 3pi data and MC to compare them, making use of the `pandas` and `matplotlib` Python packages.
 
 - perform an invariant mass fit to the B0 -> D* 3pi MC sample using the `zfit` package.
 
 - perform a fit to the B0 -> D* 3pi decays in data, using our knowledge of `zfit` from the MC fit. The objective of the fit is to measure the `normalsiation yield`, so the total number of B0 -> D* 3pi decays in the peak. In the data fit, we will import some shape parameters measured in the B0 -> D* 3pi MC fit to help constrain the peak shape. To do this, we will learn how to store useful code output into `JSON` files, which can be read back into other scripts later. 
 
 - compare the B0 -> D* 3pi and B0 -> D* tau nu MC sample variable distributions, to understand some of the key similarities and differences between them. 
 
 - learn how to write useful output numbers of the analysis into `LaTeX` tables, which can be used in your documentation. In particular, the normalisation yield measured in the data fit can be used in the measurement of `R(D*)`, so it is important to persist this to `JSON` format and also in `LaTeX` format for your report.
 
## Useful links
 
 - LHCb measurement of `R(D*)` using 2011 and 2012 data [here](https://arxiv.org/abs/1708.08856)
 
 - `zfit` documentation [here](https://github.com/zfit/zfit)
 
 - Python resources for High Energy Physics [here](https://github.com/hsf-training/PyHEP-resources)
 
 - HSF analysis essentials tutorial [here](https://hsf-training.github.io/analysis-essentials/)
 
## Location of data and MC files

We will be working with 2015 and 2016 data and MC. The data are stored on the EOS filesystem at CERN. The data can be found at:
```
/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples/norm/data
```
and the MC at:
```
/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples/norm/Bd_Dst3pi
```
In these folders, you will find files for different years (`2015` and `2016`) and magnet polarities (`up` and `down`).
 
