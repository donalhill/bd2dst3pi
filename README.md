# Summary of the notebooks

The notebooks are in the folder `notebooks`:
- `1_Fit/CONSTRAINED`: 
    - `MC`: Fit to the $m(D^*)$-constrained $m(D^*3\pi)$ invariant mass for the MC
    - `Kpipi`: Fit to the $m(D^*)$-constrained $m(D^*3\pi)$ invariant mass for the $B^0\to D^* K \pi \pi$ sample
    - `COMMON`: 
        1. Fit to $m(3\pi)$ around $m(D_s)$ and computation of the sWeights associated with $D_s\to3\pi$
        2. Fit to the sWeighted $m(D^*)$-constrained $m(D^*D_s)$ distribution
        3. Fit to the $m(D^*)$-constrained $m(D^*3\pi)$ invariant mass for the LHCb data.
        4. Compute the $_s$Weights associated with the signal peak in the $m(D^*3\pi)$ spectrum
- `2_BDT`: 
    1. Choose the background sample (i.e., the high mass cut on the $m(D^*)$-constrained $m(D^*3\pi)$)
    2. Compare the $B^0 \to D^{*-} 3\pi$ $_s$Weighted distributions of the variables used in the training with the MC distributions. Since the MC is used as a proxy of the LHCb signal, the training variables must agree.
    3. Compare the distributions of the variables used in the training in the background and signal samples.
    4. Training, test and apply the BDTs.
    5. As a check, compare the $_s$Weighted BDT output to the BDT output in MC. We see that the $_s$Weights are imperfect and shift the BDT ouput of the signal to lower values.
- `3_optimisation_BDT`:
    1. Fits to the `common` data for different BDT cuts
    2. Minimise the relative uncertainty $\frac{\Delta S}{S}$ to choose the optimal BDT cut
    3. Shows the fit in the `common` data for the optimal BDT cut
- `4_systematic_uncertainties`:
    1. Compute $n_{B^0\to D^*3\pi}$ when changing the fixed parameters of the fit to $m(D^*3\pi)$, where the optimal BDT cut is applied, within their uncertainty interval.
        - **Mode 1**: The fixed (=tail) parameters of the $D^*\pi\pi\pi$ are randomly changed within their uncertainty.
        - **Mode 2**: The same is done with the ratio of yield $\frac{n_{B^0 \to D^{*-} K^+ \pi^+ \pi^-}}{n_{B^0 \to D^{*-\pi^+\pi^+\pi^-}}}$ and the parameters of the $B^0 \to D^{*-} K^+ \pi^+ \pi^-$ distribution
        - **Mode 3**: The same is done with the three parameters of the sWeighted $B^0 \to D^{*-} (D_s^+ \to 3\pi)$ distribution
    2. Compute the systematic uncertainty associated with the fixed parameters, which is the quadratic sum of the standard deviation of the three $n_{sig}$ distributions obtained in the 3 modes
   
# Summary of the scripts


The notebooks use the functions defined in the scripts, located in the the folder `scripts`:
- `variables.py`: contains global variables, such as the "latex" names of the variables and particles.
- `load_save_data.py`:
    - Load and save root/json/pickle files
    - Apply cuts on some variables, add the $m(D^*)$-constrained $m(D^*3\pi)$ variable
    - Get a json latex tables from the parameters that are saved in a json file
- `fit.py`:
    - Define some PDFs
    - Define the parameters from a dictionnary of training variables
    - Perform a fit (using zfit)
    - Check if the fit has an error (if it did not converge, has not a positive-definite hesse, ...)
    - Save the result of the fit in a JSON file
- `BDT/BDT.py/`
    - Plot the super-imposed signal and background distributions of the training variables
    - Plot the correlation matrix of the training variables
    - Prepare the signal and background sample in order to be used for the training
    - Train a gradient or adaboost classifier
    - Test the BDT: ROC curve, over-traning plots, Kolmogorov-Smirnov test
    - Apply the BDT to LHCb data and save the ouput
- `plot/tool.py/`
    - Functions to change what the plot looks like (grid, logscale, 'LHCb preliminary' test, ...)
    - Save a plot
    - Functions to retrieve the name and unit of the variable (for the plot), from `variables.py`. This is used in the legend of the plots.
    - Some other tool functions useful to plot (for instance, remove the latex syntax of a string in order to use it in the name of the saved file, etc.)
- `plot/histogram.py`:
    - Plot 1D and 2D histograms (with the correct label for the axes)
    - Plot scatter plots
    - Plot histograms of the quotient of 2 variables (with the correct label for the axes)
- `plot/fit.py`:
    - Plot the distribution of a variable together with his fitted PDF and the pull histogram
    - Compute the number of degrees of freedom of a model only from the zfit model
    - Compute the reduced chi2 of a model
- `plot/line.py`: plot y vs x, or y1, y2, ... vs x (i.e., several curves as a function of x)


__________________________________________

# Code for the analysis of B0 -> D* 3pi decays

This GitHub repository houses code for the analysis of B0 -> D* 3pi decays using LHCb data and MC. This decay is used as a normalisation mode in B0 -> D* (tau -> 3pi nu) nu analyses.

## Setting up

To clone this project, do this once you have logged into `lxplus`:
```bash
git clone https://github.com/donalrinho/bd2dst3pi.git
```
Then we need to set up a Python environment with all of the packages we require. To do this:
```bash
cd bd2dst3pi
source setup/setup_env.sh
```
which will install a Conda environment called `bd2dst3pi_env`. You will be placed inside this env after the process completes. To leave the env at any time, do:
```bash
conda deactivate
```
and to re-enter the env, do:
```bash
source setup/setup.sh
```

## Working on your own branch in Git

For adding your own code and output plots/tables e.t.c., it is best for you to switch to your own Git branch. To do this:
```bash
git checkout -b your-new-branch
```
where `your-new-branch` is the name you give the branch. Then if you make a new notebook, for example, you can add it with:
```bash
git add notebooks/my_awesome_analysis.ipynb
git commit -m "Adding my new great results."
git push origin your-new-branch
```
This helps you to keep your own personal copy of the `bd2dst3pi` project. And if we need to, your changes can be merged into the master version with a `merge request`.

## Using Jupyter notebooks

The Conda env we installed above comes with a full ROOT install, so it is possible to write code to analyse the data in ROOT C++ or PyROOT. For interactive analysis, it is nice to work in [Jupyter notebooks](https://jupyter.org/). With these notebooks, you can combine code blocks with documentation (even inclduing LaTeX maths) to explain what various steps are doing. This is a good way to learn and also to explain your work to others. The notbooks also render any plots you make interactively, so you can see your output directly in your browser.

Because we are working with files at CERN, our notebooks need to live on `lxplus`. This is why we have cloned the project above into `lxplus`. However, with a couple of steps, it is possible to use a web browser on our own machine (laptop/desktop) to veiw the notebooks.


The first step is to add this function to your `~/.bashrc` file on `lxplus`:
```bash
function jpt(){
    # Fires-up a Jupyter notebook by supplying a specific port
    jupyter notebook --no-browser --port=$1
}

export -f jpt
```
This function can then be called from the terminal in `lxplus`, where you supply a port number like this:
```bash
ssh -Y jsmith@lxplus706.cern.ch
source .bashrc
cd /afs/cern.ch/user/j/jsmith/bd2dst3pi
source setup/setup.sh
jpt 8889
```
Note that we have done this on a specific machine `706` on `lxplus`. The next step is to access this port from our own local machine (laptop/desktop). This allows our local machine to "listen" to the remote `lxplus` machine. To do this, we add a function to the `.bashrc` (`.bash_profile` on a Mac) of our local machine:
```bash
function jptt(){
    # Forwards port $1 into port $2 and listens to it
    ssh -N -f -L localhost:$2:localhost:$1 jsmith@lxplus706.cern.ch
}

export -f jptt
```
Here, the function we have defined points to the specific machine `706` where we called the `jpt` command on `lxplus`. So just make sure you always use `706` when doing your `jpt` command.

Now we  run the following command from a terminal in our local machine:
```bash
jptt 8889 8888
```
Note that the first port number mathces the one we specified on `lxplus` above. We then choose a different port for your loacl machine. When you run this command, you will be asked to put in your CERN user account password (the one you use to login to `lxplus`).

The final step is to type this into your local web browser:
```bash
localhost:8888
```
which should launch the notbook browser. In your browser, you will need to put in a password, which you will find in the screen output in your `lxplus` session. It will look something like this:
```
[I 19:49:51.138 NotebookApp] Loading IPython parallel extension
[I 19:49:52.187 NotebookApp] JupyterLab extension loaded from /afs/cern.ch/work/d/dhill/miniconda/envs/bd2dst3pi_env/lib/python3.7/site-packages/jupyterlab
[I 19:49:52.188 NotebookApp] JupyterLab application directory is /afs/cern.ch/work/d/dhill/miniconda/envs/bd2dst3pi_env/share/jupyter/lab
[I 19:49:52.192 NotebookApp] Serving notebooks from local directory: /afs/cern.ch/user/d/dhill/bd2dst3pi
[I 19:49:52.192 NotebookApp] Jupyter Notebook 6.1.4 is running at:
[I 19:49:52.192 NotebookApp] http://localhost:8889/?token=31767d06a5338b4d84d092c9e93d02291b543e1bb042ef6d
[I 19:49:52.192 NotebookApp]  or http://127.0.0.1:8889/?token=31767d06a5338b4d84d092c9e93d02291b543e1bb042ef6d
[I 19:49:52.192 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```
The password you need comes after the `token=` command on the sixth line. In this example, `31767d06a5338b4d84d092c9e93d02291b543e1bb042ef6d` is the password.

**For any subsequent times you launch a notebook, you will just need to do these steps:**

From an `lxplus` terminal:
```bash
source .bashrc
cd /afs/cern.ch/user/j/jsmith/bd2dst3pi
source setup/setup.sh
jpt 8889
```
From local machines terminal:
```bash
jppt 8888
```
In your web browser:
```bash
localhost:8888
```

## Some background information on the B0 -> D* 3pi decay 

A hot topic in the study of B-meson decays is the comparison of decays involving heavy tau leptons and lighter leptons (muon, electron). We can measure how often decays like B0 -> D* tau nu occur compared to how often B0 -> D* mu nu decays occur. The ratio of branching fractions for these two decays, `R(D*)`, is very precisely predicted in the Standard Model (SM). But measurements of `R(D*)` at LHCb and the B-factories are consistently higher than the SM prediction. This could be evidence of New Physics, or experimental systematic effects! 

To understand better, we want to measure `R(D*)` on a larger amount of LHCb data. To study B0 -> D* tau nu decays, we can look at cases where the tau decays to three pions and a neutrino, tau -> 3pi nu. The experimental signature of this decay is the production of a D* at the B decay vertex, with three pions originiating from a vertex disaplced from the B vertex (since the tau flies before it decays).

When measuring decays like this, it is important to have a reference decay to compare to; we call this a `normalisation mode`. This helps us to cancel out a lot of systematic uncertainties in the measurement, which come from imperfect knowledge of the detector and our modelling of the detector and physics in the MC simulations. The normalisation mode we use for B0 -> D* (tau -> 3pi nu) nu is `B0 -> D* 3pi`. This decay involves the same visible particles as the B0 -> D* tau nu decay (a D* and three pions) , but does not involve any non-detected neutrinos. We call such decays `fully reconstructed`, since all of the particles produced in the decay are measured in the detector. As such, we should be able to see a nice peak at the B0 mass when we combine a D* and three pions together.

## Project objectives

In this project, we will study LHCb data and MC of the B0 -> D* 3pi decay, which will be helpful in studies of the B0 -> D* tau nu signal in measurements of `R(D*)`. We will:
 
 - learn how to work with Jupyter notebooks for interactive analysis in Python.
 
 - learn how to load `ROOT` files using the `root_pandas` package, putting the data into `pandas` DataFrame format.
 
 - learn how to calculate new variables in `pandas`.
 
 - plot variable distributions in B0 -> D* 3pi data and MC to compare them, making use of the `pandas` and `matplotlib` Python packages.
 
 - perform an invariant mass fit to the B0 -> D* 3pi MC sample using the `zfit` package.
 
 - perform a fit to the B0 -> D* 3pi decays in data, using our knowledge of `zfit` from the MC fit. The objective of the fit is to measure the `normalsiation yield` i.e. the total number of B0 -> D* 3pi decays in the peak. In the data fit, we will import some shape parameters measured in the B0 -> D* 3pi MC fit to help constrain the peak shape. To do this, we will learn how to store useful code output into `JSON` files, which can be read back into other scripts later. 
 
 - perform a BDT selection using `scikit-learn` to distinguish between B0 -> D* 3pi signal events and combinatorial (random combinations) background.
 
 - learn how to write useful output numbers of the analysis into `LaTeX` tables, which can be used in your documentation. In particular, the normalisation yield measured in the data fit can be used in the measurement of `R(D*)`, so it is important to persist this to `JSON` format and also in `LaTeX` format for your report.
 
## Useful links
 
 - LHCb measurement of `R(D*)` using 2011 and 2012 data [here](https://arxiv.org/abs/1708.08856)
 
 - `zfit` documentation [here](https://github.com/zfit/zfit)
 
 - Python resources for High Energy Physics [here](https://github.com/hsf-training/PyHEP-resources)
 
 - HSF analysis essentials tutorial [here](https://hsf-training.github.io/analysis-essentials/)
 
## Location of data and MC files

We will be working with 2015 and 2016 data and MC. The data are stored on the EOS filesystem at CERN. The files can be found here:
```
/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples/
```
The fully selected normalisation mode data can be found in this sub-directory:
```
/norm/data
```
and the fully selected MC is here:
```
/norm/Bd_Dst3pi
```
In these folders, you will find files for different years (`2015` and `2016`) and magnet polarities (`up` and `down`).

## Some useful paths

In the `bd2dst3pi/bd2dst3pi/locations.py` script, some useful shortcuts are defined:
```python
loc.ROOT = repo+'/'
loc.OUT = loc.ROOT+'output/'
loc.PLOTS = loc.OUT+'plots'
loc.TABLES = loc.OUT+'tables'
loc.JSON = loc.OUT+'json'
loc.EOS = '/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples'
loc.DATA = loc.EOS+'/norm/data'
loc.MC = loc.EOS+'/norm/Bd_Dst3pi'
```
You can use these inside notebooks by doing:
```python
from fcc_python_tools.locations import loc
```
and then `loc.DATA` for example to get the path to the data files. You can add to this list if you like. You will notice some paths like `loc.JSON`, where you should store any `JSON` files you make, and `loc.PLOTS` where you should put the plots. This helps to keep all of your output organised nicely, and means you only define your paths in a single place.

## Example notebooks to get you started

All of the example notebooks are in the `bd2dst3pi/notebooks/0.Tutorial/` folder. Once you have launched your Jupyter session in your browser following the instuctions above, you can go into the `notebooks` folder and click on an example notebook. When you start making your own notebooks, you can add them into this folder as well.

To make a new notebook, you can click on the `New` button to the right of your Jupyter window. In the menu shown, select `bd2dst3pi_env` to choose the analysis Conda env we made above.

The example notebooks cover different aspects of data processing and fitting, to help you become familiar with some of the tools. They are numbered 1-5, so you can work through them in a logical order. In each notebook, there are some **follow-up** tasks to complete, which will help you to try out the tools yourself and understand things better. You can either edit existing notebooks to do more things, or make new notebooks.
