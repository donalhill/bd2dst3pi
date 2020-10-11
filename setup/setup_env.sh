export ANAROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

wget -nv http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /afs/cern.ch/work/${USER:0:1}/$USER/miniconda
source /afs/cern.ch/work/${USER:0:1}/$USER/miniconda/etc/profile.d/conda.sh
conda create -n bd2dst3pi_env python=3.7 root -c conda-forge
conda activate bd2dst3pi_env
conda config --env --add channels conda-forge
#Install required packages
conda install -y root_pandas iminuit tensorflow uncertainties matplotlib flake8
pip install snakemake
pip install zfit
pip install progressbar
pip install num2tex
pip install uproot4 awkward1
pip install tqdm
pip install hepstats
pip install jupyterlab
pip install notebook
#hepvecctor which is not released in conda or pip
cd $ANAROOT
git clone git://github.com/henryiii/hepvector
cd hepvector
python setup.py install
cd ..
python -m ipykernel install --user --name=bd2dst3pi_env
