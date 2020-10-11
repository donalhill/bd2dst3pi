#!/bin/bash

shopt -s expand_aliases

## Define root variable
export ANAROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

set +u

source ~/miniconda/etc/profile.d/conda.sh
conda activate bd2dst3pi_env

cd $ANAROOT

export PYTHONPATH=$ANAROOT:$ANAROOT/python:$PYTHONPATH
