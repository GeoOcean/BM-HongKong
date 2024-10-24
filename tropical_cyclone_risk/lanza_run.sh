#!/bin/bash

#PBS -N nbs_xbeach_$case
#PBS -q geocean
#PBS -l mem=15gb
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
source /software/geocean/conda/bin/activate
conda activate pablov1
python run.py