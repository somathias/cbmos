#!/bin/sh
#SBATCH -A snic2021-22-607 -t 24:00:00 -p core -n 2    

python3 exp-linear_tissue_growth_run_MRFE_and_fixed.py ../data/20220509_rackham 13 5.0
