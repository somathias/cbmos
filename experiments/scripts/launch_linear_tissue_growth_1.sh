#!/bin/sh
#SBATCH -A snic2021-22-607 -t 54:00:00 -p core -n 2    

python3 exp-linear_tissue_growth_script.py ../data/20220507_rackham 16 1.0
python3 exp-linear_tissue_growth_script.py ../data/20220507_rackham 15 1.0
python3 exp-linear_tissue_growth_script.py ../data/20220507_rackham 14 1.0
python3 exp-linear_tissue_growth_script.py ../data/20220507_rackham 13 1.0
