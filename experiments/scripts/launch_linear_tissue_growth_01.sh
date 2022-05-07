#!/bin/sh
#SBATCH -A snic2021-22-607 -t 24:00:00 -p core -n 2    

python3 exp-linear_tissue_growth_script.py ../data/20220506_rackham 16 0.1
python3 exp-linear_tissue_growth_script.py ../data/20220506_rackham 15 0.1
python3 exp-linear_tissue_growth_script.py ../data/20220506_rackham 14 0.1
python3 exp-linear_tissue_growth_script.py ../data/20220506_rackham 13 0.1
