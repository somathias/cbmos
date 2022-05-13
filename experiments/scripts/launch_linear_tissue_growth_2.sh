#!/bin/sh
#SBATCH -A snic2021-22-607 -t 48:00:00 -p core -n 2    

python3 exp-linear_tissue_growth_script.py ../data/20220506_rackham 13 2

