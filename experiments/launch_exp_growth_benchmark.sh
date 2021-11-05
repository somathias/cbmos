#!/bin/sh
#SBATCH -A snic2021-22-607 -t 12:00:00 -p core -n 4    
  
pip3 install .

python3 exp-exp_growth_benchmark.py exp_growth_benchmark_0.3_3_nmax_1000 0.3 3 False
