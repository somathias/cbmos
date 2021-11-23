#!/bin/sh
#SBATCH -A snic2021-22-607 -t 12:00:00 -p core -n 4    
  
pip3 install .

python3 exp-growth_benchmark.py growth_benchmark_0.3_3_2.0_nmax_1000 0.3 3 2.0
