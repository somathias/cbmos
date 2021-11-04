#!/bin/sh
#SBATCH -A snic2021-22-607 -t 24:00:00 -p core -n 4    

  
pip3 install .

python3 exp-growth_benchmark.py growth_benchmark_0.3_2_np_600.json 0.3 2 False
