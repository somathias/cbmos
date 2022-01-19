#!/bin/sh
#SBATCH -A snic2021-22-607 -t 2:00:00 -p core -n 4    
  
pip3 install ../../.

python3 exp-locally_compressed_spheroid_benchmark.py exp_locally_compressed_spheroid_benchmark_s0 0
