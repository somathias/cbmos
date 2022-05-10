#!/bin/sh
#SBATCH -A snic2021-22-607 -t 24:00:00 -p core -n 2    

python3 exp-locally_compressed_spheroid_benchmark.py ../data/20220510_rackham_size13_eps0.005_m14 13 0.005 14
