#!/bin/sh
#SBATCH -A uppmax2021-2-24 -t 12:00:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable    

singularity run --nv ../nobackup/private/container.sif -c "conda activate gpulibs && pip install . && python3 exp-growth_benchmark_snowy.py growth_benchmark_0.3_3_nmax_2000_snowy 0.3 3 True"  


