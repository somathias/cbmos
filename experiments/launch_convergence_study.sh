#!/bin/sh
#SBATCH -A snic2019-8-227 -t 6:00:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable
  

singularity run --nv ../nobackup/private/container.sif -c "conda activate gpulibs && pip install . && python3 exp-timed_convergence_study.py timed_convergence_gpu_snowy_larger_dt.json"

  
#pip3 install .

#python3 exp-timed_convergence_study.py timed_convergence_snowy.json


