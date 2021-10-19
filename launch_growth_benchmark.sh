#!/bin/sh
#SBATCH -A snic2019-8-227 -t 6:00:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable
  

singularity run --nv ../nobackup/private/container.sif -c "conda activate gpulibs && pip install . && python3 exp-growth_benchmark.py growth_benchmark_0.3_2_cp.json 0.3 2 True"

  
#pip3 install .

#python3 exp-growth_benchmark.py growth_benchmark_0.3_2_np.json 0.3 2 False
