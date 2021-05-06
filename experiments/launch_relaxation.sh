#!/bin/sh
#SBATCH -A snic2019-8-227 -t 4:00:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable

singularity run --nv ../nobackup/private/container.sif -c "conda activate gpulibs && pip install . && python3 exp-relaxation_benchmark.py relaxation_benchmark_snowy.json"
