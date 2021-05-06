#!/bin/sh
#SBATCH -A snic2019-8-227 -t 4:00:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable

singularity run --nv ../nobackup/private/container.sif -c "conda activate gpulibs && pip install . && python3 exp-proliferation_benchmark.py proliferation_benchmark_snowy_exact_0.1_1.0.json 0.1 1.0"
