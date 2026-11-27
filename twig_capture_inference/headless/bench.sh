#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --account=wilson
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:45:00
#SBATCH -o /home/ab714/jobs/browser_bench/bench_%j.log
nvidia-smi -L
export PATH=$HOME/.nvm/versions/node/v24.15.0/bin:$PATH
cd "${BENCH_DIR:-$HOME/jobs/browser_bench}"  # must hold node_modules/playwright-core
node bench.mjs "${WHICH:-webgpu:fp16pad,webgpu:fp32pad,webgpu:fp16,wasm:fp16}"
