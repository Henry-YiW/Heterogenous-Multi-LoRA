#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdpp-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# srun python gen_img_prompt_pairs.py --lora_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool/reality'
srun python gen_img_prompt_pairs.py --lora_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool/anime' --category 'anime'