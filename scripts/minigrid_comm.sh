#! /bin/bash

#SBATCH --nodes=1
#SBATCH --partition gpu
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB


date

uv run train_minigrid_comm.py --max_len=2 --vocab_size=10 --batch_size=10 --n_epochs=1000 --episodes_per_epoch=10 --stats_freq=1  --lr=0.001