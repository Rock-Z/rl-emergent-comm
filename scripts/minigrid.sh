#! /bin/bash

#SBATCH --nodes=1
#SBATCH --partition gpu
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB


date

uv run train_minigrid.py
uv run train_minigrid.py --lr=0.0001
uv run train_minigrid.py --lr=0.0002

uv run train_minigrid.py --gamma=0.99
uv run train_minigrid.py --lr=0.0001 --gamma=0.99
uv run train_minigrid.py --lr=0.0002 --gamma=0.99

uv run train_minigrid.py --entropy-coef=0.1
uv run train_minigrid.py --lr=0.0001 --entropy-coef=0.1
uv run train_minigrid.py --lr=0.0002 --entropy-coef=0.1
