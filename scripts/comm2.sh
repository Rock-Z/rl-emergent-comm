#! /bin/bash

#SBATCH --nodes=1
#SBATCH --partition gpu
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB


date

uv run compo_vs_generalization/train.py --n_attributes=2 --n_values=25 --vocab_size=5 --max_len=8 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5

uv run compo_vs_generalization/train.py --n_attributes=2 --n_values=25 --vocab_size=25 --max_len=4 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5

uv run compo_vs_generalization/train.py --n_attributes=2 --n_values=25 --vocab_size=625 --max_len=2 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5

uv run compo_vs_generalization/train.py --n_attributes=3 --n_values=10 --vocab_size=10 --max_len=6 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5

uv run compo_vs_generalization/train.py --n_attributes=3 --n_values=10 --vocab_size=100 --max_len=3 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5

uv run compo_vs_generalization/train.py --n_attributes=3 --n_values=10 --vocab_size=1000000 --max_len=1 --batch_size=5120 --data_scaler=60 --n_epochs=3000 --lr=0.001 --sender_hidden=500 --receiver_hidden=500 --receiver_emb=30 --sender_emb=5
