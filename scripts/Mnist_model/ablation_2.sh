
cd ../..

CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 0 --dataset mnist --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2022 --lr 0.001
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 1 --dataset mnist --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2022 --lr 0.001
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 2 --dataset mnist --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2022 --lr 0.001
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 3 --dataset mnist --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2022 --lr 0.001