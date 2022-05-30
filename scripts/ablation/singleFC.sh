cd ../..

CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 0 --dataset mnist --model SingleFcNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 1 --dataset mnist --model SingleFcNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 2 --dataset mnist --model SingleFcNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 3 --dataset mnist --model SingleFcNet --sp PR --is_pruning --is_wandb