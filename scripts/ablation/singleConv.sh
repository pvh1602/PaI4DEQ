cd ../..

CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 0 --dataset cifar10 --model SingleConvNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 1 --dataset cifar10 --model SingleConvNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 2 --dataset cifar10 --model SingleConvNet --sp PR --is_pruning --is_wandb
CUDA_VISIBLE_DEVICES=2 python run_exps.py --is_ablation --ablation_mode 3 --dataset cifar10 --model SingleConvNet --sp PR --is_pruning --is_wandb