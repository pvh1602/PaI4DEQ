
cd ../..

CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 0 --dataset cifar10 --model MultiConvNet --sp PR --is_pruning --is_wandb --seed 2 --lr 0.01
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 1 --dataset cifar10 --model MultiConvNet --sp PR --is_pruning --is_wandb --seed 2 --lr 0.01
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 2 --dataset cifar10 --model MultiConvNet --sp PR --is_pruning --is_wandb --seed 2 --lr 0.01
CUDA_VISIBLE_DEVICES=3 python run_exps.py --is_ablation --ablation_mode 3 --dataset cifar10 --model MultiConvNet --sp PR --is_pruning --is_wandb --seed 2 --lr 0.01