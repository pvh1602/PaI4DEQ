
cd ../..

CUDA_VISIBLE_DEVICES=0 python run_exps.py --is_ablation --ablation_mode 0 --dataset shvn --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2  --lr 0.001 --lr_drop_step 25
CUDA_VISIBLE_DEVICES=0 python run_exps.py --is_ablation --ablation_mode 1 --dataset shvn --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2  --lr 0.001 --lr_drop_step 25
CUDA_VISIBLE_DEVICES=0 python run_exps.py --is_ablation --ablation_mode 2 --dataset shvn --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2  --lr 0.001 --lr_drop_step 25
CUDA_VISIBLE_DEVICES=0 python run_exps.py --is_ablation --ablation_mode 3 --dataset shvn --model SingleConvNet --sp PR --is_pruning --is_wandb --seed 2  --lr 0.001 --lr_drop_step 25