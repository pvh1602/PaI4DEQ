cd ../..

for pruner in synflow
    do 
    for sparsity in 0.1 0.5 0.7 0.9 0.95 0.97 0.99 0.999
        do
            CUDA_VISIBLE_DEVICES=0 python run_exps.py --model MultiConvNet --pruner $pruner --sparsity $sparsity --dataset cifar10 --is_wandb --seed 1 --is_pruning --lr 0.01 --prune_epochs 100
        done
    done