cd ..

for pruner in snip
    do 
    for sparsity in 0.9 0.95 0.995
        do
            CUDA_VISIBLE_DEVICES=5 python run_exps.py --model SingleConvNet --pruner $pruner --sparsity $sparsity --dataset cifar10 --is_wandb --is_pruning
        done
    done
