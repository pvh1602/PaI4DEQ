cd ..

for pruner in snip
    do 
    for sparsity in 0.7 0.99 0.995
        do
            CUDA_VISIBLE_DEVICES=3 python run_exps.py --model MultiConvNet --pruner $pruner --sparsity $sparsity --dataset cifar10 --is_wandb --is_pruning
        done
    done
