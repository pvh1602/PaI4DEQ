cd ..

for pruner in snip
    do 
    for sparsity in 0.1 0.2 0.5
        do
            CUDA_VISIBLE_DEVICES=1 python run_exps.py --model MultiConvNet --pruner $pruner --sparsity $sparsity --dataset cifar10 --is_wandb --is_pruning
        done
    done
